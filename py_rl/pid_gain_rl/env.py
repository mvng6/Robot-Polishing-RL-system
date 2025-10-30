"""
PID Gain 최적화 환경
"""
import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

from .loggers.base_logger import AppLogger
from .loggers.control_performance import ControlPerformanceLogger
from .loggers.reward_breakdown import RewardBreakdownLogger
from .loggers.learning_done import LearningDoneLogger
from .constants import Constants
from .utils.math_utils import create_initial_state, scale_action_to_pid
from .agent import PIDGainSACAgent
from .comm import PIDGainCommunicator
from .monitor import RLRealtimeMonitor

class PIDGainOptimizationEnvironment:
    """
    PID Gain 최적화 환경
    - 에피소드 실행 및 관리
    - 보상 계산 (연속형 지수 스코어 기반)
    - 제어공학 지표 계산 및 저장 (논문용 10개 핵심 지표)
    - 데이터 수집 및 저장
    - 학습 진행 모니터링
    """

    def __init__(self, cfg=None):
        if cfg is None:
            raise ValueError("cfg 파라미터가 필요합니다")
        self.cfg = cfg
        self.agent = PIDGainSACAgent(cfg)
        self.comm = PIDGainCommunicator(
            cfg["HOST"],
            cfg["PORT"],
            cfg["RECV_TIMEOUT_SEC"],
            cfg["RECV_LOOP_TIMEOUT_SEC"],
            cfg,
        )
        self.best_episode_reward = -float("inf")
        self.best_agent_episode = -1
        self.fail_count = 0
        self.FAIL_MAX = cfg["COMM_FAIL_MAX"]
        self.last_log_time = None
        self.last_valid_state = None
        self.last_sander_active = False

        # ==== PID Gain 최적화용 변수들 ====
        self.band_tol_N = Constants.BAND_TOLERANCE_N
        self.safety_force_limit = Constants.SAFETY_FORCE_LIMIT

        # ==== 로깅 폴더 구조: learning_done_{timestamp}/ 한 곳에 모두 ====
        # 1. LearningDoneLogger가 learning_done 폴더 생성
        self.ldlogger = LearningDoneLogger(self.cfg["LOG_DIR"])

        # 2. 나머지 Logger들은 learning_done 폴더 안에 서브폴더 생성
        self.cplogger = ControlPerformanceLogger(self.ldlogger.log_dir)
        self.rlogger = RewardBreakdownLogger(self.ldlogger.log_dir)

        # ==== ADDED: PID gain 최적화용 변수들 ====
        self.episode_force_data = []  # 에피소드 동안 힘 데이터 수집
        self.episode_pi_output_data = []  # 에피소드 동안 PI 출력 데이터 수집
        self.episode_start_time = None
        self.current_pid_gains = None  # 현재 사용 중인 PID gain

        # ==== ADDED: 이전 에피소드 정보 추적 ====
        self.previous_pid_gains = None  # 이전 에피소드의 PID gain
        self.pid_gains_next = (
            None  # 다음 에피소드에 실제로 적용할 PID (미리 전송한 값)
        )
        self.historical_errors = []  # 이전 에피소드들의 에러 통계
        self.episode_count = 0  # 에피소드 카운터

        # ==== ADDED: 이전 5개 에피소드 성능 히스토리 (최적값) ====
        self.episode_history = []  # 최근 5개 에피소드의 (PID, 성능) 기록
        self.max_history = 5  # 최대 5개 에피소드 히스토리 유지 (최적값)

    def _log(self, level, message):
        AppLogger.log(level, message)


    def calculate_episode_reward(
        self, force_data, pi_output_data, target_force=None, episode_len_s=None
    ):
        """
        연속형 지수 스코어 + PBRS 기반 보상 함수 (최종 범위: [-1.0, 1.0])
        - 스코어(0~1): S_ts(정착시간), S_mp(오버슈트%), S_ess(정상상태오차), S_band(밴드유지율), S_u(입력RMS)
        - 실패 페널티: P_fail (0~1)
        - PBRS(progress): 잠재함수 개선분(>0만 반영)
        - reward_score = Σ w_i*S_i - w_fail*P_fail + w_pbrs*progress ∈ [0,1]
        - 최종 보상: reward = 2*reward_score - 1 ∈ [-1,1]
        - 안전 위반 시: reward = -1.0 (Constants.SAFETY_FORCE_PENALTY)

        Args:
            force_data: 에피소드 동안의 힘 데이터 리스트
            pi_output_data: 에피소드 동안의 PI 출력 데이터 리스트
            target_force: 목표 힘 (None이면 설정에서 가져옴)
            episode_len_s: 에피소드 길이 (None이면 설정에서 가져옴)
        Returns:
            total_reward: 에피소드 총보상 ([-1, 1])
            metrics: 성능 지표 딕셔너리
        """
        # 힘 데이터가 비어 있으면 보상 0과 빈 메트릭스 반환
        if not force_data:
            return 0.0, {}

        # ----- 동적 설정값 가져오기 -----
        if target_force is None:
            target_force = self.cfg["TARGET_FORCE"]
        if episode_len_s is None:
            episode_len_s = self.cfg["EPISODE_SECONDS"]

        # ----- 데이터/기본값 -----
        force_array = np.array(force_data, dtype=np.float64)
        n_samples = len(force_array)
        fs_hz = n_samples / max(episode_len_s, 1e-6)
        dt = 1.0 / max(fs_hz, 1e-6)
        errors = force_array - target_force
        T_abs = max(abs(target_force), 1.0)

        # ----- 핵심 지표 -----
        rmse = float(np.sqrt(np.mean(errors**2))) # 전체 오차 크기

        # 음수 타깃(-) 오버슈트: 더 음수(min)로 내려가면 overshoot
        if target_force < 0:
            extreme_force = float(np.min(force_array))
            overshoot_pct = max(0.0, (target_force - extreme_force) / T_abs)
        else:
            extreme_force = float(np.max(force_array))
            overshoot_pct = max(0.0, (extreme_force - target_force) / T_abs)

        # 밴드 유지율 (±1.5N)
        tol_main = Constants.BAND_TOLERANCE_N
        in_band = np.abs(errors) <= tol_main
        band_ratio = float(np.mean(in_band)) if n_samples > 0 else 0.0

        # 정착시간: warmup 이후에만 체크
        SHAPING_WARMUP_S = Constants.SHAPING_WARMUP_S
        start_idx = int(fs_hz * SHAPING_WARMUP_S)
        tol_settling = max(Constants.SETTLING_BAND_TOLERANCE, 0.01 * T_abs)
        hold_samples = int(fs_hz * Constants.SETTLING_HOLD_TIME_S)
        in_settling = np.abs(errors) <= tol_settling
        settling_time_s = episode_len_s
        runlen = 0
        for k in range(start_idx, n_samples):
            if in_settling[k]:
                runlen += 1
                if runlen >= hold_samples:
                    settling_time_s = max(0.0, (k - hold_samples) * dt)
                    break
            else:
                runlen = 0

        # 분산 계산
        error_var = float(np.var(errors))
        variance_n = float(min(1.0, error_var / (T_abs**2)))

        # 제어신호 품질
        u_rms_n = 0.0
        du_rms_n = 0.0
        sat_ratio = 0.0
        if pi_output_data:
            u = np.array(pi_output_data, dtype=np.float64)
            u_max = Constants.PI_OUTPUT_MAX
            if u_max <= 0:
                u_max = 1.0
            u_rms_n = float(np.sqrt(np.mean(u**2)) / u_max)
            if len(u) > 1:
                du = np.diff(u)
                du_rms_n = float(np.sqrt(np.mean(du**2)) / u_max)
            sat_threshold = Constants.PI_OUTPUT_SAT_THRESHOLD * u_max
            sat_ratio = float(np.mean(np.abs(u) >= sat_threshold))

        # ----- Potential-based shaping (개선분만) -----
        abs_e = np.abs(errors)
        skip = min(start_idx, len(abs_e))
        abs_e_eff = abs_e[skip:] if skip < len(abs_e) else abs_e
        progress = 0.0
        if len(abs_e_eff) > 1:
            phi = -abs_e_eff / T_abs
            gamma = Constants.POTENTIAL_GAMMA
            F = gamma * phi[1:] - phi[:-1]
            progress = float(np.mean(np.clip(F, 0.0, None)))

        # ----- 핵심 지표 기반 가중 보상 구성 -----
        # 정상상태 오차(ess): 마지막 10% 구간 평균 오차
        if n_samples > 0:
            tail_len = max(1, int(n_samples * 0.1))
            ess = float(np.mean(errors[-tail_len:]))
        else:
            ess = 0.0

        # 밴드 유지율(±1.5N)
        margin = np.maximum(0.0, tol_main - np.abs(errors)) / max(tol_main, 1e-6)
        band_ratio = float(np.mean(np.abs(errors) <= tol_main)) if n_samples > 0 else 0.0

        # 🎯 점진적 오버슈트 페널티 (3단계)
        overshoot_pct_abs = overshoot_pct * 100.0  # % 단위로 변환
        
        if overshoot_pct_abs <= Constants.OVERSHOOT_THRESHOLD_MILD:
            # 5% 이하: 경미 - 선형 페널티 (거의 무시)
            P_overshoot = (overshoot_pct_abs / Constants.OVERSHOOT_THRESHOLD_MILD) * 0.1
        elif overshoot_pct_abs <= Constants.OVERSHOOT_THRESHOLD_MODERATE:
            # 5~15%: 보통 - 제곱 페널티
            normalized = (overshoot_pct_abs - Constants.OVERSHOOT_THRESHOLD_MILD) / \
                        (Constants.OVERSHOOT_THRESHOLD_MODERATE - Constants.OVERSHOOT_THRESHOLD_MILD)
            P_overshoot = 0.1 + (normalized ** 2) * 0.4  # 0.1 ~ 0.5
        else:
            # 15% 이상: 심각 - 3차 페널티 (급격히 증가)
            normalized = min(1.0, (overshoot_pct_abs - Constants.OVERSHOOT_THRESHOLD_MODERATE) / \
                           (Constants.OVERSHOOT_THRESHOLD_SEVERE - Constants.OVERSHOOT_THRESHOLD_MODERATE))
            P_overshoot = 0.5 + (normalized ** 3) * 0.5  # 0.5 ~ 1.0

        # ----- 추종 실패 페널티 (연속형) -----
        # 1. RMSE 기반: 큰 오차가 지속되면 추종 실패
        rmse_fail_ratio = max(0.0, (rmse - Constants.TRACKING_FAIL_RMSE_THRESHOLD) / Constants.TRACKING_FAIL_RMSE_THRESHOLD)
        rmse_fail_ratio = min(1.0, rmse_fail_ratio)  # 0~1 클립
        
        # 2. 밴드 체류율 기반: 밴드 밖에 오래 있으면 추종 실패
        band_fail_ratio = max(0.0, (Constants.TRACKING_FAIL_BAND_RATIO - band_ratio) / Constants.TRACKING_FAIL_BAND_RATIO)
        band_fail_ratio = min(1.0, band_fail_ratio)  # 0~1 클립
        
        # 두 지표 중 더 나쁜 것을 페널티로 사용 (max)
        P_tracking_fail = max(rmse_fail_ratio, band_fail_ratio)

        # ----- 보상: 지수 스코어 + PBRS -----
        # 스코어화(0~1)
        S_ts   = float(np.exp(-float(settling_time_s) / max(Constants.SCORE_TAU_TS, 1e-6)))
        S_mp   = float(np.exp(-(overshoot_pct_abs) / max(Constants.SCORE_TAU_MP_PERCENT, 1e-6)))  # overshoot in %
        S_ess  = float(np.exp(-abs(ess) / max(Constants.SCORE_TAU_ESS_N, 1e-6)))
        S_band = float(band_ratio)
        S_u    = float(np.exp(-float(u_rms_n) / max(Constants.SCORE_TAU_U, 1e-6)))
        P_fail = P_tracking_fail

        W_PBRS = Constants.SCORE_W_PBRS
        reward_score = (
            Constants.SCORE_W_TS * S_ts
            + Constants.SCORE_W_MP * S_mp
            + Constants.SCORE_W_ESS * S_ess
            + Constants.SCORE_W_BAND * S_band
            + Constants.SCORE_W_U * S_u
            - Constants.SCORE_W_FAIL * P_fail
            + W_PBRS * progress
        )

        # 안전 위반 시 큰 페널티
        if abs(extreme_force) > Constants.SAFETY_FORCE_LIMIT:
            reward = -1.0  # [-1,1] 스케일: 최대 패널티
            print(
                f"⚠️ [안전 위반] 극한 힘: {extreme_force:.1f}N (절댓값 > {Constants.SAFETY_FORCE_LIMIT}N)"
            )
        else:
            # 최종 보상: [0,1] → [-1,1] 중앙정규화 후 약한 클립
            reward = float(np.clip(2.0 * reward_score - 1.0, -1.0, 1.0))

        # ----- 메트릭 (실제 사용되는 필수 지표만) -----
        metrics = {
            # 기본 지표 (CSV 저장 및 로그 출력용)
            "rmse": rmse,
            "overshoot": overshoot_pct * 100.0,
            "settling_time": settling_time_s,
            "band_time": band_ratio * episode_len_s,
            "out_of_band_time": (1.0 - band_ratio) * episode_len_s,
        }
        
        # 🔍 보상 및 메트릭 검증 (NaN/Inf 체크)
        if np.isnan(reward) or np.isinf(reward):
            print(f"❌ [오류] calculate_episode_reward에서 NaN/Inf 보상 발견!")
            print(f"   reward: {reward}, rmse: {rmse}, overshoot: {overshoot_pct}")
            print(f"   force_array min/max: {np.min(force_array):.2f}/{np.max(force_array):.2f}")
            reward = 0.0  # 안전한 기본값
            print(f"   → 보상을 0.0으로 대체")
        
        # 메트릭도 검증
        for key, val in metrics.items():
            if np.isnan(val) or np.isinf(val):
                print(f"❌ [오류] 메트릭 '{key}'에 NaN/Inf 발견: {val}")
                metrics[key] = 0.0 if key != "settling_time" else episode_len_s

        return reward, metrics

    def generate_episode_reward_graph(self, save_to_rlogger_folder=True):
        if not hasattr(self, "agent") or not self.agent.episode_rewards:
            self._log("WARNING", "생성할 보상 데이터가 없습니다")
            return
        try:
            episode_rewards = self.agent.episode_rewards
            episodes = list(range(1, len(episode_rewards) + 1))
            plt.figure(figsize=(12, 6))
            plt.plot(
                episodes,
                episode_rewards,
                "b-",
                linewidth=2,
                marker="o",
                markersize=4,
            )
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Episode Reward", fontsize=12)
            plt.title(
                "Episode Rewards Over Time", fontsize=14, fontweight="bold"
            )
            plt.grid(True, alpha=0.3)
            if len(episode_rewards) > 1:
                avg_reward = np.mean(episode_rewards)
                plt.axhline(
                    y=avg_reward,
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Average: {avg_reward:.2f}",
                )
                plt.legend()

            # RewardBreakdownLogger 폴더에 저장 (기본값)
            if save_to_rlogger_folder and hasattr(self, "rlogger"):
                filename = os.path.join(
                    self.rlogger.log_dir, "episode_rewards.png"
                )
            else:
                # 기존 방식 (LOG_DIR에 저장)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = (
                    f"{self.cfg['LOG_DIR']}/episode_rewards_{timestamp}.png"
                )
                os.makedirs(os.path.dirname(filename), exist_ok=True)

            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            self._log("INFO", f"📈 에피소드별 보상 그래프 저장: {filename}")
        except Exception as e:
            self._log("ERROR", f"에피소드별 보상 그래프 생성 오류: {e}")

    def is_episode_done(self, force_data, target_force):
        """
        에피소드 종료 조건 확인 (PID gain 최적화용)
        Args:
            force_data: 현재까지의 힘 데이터
            target_force: 목표 힘
        Returns:
            done: 에피소드 종료 여부
            reason: 종료 이유
        """
        if not force_data:
            return False, "no_data"

        current_force = force_data[-1]

        # 안전 위반 체크
        if current_force > self.safety_force_limit:
            self._log(
                "WARNING",
                f"안전 위반: 힘 {current_force:.1f}N > {self.safety_force_limit}N",
            )
            return True, "safety_violation"

        # 에피소드는 시간으로만 종료
        return False, "time_based"

    # ---- PID Gain 최적화용 유틸리티 메서드들 ----

    def reset_episode(self):
        """에피소드 리셋 (PID gain 최적화용)"""
        self.episode_force_data = []
        self.episode_pi_output_data = []
        self.episode_start_time = None
        self.current_pid_gains = None
        self.last_log_time = None
        self.last_valid_state = None
        self.last_sander_active = False

        # 로봇 리셋 신호 전송
        ok = self.comm.send_reset()
        if ok:
            self._log("INFO", "🔄 에피소드 리셋 완료")
        else:
            self._log("WARNING", "⚠️ 리셋 신호 전송 실패")
        return ok

    # ---- PID Gain 최적화 메인 루프 ----
    def run_pid_optimization_training(self, episodes=None):
        """
        PID gain 최적화를 위한 강화학습 메인 루프
        - 에피소드당 한 번 PID gain 선택
        - 설정된 시간 동안 해당 PID gain으로 제어
        - 에피소드 종료 후 총보상 계산 및 학습
        """
        episodes = episodes or self.cfg["EPISODES"]
        if not self.comm.connect():
            self._log("ERROR", "로봇PC 연결 실패")
            return

        model_save_dir = self.cfg["MODEL_SAVE_DIR"]
        os.makedirs(model_save_dir, exist_ok=True)

        # RL 활성화 대기
        self._log("INFO", "🔄 RL 활성화 대기 중...")
        wait_start = time.perf_counter()
        while True:
            state, sander_active = self.comm.get_latest_state()
            if sander_active:
                wait_duration = time.perf_counter() - wait_start
                print(
                    f"\r{' ' * 80}\r🎯 RL 활성화! ({wait_duration:.1f}s 대기)"
                )
                self._log("INFO", f"RL 활성화 ({wait_duration:.1f}s)")
                break
            if state is not None:
                current_force = state[0]
                print(
                    f"\r⏳ 대기 중 Force:{current_force:6.1f}N",
                    end="",
                    flush=True,
                )
            time.sleep(1.0)
            if time.perf_counter() - wait_start > 300:
                print()
                self._log("WARNING", "RL 활성화 타임아웃 (5분)")
                return

        episode_stats = []
        best_reward = -float("inf")
        best_pid_gains = None

        # =========================
        # [A] 학습 시작 직후 1회: 모니터 시작
        # =========================
        monitor = RLRealtimeMonitor(
            title="PID Gain RL Monitor",
            hz=10,  # 실시간 갱신 10 Hz
            rolling_window=30.0,  # 롤링 윈도우 30초 (에피소드 3개 분량)
        )
        monitor.start()
        print("📊 [Monitor] 실시간 모니터 시작됨 (10 Hz, 롤링 윈도우 30초)")

        # 전역 시간 기준점 (학습 시작 시간)
        training_start_time = time.perf_counter()

        for ep in range(episodes):
            # 에피소드별 리셋 없음 (연속 모니터링)

            self.episode_count = ep
            print(f"\n{'='*60}\n에피소드 {ep+1}/{episodes}")
            if ep > 0 and self.previous_pid_gains is not None:
                # 이전 에피소드 정보를 활용한 상태 추정

                initial_state = create_initial_state(
                    [],
                    self.cfg["TARGET_FORCE"],
                    self.previous_pid_gains,
                    self.episode_history,
                    self.cfg["RECV_INTERVAL_SEC"],
                )
            else:
                # 첫 에피소드: 기준 PID gain으로 상태 추정
                print("🆕 첫 에피소드 - 기준 PID (40, 50, 0)로 상태 추정")
                base_pid = np.array([40.0, 50.0, 0.0], dtype=np.float32)
                initial_state = create_initial_state(
                    [], self.cfg["TARGET_FORCE"], base_pid, None, self.cfg["RECV_INTERVAL_SEC"]
                )

            # 2. PID gain 사용 및 에피소드 시작 신호 전송
            if ep == 0:
                # 첫 에피소드: 로봇제어PC 자체 PID 사용 (P=40, I=50, D=0)
                pid_gains = np.array([40.0, 50.0, 0.0], dtype=np.float32)
                print(
                    f"🎯 [에피소드 1] 로봇제어PC 자체 PID 사용: Kp={pid_gains[0]:.0f}, Ki={pid_gains[1]:.0f}, Kd={pid_gains[2]:.0f}"
                )
                self._log(
                    "INFO",
                    f"🎯 에피소드 1 기준 PID: Kp={pid_gains[0]:.0f}, Ki={pid_gains[1]:.0f}, Kd={pid_gains[2]:.0f}",
                )
            else:
                # 2번째 에피소드부터: 이전 에피소드 종료 시 전송한 PID 사용
                assert (
                    self.pid_gains_next is not None
                ), f"에피소드 {ep+1}: 이전 에피소드에서 next PID가 설정되지 않았습니다!"
                pid_gains = self.pid_gains_next.copy()
                print(
                    f"🤖 [에피소드 {ep+1}] PID: Kp={pid_gains[0]:.2f}, Ki={pid_gains[1]:.2f}, Kd={pid_gains[2]:.2f}"
                )

                # ⭐ 에피소드 시작 신호: episode_done=False 전송 (플래그 리셋)
                print("📤 에피소드 시작 신호 전송 (episode_done=False)")
                self.comm.send_pid_once(
                    pid_gains[0],
                    pid_gains[1],
                    0.0,  # 하드웨어에는 항상 D=0.0 전송 (미분 작용 비활성화)
                    timing_accurate=True,
                    episode_done=False,
                    learning_done=False,
                )

            # 3. PID 적용 대기
            time.sleep(0.1)

            # 5. 새로운 PID gain으로 제어된 실제 상태 관측
            actual_state, _ = self.comm.get_latest_state()
            if actual_state is not None:
                # 실제 관측된 상태로 업데이트
                actual_initial_state = create_initial_state(
                    [actual_state[0]], self.cfg["TARGET_FORCE"], dt_sec=self.cfg["RECV_INTERVAL_SEC"]
                )
                self._log(
                    "INFO", f"📊 실제 상태 관측: Force={actual_state[0]:.2f}N"
                )
            else:
                # 관측 실패 시 추정 상태 사용
                actual_initial_state = initial_state
                print("⚠️  [경고] 실제 상태 관측 실패, 추정 상태 사용")
                self._log("WARNING", "실제 상태 관측 실패, 추정 상태 사용")

            # 6. 에피소드 시간 동안 1kHz 데이터 수집
            self.episode_force_data = []
            self.episode_pi_output_data = []
            self.current_pid_gains = pid_gains.copy()

            # sander_active 상승 에지 대기 (False -> True 전환 감지)
            # 첫 에피소드는 이미 True 상태이므로 상승 에지 대기 건너뜀
            if ep == 0:
                print(
                    "✅ [첫 에피소드] 이미 sander_active=True 상태 → 바로 시작"
                )
                self._log(
                    "INFO", "✅ 첫 에피소드: 이미 활성화 상태, 바로 시작"
                )
            else:
                # 2번째 에피소드부터: 상승 에지 대기 (모니터링 지속)
                print("⏳ sander_active 상승 에지 대기 중 (모니터링 지속)...")
                self._log("INFO", "⏳ sander_active 상승 에지 대기 중...")
                wait_start = time.perf_counter()
                prev_active = None
                last_monitor_sent = 0.0  # 모니터 전송용 타이머 초기화
                while True:
                    state, sander_active = self.comm.get_latest_state()

                    # 첫 읽기: 이전 상태 초기화
                    if prev_active is None:
                        prev_active = sander_active
                        time.sleep(0.001)
                        continue

                    # 모니터링 지속 (에이전트는 기억 안 함)
                    if state is not None:
                        now = time.perf_counter()
                        if now - last_monitor_sent >= 0.1:  # 10 Hz
                            t_global = now - training_start_time
                            monitor.post_force(
                                t_global,
                                float(abs(state[0])),
                                float(abs(self.cfg["TARGET_FORCE"])),
                            )
                            last_monitor_sent = now

                    # 상승 에지 (False -> True) 포착
                    if (prev_active is False) and (sander_active is True):
                        wait_duration = time.perf_counter() - wait_start
                        print(
                            f"✅ sander_active 상승 에지 감지! (False→True, {wait_duration:.2f}s 대기) → 에피소드 시작"
                        )
                        self._log(
                            "INFO",
                            f"✅ sander_active 상승 에지 감지 ({wait_duration:.2f}s)",
                        )
                        break

                    prev_active = sander_active
                    time.sleep(0.001)

                    # 타임아웃 (30초)
                    if time.perf_counter() - wait_start > 30.0:
                        print(
                            "⚠️ sander_active 대기 타임아웃 (30초) - 에피소드 강제 시작"
                        )
                        self._log(
                            "WARNING",
                            "sander_active 대기 타임아웃 - 에피소드 강제 시작",
                        )
                        break

            # sander_active 상승 에지부터 동적 길이 데이터 수집
            self._log(
                "INFO",
                f"📊 {self.cfg['EPISODE_SECONDS']:.0f}초 1kHz 데이터 수집 시작...",
            )
            start_time = time.perf_counter()
            self.episode_start_time = start_time  # 에피소드 시작 시간 기록

            data_count = 0
            prev_error = 0.0
            prev_pi_output = 0.0
            last_monitor_sent = 0.0  # 모니터 전송용 타이머

            # 주기 고정 방식으로 1kHz 정확도 향상
            dt = 0.001  # 1ms
            t_next = time.perf_counter()

            while (time.perf_counter() - start_time) < self.cfg[
                "EPISODE_SECONDS"
            ]:
                state, sander_active = self.comm.get_latest_state()
                if state is None:
                    time.sleep(0.001)
                    continue

                self.episode_force_data.append(state[0])  # 힘 데이터 수집
                self.episode_pi_output_data.append(
                    state[5]
                )  # PI 출력 데이터 수집
                data_count += 1
                
                # 🚨 안전 위반 체크
                if abs(state[0]) > Constants.SAFETY_FORCE_LIMIT:
                    print(f"\n🚨🚨🚨 [안전 위반] 힘: {state[0]:.1f}N (한계: ±{Constants.SAFETY_FORCE_LIMIT}N) 🚨🚨🚨")
                    self._log("ERROR", f"안전 위반! 힘: {state[0]:.1f}N > {Constants.SAFETY_FORCE_LIMIT}N")
                    
                    # 안전 위반 시 큰 패널티 부여 및 에피소드 즉시 종료
                    safety_violation_reward = Constants.SAFETY_FORCE_PENALTY
                    self._log("INFO", f"안전 위반 패널티: {safety_violation_reward:.2f}")
                    
                    # 🎲 다음 에피소드용 랜덤 PID 생성 (안전한 범위에서 재시작)
                    next_pid_for_reset = self.agent.select_action_random()
                    print(f"🎲 다음 에피소드용 랜덤 PID 생성: Kp={next_pid_for_reset[0]:.2f}, Ki={next_pid_for_reset[1]:.2f}, Kd={next_pid_for_reset[2]:.2f}")
                    self.pid_gains_next = next_pid_for_reset.copy()  # 저장
                    
                    # 로봇 초기화 동작 신호 전송 (다음 에피소드 PID와 함께)
                    print("🔄 로봇 초기화 동작 시작 (episode_done=True + 다음 PID 전송)...")
                    self.comm.send_pid_once(
                        next_pid_for_reset[0],  # ✅ 다음 에피소드 PID 전송
                        next_pid_for_reset[1],
                        0.0,  # 하드웨어에는 항상 D=0.0 전송
                        timing_accurate=False,
                        episode_done=True,  # 에피소드 종료 신호
                        learning_done=False,
                    )
                    
                    # 에피소드 강제 종료 - 안전 위반 보상 반환
                    episode_stats.append({
                        "episode": ep + 1,
                        "reward": safety_violation_reward,
                        "rmse": 9999.0,  # 실패 표시
                        "overshoot": 100.0,
                        "settling_time": self.cfg["EPISODE_SECONDS"],
                        "pid_gains": pid_gains.copy(),
                        "duration": time.perf_counter() - start_time,
                        "safety_violation": True,
                    })
                    
                    # 에이전트에 경험 저장 (매우 나쁜 보상)
                    self.agent.episode_rewards.append(safety_violation_reward)
                    
                    # ⏳ 로봇 초기화 동작 대기 (공압 툴 붙였다 떼기 완료 대기)
                    print("⏳ 로봇 초기화 동작 대기 중 (2초, 모니터링 지속)...")
                    reset_wait_start = time.perf_counter()
                    
                    while (time.perf_counter() - reset_wait_start) < 2.0:
                        # 계속 데이터 받아서 모니터에 전송
                        reset_state, _ = self.comm.get_latest_state()
                        if reset_state is not None:
                            now = time.perf_counter()
                            if now - last_monitor_sent >= 0.1:  # 10 Hz
                                t_global = now - training_start_time
                                monitor.post_force(
                                    t_global,
                                    float(abs(reset_state[0])),
                                    float(abs(self.cfg["TARGET_FORCE"])),
                                )
                                last_monitor_sent = now
                        time.sleep(0.01)  # CPU 부하 방지
                    
                    print("✅ 로봇 초기화 동작 완료")
                    
                    # 다음 에피소드로 이동
                    print(f"⏭️  안전 위반으로 에피소드 {ep+1} 조기 종료 → 다음 에피소드로 이동\n")
                    self._log("WARNING", f"에피소드 {ep+1} 안전 위반으로 조기 종료")
                    
                    # break를 통해 현재 에피소드 데이터 수집 루프 탈출
                    break

                # 실시간 제어 지표 데이터 수집
                current_time = time.perf_counter() - self.episode_start_time
                self.cplogger.add_data_point(
                    time=current_time,
                    force=state[0],
                    target=self.cfg["TARGET_FORCE"],
                    control_effort=np.sum(np.abs(pid_gains)),
                    pi_output=state[5],
                    pid_gains=pid_gains,  # PID gain 정보 추가
                )

                # RewardBreakdownLogger에 스텝 로그 추가 (state가 None이 아님을 보장)
                if hasattr(self, "rlogger"):
                    error = abs(state[0] - self.cfg["TARGET_FORCE"])
                    PROG_SCALE = 5.0
                    prog = np.exp(-error / PROG_SCALE)
                    in_band_now = error <= Constants.BAND_RATIO_TOLERANCE * abs(self.cfg["TARGET_FORCE"])
                    edot_abs = (
                        abs(prev_error - error) / 0.001
                        if data_count > 1
                        else 0.0
                    )
                    du_abs = (
                        abs(state[5] - prev_pi_output)
                        if data_count > 1
                        else 0.0
                    )
                    self.rlogger.log_step(
                        ep + 1,
                        data_count,
                        prog,
                        in_band_now,
                        edot_abs,
                        du_abs,
                        0.0,
                        False,
                    )
                    prev_error = error
                    prev_pi_output = state[5]

                # 주기 고정 방식으로 정확한 1kHz 수집
                t_next += dt
                delay = t_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

                # 10 Hz로만 실시간 Force 전송 (수신/제어는 1 kHz 유지)
                now = time.perf_counter()
                if now - last_monitor_sent >= 0.1:  # 10 Hz
                    # 전역 시간 사용 (학습 시작부터의 경과 시간)
                    t_global = now - training_start_time
                    monitor.post_force(
                        t_global,
                        float(abs(state[0])),  # current_force
                        float(abs(self.cfg["TARGET_FORCE"])),  # desired_force
                    )
                    last_monitor_sent = now

            print(
                f"📈 [수집] 완료: {data_count}개 데이터 (목표: {int(self.cfg['EPISODE_SECONDS'] * 1000)})"
            )
            self._log(
                "INFO",
                f"📈 수집된 데이터 포인트: {data_count}개 (목표: {int(self.cfg['EPISODE_SECONDS'] * 1000)})",
            )

            # 안전 위반으로 조기 종료된 경우 체크
            safety_violated = (len(episode_stats) > 0 and 
                             episode_stats[-1].get("episode") == ep + 1 and 
                             episode_stats[-1].get("safety_violation", False))
            
            if safety_violated:
                print("⏭️  안전 위반으로 인한 조기 종료 - 최소 학습 후 다음 에피소드로 이동")
                self._log("WARNING", f"에피소드 {ep+1}: 안전 위반 - 최소 transition 저장 및 1회 학습 수행")

                # 안전 위반 보상으로 transition 저장
                bad_reward = episode_stats[-1]["reward"] if episode_stats else Constants.SAFETY_FORCE_PENALTY
                final_state_violation = np.zeros(self.cfg["STATE_DIM"], dtype=np.float32)
                self.agent.store_transition(
                    actual_initial_state,
                    pid_gains,
                    bad_reward,
                    final_state_violation,
                    True,
                )
                # 단발 학습 1회
                if len(self.agent.replay) >= max(2, Constants.MIN_BATCH_SIZE):
                    self.agent.update_parameters_one_step(
                        batch_size=min(self.cfg["BATCH_SIZE"], len(self.agent.replay)),
                        num_updates=1,
                    )
                # 다음 PID는 이미 안전 위반 처리 중에 설정됨 (self.pid_gains_next)
                continue  # 현재 에피소드 처리 종료, 다음 에피소드로
            
            # 5. 에피소드 총보상 계산 (정상 종료 시에만)
            episode_reward, metrics = self.calculate_episode_reward(
                self.episode_force_data,
                self.episode_pi_output_data,
                self.cfg["TARGET_FORCE"],
                self.cfg["EPISODE_SECONDS"],
            )
            print(
                f"🏆 [결과] 보상: {episode_reward:.2f}, RMSE: {metrics['rmse']:.2f}, 오버슈트: {metrics['overshoot']:.1f}%"
            )

            # =========================
            # [C] 에피소드 종료 시: reward 점 업데이트 (한 번)
            # =========================
            monitor.post_reward(ep + 1, float(episode_reward))

            # 6. Transition 저장 (한 스텝 MDP) - 실제 관측된 초기 상태 및 최종 상태 사용
            # 최종 상태: 에피소드 결과를 반영한 상태
            if len(self.episode_force_data) > 0:
                final_state = create_initial_state(
                    self.episode_force_data,
                    self.cfg["TARGET_FORCE"],
                    pid_gains,
                    self.episode_history,
                    self.cfg["RECV_INTERVAL_SEC"],
                )
            else:
                final_state = np.zeros(
                    self.cfg["STATE_DIM"], dtype=np.float32
                )
            
            self.agent.store_transition(
                actual_initial_state,
                pid_gains,
                episode_reward,
                final_state,
                True,
            )

            # 7. 통계 업데이트 (학습 전에 먼저 업데이트)
            episode_duration = time.perf_counter() - start_time
            episode_stat = {
                "episode": ep + 1,
                "duration": episode_duration,
                "pid_gains": pid_gains.copy(),
                "reward": episode_reward,
                "metrics": metrics,
            }
            episode_stats.append(episode_stat)
            self.agent.episode_rewards.append(episode_reward)

            # RewardBreakdownLogger 플러시 (에피소드 경계에서) - 그래프 생성은 최종에만
            if hasattr(self, "rlogger"):
                self.rlogger.flush_if_needed(
                    ep + 1,
                    force=False,
                    episode_rewards=self.agent.episode_rewards,
                )

            # 최고 성능 PID gain 저장 (동일 점수도 저장)
            if episode_reward >= best_reward:
                best_reward = episode_reward
                best_pid_gains = pid_gains.copy()
                self.agent.save_model(
                    f"{model_save_dir}/best_pid_agent_episode_{ep+1}_reward_{best_reward:.2f}.pth"
                )
                print(
                    f"💾 [저장] 최고 성능 에이전트 저장: 에피소드 {ep+1}, 보상 {best_reward:.2f}"
                )

            # 8. 학습 (안정적 gradient 추정을 위한 최소 버퍼 크기 보장)
            # 중요: 학습을 먼저 수행한 후 다음 PID를 계산해야 학습된 네트워크 사용!
            buffer_size = len(self.agent.replay)
            
            # 최소 버퍼 크기 체크 (안정적 학습 보장)
            if buffer_size >= Constants.MIN_BUFFER_FOR_LEARNING:
                # 고정된 업데이트 횟수 사용 (안정적 학습)
                actual_updates = self.cfg["UPDATES_PER_EPISODE"]
                
                # 배치 크기: 안정적 학습을 위한 조정
                effective_batch_size = min(
                    self.cfg["BATCH_SIZE"],  # 최대 128
                    max(Constants.MIN_BATCH_SIZE, buffer_size // 2)  # 버퍼의 절반 사용
                )
                
                print(
                    f"🧠 [학습] 강화학습 업데이트 중... (에피소드 {ep+1}, {actual_updates}회, "
                    f"배치크기: {effective_batch_size}, 버퍼: {buffer_size}개)"
                )
                
                self.agent.update_parameters_one_step(
                    effective_batch_size, actual_updates
                )
                
                print(
                    f"✅ [학습] 신경망 업데이트 완료! 다음 에피소드는 학습된 네트워크 사용"
                )
                self._log(
                    "INFO", 
                    f"🧠 학습 완료: {actual_updates}회 업데이트, 배치={effective_batch_size}"
                )
            else:
                # 초기 탐색 단계 (학습 없이 데이터만 수집)
                print(
                    f"📊 [에피소드 {ep+1}] 초기 탐색 중... (버퍼: {buffer_size}/{Constants.MIN_BUFFER_FOR_LEARNING}개, "
                    f"학습 시작까지 {Constants.MIN_BUFFER_FOR_LEARNING - buffer_size}개 필요)"
                )
                self._log(
                    "INFO",
                    f"📊 초기 탐색: 버퍼 {buffer_size}/{Constants.MIN_BUFFER_FOR_LEARNING}개"
                )

            # 9. 로깅 (상세)
            self._log("INFO", f"🎯 에피소드 {ep+1} 완료")
            self._log("INFO", f"⏱️  지속시간: {episode_duration:.1f}s")
            self._log("INFO", f"🏆 보상: {episode_reward:.2f}")
            self._log("INFO", f"📊 RMSE: {metrics['rmse']:.2f}")
            self._log("INFO", f"📈 오버슈트: {metrics['overshoot']:.1f}%")
            self._log("INFO", f"⏰ 정착시간: {metrics['settling_time']:.2f}s")
            self._log("INFO", f"🎯 밴드유지: {metrics['band_time']:.1f}s")
            self._log("INFO", f"🏅 최고보상: {best_reward:.2f}")

            # 10. 이전 에피소드 정보 업데이트
            self.previous_pid_gains = pid_gains.copy()

            # 에피소드 히스토리 업데이트 (최근 5개 에피소드만 유지)
            episode_record = {
                "episode": ep + 1,
                "pid_gains": pid_gains.copy(),
                "reward": episode_reward,
                "metrics": metrics,
            }
            self.episode_history.append(episode_record)

            # 최근 5개 에피소드만 유지
            if len(self.episode_history) > self.max_history:
                self.episode_history.pop(0)

            print(
                f"📊 [히스토리] 에피소드 {ep+1} 기록: 보상={episode_reward:.2f}, PID={pid_gains}"
            )
            print(
                f"📈 [히스토리] 총 {len(self.episode_history)}개 에피소드 기록 유지 (최대 5개)"
            )

            # 에러 통계 업데이트 (최근 10개 에피소드만 유지)
            if len(self.episode_force_data) > 0:
                episode_errors = [
                    f - self.cfg["TARGET_FORCE"]
                    for f in self.episode_force_data
                ]
                avg_error = np.mean(episode_errors)
                self.historical_errors.append(avg_error)

                # 최근 10개 에피소드만 유지
                if len(self.historical_errors) > 10:
                    self.historical_errors.pop(0)

                print(
                    f"📈 [에러] 평균 에러: {avg_error:.2f}N, 히스토리={len(self.historical_errors)}개"
                )
            else:
                print(
                    "⚠️  [경고] 에피소드 데이터 없음, 에러 히스토리 업데이트 건너뜀"
                )

            print(
                f"💾 [저장] 다음 에피소드용 PID: Kp={self.previous_pid_gains[0]:.2f}, Ki={self.previous_pid_gains[1]:.2f}, Kd={self.previous_pid_gains[2]:.2f}"
            )

            # 11. 에피소드별 제어 지표 저장
            print(f"💾 [저장] 에피소드 지표 저장 중...")
            self.cplogger.save_episode_metrics(ep + 1)
            self.cplogger.reset_episode_data()

            # 12. 다음 에피소드 PID 게인 미리 계산 (마지막 에피소드가 아닌 경우)
            if ep < episodes - 1:
                # 다음 에피소드를 위한 초기 상태 생성
                next_initial_state = create_initial_state(
                    [],
                    self.cfg["TARGET_FORCE"],
                    pid_gains,  # 현재 에피소드의 PID (실제 사용된 것)
                    self.episode_history,
                    self.cfg["RECV_INTERVAL_SEC"],
                )
                # 다음 에피소드 PID 게인 선택 (+ ε-greedy: 10 에피소드마다 5% 확률 랜덤)
                use_random = ((ep + 1) % 10 == 0) and (random.random() < 0.05)
                if use_random:
                    next_pid_gains = self.agent.select_action_random()
                    self._log(
                        "INFO",
                        "🎲 ε-greedy 적용: 랜덤 PID 선택 (10-주기, p=0.05)",
                    )
                else:
                    next_pid_gains, _ = self.agent.select_action(
                        next_initial_state, evaluate=False
                    )
                self.pid_gains_next = (
                    next_pid_gains.copy()
                )  # ✅ 저장! (다음 에피소드에서 사용)
                print(
                    f"🎯 [다음 에피소드] PID 계산 및 저장: Kp={next_pid_gains[0]:.2f}, Ki={next_pid_gains[1]:.2f}, Kd={next_pid_gains[2]:.2f}"
                )
                self._log(
                    "INFO",
                    f"🎯 다음 에피소드 PID: Kp={next_pid_gains[0]:.2f}, Ki={next_pid_gains[1]:.2f}, Kd={next_pid_gains[2]:.2f}",
                )
            else:
                # 마지막 에피소드인 경우 현재 PID 사용
                next_pid_gains = pid_gains
                self.pid_gains_next = next_pid_gains.copy()

            # 13. 에피소드 완료 신호 전송 (다음 에피소드 PID 게인과 함께)
            print(f"📤 [전송] 에피소드 {ep+1} 완료 신호 + 다음 PID 전송 중...")
            episode_done_success = self.comm.send_pid_once(
                # ✅ 저장된 값 전송
                self.pid_gains_next[0],
                self.pid_gains_next[1],
                0.0,  # 하드웨어에는 항상 D=0.0 전송
                timing_accurate=True,
                episode_done=True,
                learning_done=False,
            )
            if episode_done_success:
                print(
                    f"✅ [전송] 에피소드 {ep+1} 완료 신호 + 다음 PID 전송 성공"
                )
                self._log(
                    "INFO",
                    f"📤 에피소드 {ep+1} 완료 신호 + 다음 PID 전송 성공",
                )
            else:
                print(
                    f"❌ [오류] 에피소드 {ep+1} 완료 신호 + 다음 PID 전송 실패"
                )
                self._log(
                    "ERROR", f"에피소드 {ep+1} 완료 신호 + 다음 PID 전송 실패"
                )

            # 12. 로봇 리셋 대기 (모니터링 지속)
            if ep < episodes - 1:
                print("⏳ 로봇 리셋 대기 중 (2초, 모니터링 지속)...")
                wait_start = time.perf_counter()

                while (time.perf_counter() - wait_start) < 2.0:
                    # 계속 데이터 받아서 모니터에 전송 (에이전트는 기억 안 함)
                    state, _ = self.comm.get_latest_state()
                    if state is not None:
                        now = time.perf_counter()
                        if now - last_monitor_sent >= 0.1:  # 10 Hz
                            t_global = now - training_start_time
                            monitor.post_force(
                                t_global,
                                float(abs(state[0])),
                                float(abs(self.cfg["TARGET_FORCE"])),
                            )
                            last_monitor_sent = now
                    time.sleep(0.01)  # CPU 부하 방지

        # 12. 최종 결과
        self._log("INFO", "\n🎯 PID Gain 최적화 완료!")
        self._log("INFO", f"✅ {episodes}개 에피소드 완료")
        self._log("INFO", f"🏆 최고 보상: {best_reward:.2f}")
        self._log(
            "INFO",
            f"🎯 최적 PID: Kp={best_pid_gains[0]:.2f}, Ki={best_pid_gains[1]:.2f}, Kd={best_pid_gains[2]:.2f}",
        )

        # 13. 에피소드 요약 CSV 저장
        summary_csv = os.path.join(
            self.cplogger.control_perf_dir, "episode_summary.csv"
        )
        with open(summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "episode",
                    "Kp",
                    "Ki",
                    "Kd",
                    "reward",
                    "rmse",
                    "overshoot",
                    "settling_time",
                    "band_time",
                    "out_of_band_time",
                ]
            )
            for s in episode_stats:
                m = s["metrics"]
                w.writerow(
                    [
                        s["episode"],
                        *s["pid_gains"],
                        s["reward"],
                        m["rmse"],
                        m["overshoot"],
                        m["settling_time"],
                        m["band_time"],
                        m["out_of_band_time"],
                    ]
                )
        self._log("INFO", f"📄 에피소드 요약 저장: {summary_csv}")

        # 14. 데이터 저장
        DataSaver.save_all_data(self, episodes, force=True)

        # 15. 학습 완료 신호 전송
        print(f"📤 [전송] 모든 에피소드 완료 - 학습 종료 신호 전송 중...")
        learning_done_success = self.comm.send_pid_once(
            0, 0, 0, True, False, True
        )  # learning_done=True
        if learning_done_success:
            print(f"✅ [전송] 학습 종료 신호 전송 성공")
            self._log("INFO", "📤 학습 종료 신호 전송 성공")
        else:
            print(f"❌ [오류] 학습 종료 신호 전송 실패")
            self._log("ERROR", "학습 종료 신호 전송 실패")

        # 16. 모니터 종료
        print("📊 [Monitor] 실시간 모니터 종료 중...")
        monitor.stop()
        print("✅ [Monitor] 실시간 모니터 종료 완료")

        self.comm.close()

