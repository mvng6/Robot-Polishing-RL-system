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

from ..utils.loggers.base_logger import AppLogger
from ..utils.loggers.control_performance import ControlPerformanceLogger
from ..utils.loggers.reward_breakdown import RewardBreakdownLogger
from ..utils.loggers.learning_done import LearningDoneLogger
from ..config.constants import Constants
from ..utils.utils.math_utils import create_initial_state, scale_action_to_pid
from .agent import PIDGainSACAgent
from .comm import PIDGainCommunicator
from .monitor import RLRealtimeMonitor

class PIDGainEnvironment:
    """
    PID Gain 최적화 환경
    - 🆕 세그먼트 분할 학습 (1 에피소드 = 5 transition)
    - 🆕 STATE_DIM 차원 상태 공간 (STATE_BASE_DIM + STATE_TRAJECTORY_DIM)
    """

    def __init__(self, cfg=None):
        if cfg is None:
            raise ValueError("cfg 파라미터가 필요합니다")
        
        # 🔥 STATE_DIM 설정 (Constants에서 가져오기)
        cfg["STATE_DIM"] = Constants.STATE_DIM
        
        print(f"✅ [Env] STATE_DIM 설정: {cfg['STATE_DIM']}차원 "
              f"(기본 {Constants.STATE_BASE_DIM} + 궤적 {Constants.STATE_TRAJECTORY_DIM})")
        print(f"✅ [Env] 세그먼트 분할: {Constants.NUM_SEGMENTS}개 "
              f"({Constants.SEGMENT_LENGTH_S}초씩)")
        print(f"✅ [Env] Fine-tuning: alpha={Constants.ACTOR_INITIAL_ALPHA}, "
              f"log_std_max={Constants.ACTOR_LOG_STD_MAX}")
        
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
        연속형 지수 스코어 + PBRS + 기준선 중심화 + tanh 소프트클립
        최종 범위: [-1.0, 1.0], 개선/악화 신호 강화
        
        주요 특징:
        1. 단위 일관성: overshoot % 단위로만 사용
        2. 안전 위반: 하드 패널티 (-1.0)
        3. EWMA 기준선: 지속적 개선 신호 제공 (보상 중심화)
        4. tanh 소프트클립: 그라디언트 소멸 방지
        5. 범위: [-1, 1] (SAC 학습에 최적화)
        
        Args:
            force_data: 에피소드 힘 데이터
            pi_output_data: PI 출력 데이터
            target_force: 목표 힘 (기본값: cfg)
            episode_len_s: 에피소드 길이 (기본값: cfg)
        
        Returns:
            reward: [-1.0, 1.0] 범위 보상
            metrics: 성능 지표 딕셔너리
        """
        import numpy as np

        # ========== 1. 입력 검증 및 기본값 ==========
        if not force_data:
            return 0.0, {}
        
        if target_force is None:
            target_force = self.cfg["TARGET_FORCE"]
        if episode_len_s is None:
            episode_len_s = self.cfg["EPISODE_SECONDS"]

        force_array = np.array(force_data, dtype=np.float64)
        n_samples = len(force_array)
        fs_hz = n_samples / max(episode_len_s, 1e-6)
        dt = 1.0 / max(fs_hz, 1e-6)
        errors = force_array - target_force
        T_abs = max(abs(target_force), 1.0)

        # ========== 2. 핵심 지표 계산 ==========
        
        # RMSE (전체 추종 오차)
        rmse = float(np.sqrt(np.mean(errors**2)))

        # 🆕 초기 구간 피크 패널티 (0~0.5초 구간 전용)
        early_peak_penalty = self._calculate_early_peak_penalty(
            force_array, target_force, episode_len_s, dt
        )

        # Overshoot (% 단위로 일관화)
        if target_force < 0:
            extreme_force = float(np.min(force_array))
            overshoot_ratio = max(0.0, (target_force - extreme_force) / T_abs)
        else:
            extreme_force = float(np.max(force_array))
            overshoot_ratio = max(0.0, (extreme_force - target_force) / T_abs)
        
        overshoot_pct = float(overshoot_ratio * 100.0)  # % 단위로 변환

        # 밴드 유지율 (±BAND_TOLERANCE_N)
        tol_main = Constants.BAND_TOLERANCE_N
        band_ratio = float(np.mean(np.abs(errors) <= tol_main)) if n_samples > 0 else 0.0

        # 정착시간 (Settling Time) - warmup 후
        SHAPING_WARMUP_S = Constants.SHAPING_WARMUP_S
        start_idx = int(fs_hz * SHAPING_WARMUP_S)
        tol_settling = max(Constants.SETTLING_BAND_TOLERANCE, 0.01 * T_abs)
        hold_samples = int(fs_hz * Constants.SETTLING_HOLD_TIME_S)
        in_settling = np.abs(errors) <= tol_settling
        
        settling_time_s = episode_len_s  # 기본값: 전체 시간
        runlen = 0
        for k in range(start_idx, n_samples):
            if in_settling[k]:
                runlen += 1
                if runlen >= hold_samples:
                    settling_time_s = max(0.0, (k - hold_samples) * dt)
                    break
            else:
                runlen = 0

        # 정상상태 오차 (ESS): 마지막 10% 구간
        if n_samples > 0:
            tail_len = max(1, int(n_samples * 0.1))
            ess = float(np.mean(errors[-tail_len:]))
        else:
            ess = 0.0

        # 제어 입력 품질 (u_rms 정규화)
        u_rms_n = 0.0
        if pi_output_data:
            u = np.array(pi_output_data, dtype=np.float64)
            u_max = Constants.PI_OUTPUT_MAX if Constants.PI_OUTPUT_MAX > 0 else 1.0
            u_rms_n = float(np.sqrt(np.mean(u**2)) / u_max)

        # ========== 3. PBRS (Potential-Based Reward Shaping) ==========
        abs_e = np.abs(errors)
        skip = min(start_idx, len(abs_e))
        abs_e_eff = abs_e[skip:] if skip < len(abs_e) else abs_e
        progress = 0.0
        
        if len(abs_e_eff) > 1:
            phi = -abs_e_eff / T_abs                  # Potential 함수
            gamma = Constants.POTENTIAL_GAMMA        # 0.99 권장
            F = gamma * phi[1:] - phi[:-1]           # 개선분
            progress = float(np.mean(np.clip(F, 0.0, None)))  # 양수만

        # ========== 4. 추종 실패 패널티 (0~1) ==========
        rmse_fail_ratio = max(0.0, (rmse - Constants.TRACKING_FAIL_RMSE_THRESHOLD) / 
                          Constants.TRACKING_FAIL_RMSE_THRESHOLD)
        rmse_fail_ratio = min(1.0, rmse_fail_ratio)
        
        band_fail_ratio = max(0.0, (Constants.TRACKING_FAIL_BAND_RATIO - band_ratio) / 
                          Constants.TRACKING_FAIL_BAND_RATIO)
        band_fail_ratio = min(1.0, band_fail_ratio)
        
        P_fail = max(rmse_fail_ratio, band_fail_ratio)

        # ========== 5. 연속형 스코어 (0~1) ==========
        tau_settle = max(Constants.SCORE_TAU_TS, 1e-6)
        tau_mp = max(Constants.SCORE_TAU_MP_PERCENT, 1e-6)
        tau_ess = max(Constants.SCORE_TAU_ESS_N, 1e-6)
        tau_u = max(Constants.SCORE_TAU_U, 1e-6)

        S_ts   = float(np.exp(-float(settling_time_s) / tau_settle))
        S_mp   = float(np.exp(-float(overshoot_pct) / tau_mp))
        S_ess  = float(np.exp(-abs(ess) / tau_ess))
        S_band = float(band_ratio)
        S_u    = float(np.exp(-float(u_rms_n) / tau_u))

        # ========== 6. 🎯 기준선 대비 개선량 (EWMA) ==========
        # 첫 호출 시 초기화
        if not hasattr(self, "_baseline"):
            self._baseline = {
                "ts": settling_time_s,
                "mp": overshoot_pct,
                "ess_abs": abs(ess),
                "initialized": False,
            }
        
        # 개선량 계산 (첫 에피소드는 0, 이후부터 유효)
        if self._baseline["initialized"]:
            I_ts  = float(np.clip((self._baseline["ts"] - settling_time_s) / tau_settle, -1.0, 1.0))
            I_mp  = float(np.clip((self._baseline["mp"] - overshoot_pct) / tau_mp, -1.0, 1.0))
            I_ess = float(np.clip((self._baseline["ess_abs"] - abs(ess)) / tau_ess, -1.0, 1.0))
            
            # 가중 평균 (정착시간 40%, 오버슈트 30%, ESS 30%)
            I_improve = 0.4 * I_ts + 0.3 * I_mp + 0.3 * I_ess
        else:
            I_improve = 0.0
            self._baseline["initialized"] = True  # 다음 에피소드부터 활성화

        # EWMA 업데이트 (α=0.1: 최근 10개 에피소드 가중평균)
        alpha = 0.1
        self._baseline["ts"] = (1.0 - alpha) * self._baseline["ts"] + alpha * settling_time_s
        self._baseline["mp"] = (1.0 - alpha) * self._baseline["mp"] + alpha * overshoot_pct
        self._baseline["ess_abs"] = (1.0 - alpha) * self._baseline["ess_abs"] + alpha * abs(ess)

        # ========== 7. 통합 보상 스코어 (0~1) ==========
        W_PBRS = Constants.SCORE_W_PBRS  # 0.10
        W_IMPR = 0.10  # 개선량 가중치
        
        reward_score = (
            Constants.SCORE_W_TS * S_ts +       # 0.30: 정착시간
            Constants.SCORE_W_MP * S_mp +       # 0.35: 오버슈트 (업데이트됨)
            Constants.SCORE_W_ESS * S_ess +     # 0.20: 정상상태 오차
            Constants.SCORE_W_BAND * S_band +   # 0.15: 밴드 유지
            Constants.SCORE_W_U * S_u -         # 0.05: 제어 노력
            Constants.SCORE_W_FAIL * P_fail +   # -0.15: 추종 실패
            W_PBRS * progress +                 # +0.10: 순간 개선 (PBRS)
            W_IMPR * I_improve -                 # +0.10: 기준선 대비 개선
            early_peak_penalty                   # 🆕 초기 구간 피크 패널티
        )

        # ========== 8. 🔥 중심화 + tanh 소프트클립 [-1,1] ==========
        # Beta 값을 변수로 관리 (0.98~0.99 범위에서 조정 가능)
        beta = 0.99  # 느리게 추적 (최근 100 에피소드 가중평균)
        if not hasattr(self, "_rew_baseline"):
            self._rew_baseline = 0.0  # 초기값 변경: 0.5 → 0.0 (보상 범위 [-1, 1] 기준)
        
        # 보상 기준선 업데이트 (EWMA)
        self._rew_baseline = beta * self._rew_baseline + (1.0 - beta) * reward_score
        
        # 기준선 중심화
        r_centered = reward_score - self._rew_baseline
        
        # tanh 소프트클립 (gain으로 민감도 조절)
        gain = 2.0  # 민감도 조절 (1.5~3.0 튜닝 가능)
        reward = float(np.tanh(gain * r_centered))

        # ========== 9. 안전 위반 하드 패널티 ==========
        if abs(extreme_force) > Constants.SAFETY_FORCE_LIMIT:
            reward = Constants.SAFETY_FORCE_PENALTY  # -1.0
            print(
                f"⚠️ [안전 위반] 극한 힘: {extreme_force:.1f}N "
                f"(한계: ±{Constants.SAFETY_FORCE_LIMIT}N)"
            )

        # ========== 10. 메트릭 반환 ==========
        metrics = {
            # 기본 지표 (CSV 저장 및 로그 출력용)
            "rmse": rmse,
            "overshoot": overshoot_pct,  # % 단위로 일관화
            "settling_time": settling_time_s,
            "band_time": band_ratio * episode_len_s,
            "out_of_band_time": (1.0 - band_ratio) * episode_len_s,
            
            # 🔍 디버깅용 추가 정보
            "reward_score": reward_score,      # [0,1] 스코어
            "r_centered": r_centered,          # 중심화된 값
            "r_baseline": self._rew_baseline,  # 보상 기준선
            "progress": progress,              # PBRS 개선분
            "I_improve": I_improve,            # 기준선 대비 개선량
            "baseline_ts": self._baseline["ts"],
            "baseline_mp": self._baseline["mp"],
            "baseline_ess": self._baseline["ess_abs"],
            "early_peak_penalty": early_peak_penalty,  # 🆕 초기 구간 피크 패널티
        }

        # ========== 11. NaN/Inf 안전 가드 ==========
        if np.isnan(reward) or np.isinf(reward):
            print(f"❌ [오류] NaN/Inf 보상 발견! → 0.0으로 대체")
            print(f"   force 범위: [{np.min(force_array):.2f}, {np.max(force_array):.2f}]")
            print(f"   reward_score: {reward_score:.4f}, r_centered: {r_centered:.4f}")
            reward = 0.0
        
        for key, val in list(metrics.items()):
            if np.isnan(val) or np.isinf(val):
                print(f"❌ [오류] 메트릭 '{key}' NaN/Inf: {val} → 기본값 대체")
                metrics[key] = 0.0 if key != "settling_time" else episode_len_s

        return reward, metrics
    
    def _calculate_early_peak_penalty(self, force_array, target_force, episode_len_s, dt):
        """
        🆕 초기 구간 피크 패널티 계산
        0~0.5초 구간에서 발생한 피크에 대한 패널티
        
        Args:
            force_array: 힘 데이터 배열
            target_force: 목표 힘
            episode_len_s: 에피소드 길이 (초)
            dt: 샘플링 시간 간격
        
        Returns:
            penalty: 패널티 값 (상한 0.15~0.2)
        """
        import numpy as np
        
        time_window = Constants.EARLY_PEAK_TIME_WINDOW  # 0.5초
        max_samples = int(time_window / dt)
        
        if len(force_array) < max_samples:
            # 데이터가 부족하면 전체 사용
            early_force = force_array
        else:
            early_force = force_array[:max_samples]
        
        if len(early_force) == 0:
            return 0.0
        
        # 피크 힘 계산
        if target_force < 0:
            peak_force = float(np.min(early_force))
        else:
            peak_force = float(np.max(early_force))
        
        # 피크 패널티 계산
        peak_detector = max(0.0, abs(peak_force - target_force) / abs(target_force))
        penalty = min(
            peak_detector * Constants.EARLY_PEAK_PENALTY_SCALE,
            Constants.EARLY_PEAK_PENALTY_MAX
        )
        
        return float(penalty)

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
            plt.xlabel("Episode", fontsize=18)
            plt.ylabel("Episode Reward", fontsize=18)
            plt.title(
                "Episode Rewards Over Time", fontsize=20, fontweight="bold"
            )
            plt.tick_params(labelsize=15)  # 축 눈금 폰트 크기
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
        best_reward = -float("inf")  # 전체 에피소드 최고 보상 (통계용)
        best_reward_after_min = -float("inf")  # MIN_EPISODES_FOR_SAVING 이후 최고 보상 (저장용)
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
        
        # 🆕 Warm-start: 버퍼 초기화 (첫 에피소드 전에만)
        if Constants.WARM_START_ENABLED and len(self.agent.replay) == 0:
            self.agent.warm_start_buffer()

        for ep in range(episodes):
            # 에피소드별 리셋 없음 (연속 모니터링)

            self.episode_count = ep
            
            # 🆕 Target Entropy 동적 조정
            self.agent.update_target_entropy(ep)
            
            # 🆕 탐색 메트릭 로깅 (20 에피소드마다)
            if ep % 20 == 0 and ep > 0:
                exploration_metrics = self.agent.log_exploration_metrics(ep)
                if exploration_metrics:
                    self._log("INFO", 
                        f"📊 탐색 메트릭 [Ep {ep}]: "
                        f"std_ratio={exploration_metrics['action_std_ratio_pct']:.1f}%, "
                        f"Kd_coverage={exploration_metrics['kd_coverage_pct']:.1f}%"
                    )
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
                        timing_accurate=True,  # ✅ 정상 종료와 동일한 플래그로 전송
                        episode_done=True,  # 에피소드 종료 신호
                        learning_done=False,
                    )
                    
                    # 에피소드 강제 종료 - 안전 위반 보상 반환
                    episode_stats.append({
                        "episode": ep + 1,
                        "reward": safety_violation_reward,
                        "metrics": {
                            "rmse": 999.0,
                            "overshoot": 100.0,
                            "settling_time": self.cfg["EPISODE_SECONDS"],
                            "band_time": 0.0,
                            "out_of_band_time": self.cfg["EPISODE_SECONDS"],
                        },
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
                        # 계속 데이터 받아서 모니터에 전송 (
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
                self._log("WARNING", f"에피소드 {ep+1} 안전 위반으로 조기 종료")

                # 안전 위반 보상으로 transition 저장 (STATE_DIM 차원 상태)
                bad_reward = episode_stats[-1]["reward"] if episode_stats else Constants.SAFETY_FORCE_PENALTY
                final_state_violation = np.zeros(Constants.STATE_DIM, dtype=np.float32)
                
                # actual_initial_state가 기본 차원이면 전체 차원으로 확장
                if actual_initial_state.shape[0] == Constants.STATE_BASE_DIM:
                    actual_initial_state_20d = np.concatenate([
                        actual_initial_state,
                        np.zeros(Constants.STATE_TRAJECTORY_DIM, dtype=np.float32)
                    ])
                else:
                    actual_initial_state_20d = actual_initial_state
                
                self.agent.store_transition(
                    actual_initial_state_20d,
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
                
                # 안전 위반 보상도 best_reward 추적에 포함 (통계 목적)
                # 단, 안전 위반 보상(-1.0)은 best_reward_after_min 저장 대상이 아님 (50 에피소드 이후 최고 보상만 저장)
                if bad_reward >= best_reward:
                    best_reward = bad_reward
                
                continue  # 다음 에피소드로
            
            # ========== 🔥 세그먼트 분할 처리 (라인 1005 대체) ==========
            
            force_data = self.episode_force_data
            pi_output_data = self.episode_pi_output_data
            
            if not force_data:
                print("❌ 데이터 수집 실패 - 다음 에피소드로")
                continue
            
            # 세그먼트 분할
            num_segments = Constants.NUM_SEGMENTS  # 5
            total_samples = len(force_data)
            segment_len = total_samples // num_segments
            
            segments_data = []
            
            for i in range(num_segments):
                start_idx = i * segment_len
                end_idx = (i + 1) * segment_len if i < num_segments - 1 else total_samples
                
                seg_force = force_data[start_idx:end_idx]
                seg_pi = pi_output_data[start_idx:end_idx] if pi_output_data else []
                seg_len_s = len(seg_force) / (total_samples / self.cfg["EPISODE_SECONDS"])
                
                # 세그먼트 보상 계산
                seg_reward, seg_metrics = self.calculate_episode_reward(
                    seg_force, seg_pi, self.cfg["TARGET_FORCE"], seg_len_s
                )
                
                # 세그먼트 상태 생성 (20차원)
                seg_state = self._build_segment_state(
                    prev_pid_gains=pid_gains,
                    prev_reward=seg_reward,
                    force_segment=seg_force,
                    target_force=self.cfg["TARGET_FORCE"],
                    segment_len_s=seg_len_s,
                )
                
                segments_data.append({
                    "state": seg_state,
                    "reward": seg_reward,
                    "metrics": seg_metrics,
                    "segment_idx": i,
                })
            
            # 리플레이 버퍼에 세그먼트별 저장
            for i, seg in enumerate(segments_data):
                if i == 0:
                    # 첫 세그먼트: 에피소드 초기 상태 사용 (STATE_DIM 차원으로 변환)
                    if actual_initial_state.shape[0] == Constants.STATE_BASE_DIM:
                        # 기본 차원이면 전체 차원으로 확장 (제로 패딩)
                        current_state = np.concatenate([
                            actual_initial_state,
                            np.zeros(Constants.STATE_TRAJECTORY_DIM, dtype=np.float32)
                        ])
                    else:
                        current_state = actual_initial_state
                else:
                    current_state = segments_data[i-1]["state"]
                
                if i < num_segments - 1:
                    next_state = segments_data[i+1]["state"]
                    done = False
                else:
                    next_state = np.zeros_like(seg["state"])
                    done = True
                
                self.agent.store_transition(
                    current_state,
                    pid_gains,
                    seg["reward"],
                    next_state,
                    done,
                )
            
            # 전체 보상 (평균)
            total_reward = np.mean([s["reward"] for s in segments_data])
            
            # 최종 메트릭 (전체 에피소드 기준)
            _, final_metrics = self.calculate_episode_reward(
                force_data, pi_output_data, 
                self.cfg["TARGET_FORCE"], self.cfg["EPISODE_SECONDS"]
            )
            
            # 표준편차 Annealing 업데이트
            self.agent.update_std_scale(ep)
            
            # 로그 출력
            seg_rewards_str = ", ".join([f"{s['reward']:.3f}" for s in segments_data])
            print(
                f"📊 세그먼트별 보상: [{seg_rewards_str}], "
                f"평균: {total_reward:.3f}, "
                f"std_scale: {self.agent.std_scale:.3f}"
            )
            self._log("INFO", 
                f"📊 세그먼트별 보상: [{seg_rewards_str}], 평균: {total_reward:.3f}"
            )
            
            # ========== 기존 로직 재개 (라인 1010~) ==========
            
            # 모니터 업데이트 (total_reward 사용)
            monitor.post_reward(ep + 1, float(total_reward))

            # 통계 업데이트
            episode_duration = time.perf_counter() - start_time
            episode_stat = {
                "episode": ep + 1,
                "duration": episode_duration,
                "pid_gains": pid_gains.copy(),
                "reward": total_reward,  # 🔥 세그먼트 평균 보상
                "metrics": final_metrics,
            }
            episode_stats.append(episode_stat)
            self.agent.episode_rewards.append(total_reward)

            # RewardBreakdownLogger 플러시 (에피소드 경계에서) - 그래프 생성은 최종에만
            if hasattr(self, "rlogger"):
                # 🆕 에피소드별 보상 구성 요소 로깅
                # band_ratio 계산: band_time / episode_len_s
                band_ratio = (
                    final_metrics.get("band_time", 0.0) / self.cfg["EPISODE_SECONDS"]
                    if self.cfg["EPISODE_SECONDS"] > 0
                    else 0.0
                )
                self.rlogger.log_episode_components(
                    episode=ep + 1,
                    reward_score=final_metrics.get("reward_score", 0.0),
                    r_centered=final_metrics.get("r_centered", 0.0),
                    r_baseline=final_metrics.get("r_baseline", 0.0),
                    overshoot=final_metrics.get("overshoot", 0.0),
                    settling_time=final_metrics.get("settling_time", 0.0),
                    band_ratio=band_ratio,
                    rmse=final_metrics.get("rmse", 0.0),
                    reward=total_reward,
                )
                
                self.rlogger.flush_if_needed(
                    ep + 1,
                    force=False,
                    episode_rewards=self.agent.episode_rewards,
                )

            # 전체 에피소드 최고 보상 추적 (통계용)
            if total_reward >= best_reward:
                best_reward = total_reward
            
            # 최고 성능 PID gain 저장 (50 에피소드 이후 최고 보상만 저장)
            if (ep + 1) >= Constants.MIN_EPISODES_FOR_SAVING:
                if total_reward >= best_reward_after_min:
                    best_reward_after_min = total_reward
                    best_pid_gains = pid_gains.copy()
                    self.agent.save_model(
                        f"{model_save_dir}/best_pid_agent_episode_{ep+1}_reward_{best_reward_after_min:.2f}.pth"
                    )
                    print(
                        f"💾 [저장] 최고 성능 에이전트 저장: 에피소드 {ep+1}, 보상 {best_reward_after_min:.2f}"
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
            self._log("INFO", f"🏆 보상: {total_reward:.2f}")
            self._log("INFO", f"📊 RMSE: {final_metrics['rmse']:.2f}")
            self._log("INFO", f"📈 오버슈트: {final_metrics['overshoot']:.1f}%")
            self._log("INFO", f"⏰ 정착시간: {final_metrics['settling_time']:.2f}s")
            self._log("INFO", f"🎯 밴드유지: {final_metrics['band_time']:.1f}s")
            # 50 에피소드 이후 최고 보상 표시 (저장 대상)
            if (ep + 1) >= Constants.MIN_EPISODES_FOR_SAVING:
                self._log("INFO", f"🏅 전체 최고보상: {best_reward:.2f}, 저장 대상 최고보상: {best_reward_after_min:.2f}")
            else:
                self._log("INFO", f"🏅 전체 최고보상: {best_reward:.2f} (저장 대기 중, {Constants.MIN_EPISODES_FOR_SAVING} 에피소드 이후 저장 시작)")

            # 10. 이전 에피소드 정보 업데이트
            self.previous_pid_gains = pid_gains.copy()

            # 에피소드 히스토리 업데이트 (최근 5개 에피소드만 유지)
            episode_record = {
                "episode": ep + 1,
                "pid_gains": pid_gains.copy(),
                "reward": total_reward,
                "metrics": final_metrics,
            }
            self.episode_history.append(episode_record)

            # 최근 5개 에피소드만 유지
            if len(self.episode_history) > self.max_history:
                self.episode_history.pop(0)

            print(
                f"📊 [히스토리] 에피소드 {ep+1} 기록: 보상={total_reward:.2f}, PID={pid_gains}"
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
                    f"🎯 다음 에피소드 PID: Kp={next_pid_gains[0]:.2f}, Ki={next_pid_gains[1]:.2f}, Kd={next_pid_gains[2]:.2f}"
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
        self._log("INFO", f"🏆 전체 최고 보상: {best_reward:.2f}")
        if episodes >= Constants.MIN_EPISODES_FOR_SAVING:
            self._log("INFO", f"💾 저장된 최고 보상 (에피소드 {Constants.MIN_EPISODES_FOR_SAVING} 이후): {best_reward_after_min:.2f}")
            if best_pid_gains is not None:
                self._log(
                    "INFO",
                    f"🎯 저장된 최적 PID: Kp={best_pid_gains[0]:.2f}, Ki={best_pid_gains[1]:.2f}, Kd={best_pid_gains[2]:.2f}",
                )
            else:
                self._log("WARNING", "⚠️ 저장된 최적 PID 없음 (50 에피소드 이후 최고 보상이 없었습니다)")
        else:
            self._log("WARNING", f"⚠️ 전체 에피소드가 {Constants.MIN_EPISODES_FOR_SAVING}개 미만이어서 모델이 저장되지 않았습니다")

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

        # 14. 최종 그래프 생성
        self.generate_episode_reward_graph(save_to_rlogger_folder=True)
        
        # 🆕 보상 구성 요소 그래프 생성 (학습 종료 시)
        if hasattr(self, "rlogger"):
            self.rlogger.flush_if_needed(
                episodes,
                force=True,  # 최종 그래프 생성
                episode_rewards=self.agent.episode_rewards,
            )

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

    def _build_segment_state(self, prev_pid_gains, prev_reward, force_segment, target_force, segment_len_s):
        """
        🆕 세그먼트별 상태 벡터 생성 (STATE_DIM 차원)
        
        기본 STATE_BASE_DIM 차원 + 궤적 요약 STATE_TRAJECTORY_DIM 차원
        
        Args:
            prev_pid_gains: [Kp, Ki, Kd]
            prev_reward: 이전 보상 (또는 현재 세그먼트 보상)
            force_segment: 힘 궤적 세그먼트 (예: 2000개 샘플 = 2초)
            target_force: 목표 힘
            segment_len_s: 세그먼트 길이 (초)
        
        Returns:
            state: STATE_DIM 차원 numpy array
        """
        import numpy as np
        
        # 1. 기본 차원 (로봇PC 6 + 강화학습PC 6)
        base_state = [
            prev_pid_gains[0],  # Kp
            prev_pid_gains[1],  # Ki
            prev_pid_gains[2],  # Kd
            prev_reward,
            target_force,
            segment_len_s,
        ]
        
        # 기존 6개 통계 (강화학습PC 계산)
        if force_segment and len(force_segment) >= 10:
            force = np.array(force_segment, dtype=np.float64)
            errors = force - target_force
            
            base_stats = [
                float(np.mean(force)),       # 평균
                float(np.std(force)),        # 표준편차
                float(np.min(force)),        # 최소
                float(np.max(force)),        # 최대
                float(np.mean(errors)),      # 평균 오차
                float(np.std(errors)),       # 오차 표준편차
            ]
        else:
            base_stats = [0.0] * 6
        
        base_state.extend(base_stats)  # STATE_BASE_DIM 차원 완성
        
        # 2. 🆕 궤적 요약 STATE_TRAJECTORY_DIM 차원 추가
        if not force_segment or len(force_segment) < 10:
            trajectory_features = [0.0] * Constants.STATE_TRAJECTORY_DIM
            return np.array(base_state + trajectory_features, dtype=np.float32)
        
        force = np.array(force_segment, dtype=np.float64)
        T_abs = max(abs(target_force), 1.0)
        errors = force - target_force
        
        # (1) Overshoot % 계산
        if target_force < 0:
            overshoot = max(0.0, (target_force - np.min(force)) / T_abs * 100.0)
        else:
            overshoot = max(0.0, (np.max(force) - target_force) / T_abs * 100.0)
        
        # (2) Settling Time (간단 버전)
        fs_hz = len(force) / max(segment_len_s, 1e-6)
        tol_settling = max(Constants.SETTLING_BAND_TOLERANCE, 0.01 * T_abs)
        in_settling = np.abs(errors) <= tol_settling
        settling_idx = len(force)
        
        hold_samples = int(fs_hz * Constants.SETTLING_HOLD_TIME_S)
        for i in range(len(in_settling) - hold_samples):
            if in_settling[i:i+hold_samples].all():
                settling_idx = i
                break
        settling_time = float(settling_idx / fs_hz)
        
        # (3) RMSE
        rmse = float(np.sqrt(np.mean(errors**2)))
        
        # (4) Band Ratio
        band_ratio = float(np.mean(np.abs(errors) <= Constants.BAND_TOLERANCE_N))
        
        # (5-6) 진동 주파수 & 진폭 (FFT)
        if len(errors) > 100:
            windowed = errors * np.hanning(len(errors))
            fft = np.fft.rfft(windowed)
            psd = np.abs(fft)**2
            freqs = np.fft.rfftfreq(len(errors), d=1.0/fs_hz)
            
            valid_idx = (freqs >= 0.1) & (freqs <= 50.0)
            if valid_idx.any():
                psd_valid = psd[valid_idx]
                freqs_valid = freqs[valid_idx]
                peak_idx = np.argmax(psd_valid)
                oscillation_freq = float(freqs_valid[peak_idx])
                oscillation_amp = float(np.sqrt(psd_valid[peak_idx]) / len(errors))
            else:
                oscillation_freq = 0.0
                oscillation_amp = 0.0
        else:
            oscillation_freq = 0.0
            oscillation_amp = 0.0
        
        # (7) Rise Time (10% → 90%)
        force_initial = float(force[0])
        force_final = float(np.mean(force[-max(1, int(len(force)*0.1)):]))
        target_10 = force_initial + 0.1 * (force_final - force_initial)
        target_90 = force_initial + 0.9 * (force_final - force_initial)
        
        try:
            if target_force < 0:
                idx_10 = np.where(force <= target_10)[0][0]
                idx_90 = np.where(force <= target_90)[0][0]
            else:
                idx_10 = np.where(force >= target_10)[0][0]
                idx_90 = np.where(force >= target_90)[0][0]
            rise_time = float(abs(idx_90 - idx_10) / fs_hz)
        except:
            rise_time = 0.0
        
        # (8) Steady State Error
        steady_state_error = float(force_final - target_force)
        
        # 통합
        trajectory_features = [
            overshoot,           # 1
            settling_time,       # 2
            rmse,                # 3
            band_ratio,          # 4
            oscillation_freq,    # 5
            oscillation_amp,     # 6
            rise_time,           # 7
            steady_state_error,  # 8
        ]
        
        state = np.array(base_state + trajectory_features, dtype=np.float32)
        
        # NaN/Inf 가드
        if np.isnan(state).any() or np.isinf(state).any():
            self._log("ERROR", f"❌ 상태 벡터에 NaN/Inf 발견! 제로 패딩 대체")
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        return state