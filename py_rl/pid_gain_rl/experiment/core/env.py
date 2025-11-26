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
    - 🆕 STATE_DIM 차원 상태 공간 (현재 힘 기반 6차원)
    """

    def __init__(self, cfg=None):
        if cfg is None:
            raise ValueError("cfg 파라미터가 필요합니다")
        
        # 🔥 STATE_DIM 설정 (Constants에서 가져오기)
        cfg["STATE_DIM"] = Constants.STATE_DIM
        
        print(f"✅ [Env] STATE_DIM 설정: {cfg['STATE_DIM']}차원 "
              f"(현재 힘, 목표 힘, 오차, 오차 미분/적분, PI 출력)")
        print(f"✅ [Env] 세그먼트 분할: {Constants.NUM_SEGMENTS}개 "
              f"({Constants.SEGMENT_LENGTH_S}초씩)")
        print(f"✅ [Env] Exploration settings: alpha={Constants.ACTOR_INITIAL_ALPHA}, "
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
        self.control_log_dir = os.path.join(self.ldlogger.log_dir, "control_log")
        os.makedirs(self.control_log_dir, exist_ok=True)
        # run log 디렉토리 및 파일
        self.run_log_dir = os.path.join(self.ldlogger.log_dir, "log")
        os.makedirs(self.run_log_dir, exist_ok=True)
        self.run_log_path = os.path.join(self.run_log_dir, "run.log")

        # 2. 나머지 Logger들은 learning_done 폴더 안에 서브폴더 생성
        self.cplogger = ControlPerformanceLogger(self.ldlogger.log_dir)
        self.rlogger = RewardBreakdownLogger(self.ldlogger.log_dir)

        # ==== ADDED: PID gain 최적화용 변수들 ====
        self.episode_force_data = []  # 에피소드 동안 힘 데이터 수집
        self.episode_pi_output_data = []  # 에피소드 동안 PI 출력 데이터 수집
        self.episode_state_history = []  # 에피소드 동안 전체 상태 기록
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

    def _run_log(self, message):
        try:
            with open(self.run_log_path, "a") as f:
                f.write(message + "\n")
        except Exception:
            # 파일 쓰기 실패해도 학습 중단하지 않음
            pass


    def calculate_episode_reward(
        self, force_data, pi_output_data, target_force=None, episode_len_s=None
    ):
        """
        힘 오차 기반 단순 보상 함수.
        평균 힘 오차(%)가 작을수록 보상이 크며, 보상 범위는 [-1, 1]을 유지한다.
        """
        import numpy as np

        if target_force is None:
            target_force = self.cfg["TARGET_FORCE"]
        if episode_len_s is None:
            episode_len_s = self.cfg["EPISODE_SECONDS"]

        if not force_data:
            metrics = {
                "mean_error": 0.0,
                "mean_error_pct": 0.0,
                "rmse": 0.0,
                "rmse_pct": 0.0,
                "overshoot": 0.0,
                "settling_time": episode_len_s,
                "band_ratio": 0.0,
                "band_time": 0.0,
                "out_of_band_time": episode_len_s,
                "pi_rms": 0.0,
                "reward": 0.0,
                "reward_score": 0.0,
                "r_centered": 0.0,
                "r_baseline": 0.0,
                "progress": 0.0,
                "I_improve": 0.0,
            }
            return 0.0, metrics

        force_array = np.array(force_data, dtype=np.float64)
        n_samples = len(force_array)
        target = float(target_force)
        target_abs = max(abs(target), 1.0)

        errors = force_array - target
        abs_errors = np.abs(errors)

        mean_error = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        mean_error_pct = float(mean_error / target_abs * 100.0)
        rmse_pct = float(rmse / target_abs * 100.0)

        ref_pct = max(Constants.REWARD_ERROR_REF_PERCENT, 1e-6)
        reward = float(np.clip(1.0 - 2.0 * (mean_error_pct / ref_pct), -1.0, 1.0))

        fs_hz = n_samples / max(episode_len_s, 1e-6)
        band_tol = Constants.BAND_TOLERANCE_N
        in_band = abs_errors <= band_tol
        band_ratio = float(np.mean(in_band)) if n_samples > 0 else 0.0
        band_time = band_ratio * episode_len_s
        out_of_band_time = (1.0 - band_ratio) * episode_len_s

        settling_time = episode_len_s
        if n_samples > 0:
            hold_samples = int(max(1, fs_hz * Constants.SETTLING_HOLD_TIME_S))
            if hold_samples < n_samples:
                for idx in range(n_samples - hold_samples):
                    if np.all(in_band[idx : idx + hold_samples]):
                        settling_time = idx / max(fs_hz, 1e-6)
                        break

        if target < 0:
            extreme_force = float(np.min(force_array))
            overshoot_pct = max(
                0.0, (target - extreme_force) / target_abs * 100.0
            )
        else:
            extreme_force = float(np.max(force_array))
            overshoot_pct = max(
                0.0, (extreme_force - target) / target_abs * 100.0
            )

        pi_rms = 0.0
        if pi_output_data:
            pi_arr = np.array(pi_output_data, dtype=np.float64)
            if pi_arr.size > 0:
                pi_rms = float(np.sqrt(np.mean(pi_arr ** 2)))

        metrics = {
            "mean_error": mean_error,
            "mean_error_pct": mean_error_pct,
            "rmse": rmse,
            "rmse_pct": rmse_pct,
            "overshoot": overshoot_pct,
            "settling_time": settling_time,
            "band_ratio": band_ratio,
            "band_time": band_time,
            "out_of_band_time": out_of_band_time,
            "pi_rms": pi_rms,
            "reward": reward,
            "reward_score": reward,
            "r_centered": reward,
            "r_baseline": 0.0,
            "progress": 0.0,
            "I_improve": 0.0,
        }

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

        # 안전 위반 체크 (절댓값 기준)
        if abs(current_force) > self.safety_force_limit:
            self._log(
                "WARNING",
                f"안전 위반: 힘 {current_force:.1f}N (한계: ±{self.safety_force_limit}N)",
            )
            return True, "safety_violation"

        # 에피소드는 시간으로만 종료
        return False, "time_based"

    # ---- PID Gain 최적화용 유틸리티 메서드들 ----

    def reset_episode(self):
        """에피소드 리셋 (PID gain 최적화용)"""
        self.episode_force_data = []
        self.episode_pi_output_data = []
        self.episode_state_history = []
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

        ep = 0
        while ep < episodes:
            # 에피소드별 리셋 없음 (연속 모니터링)

            self.episode_count = ep
            self._run_log(f"[Ep {ep+1}] start_wall={time.time():.3f}")
            
            # 🆕 Target Entropy 동적 조정
            self.agent.update_target_entropy(ep)
            
            # 🆕 학습률 스케줄
            self.agent.update_lr_schedule(ep)
            
            # 🆕 탐색 메트릭 로깅 (20 에피소드마다)
            if ep % 20 == 0 and ep > 0:
                exploration_metrics = self.agent.log_exploration_metrics(ep)
                if exploration_metrics:
                    self._log("INFO", 
                        f"📊 탐색 메트릭 [Ep {ep}]: "
                        f"std_ratio={exploration_metrics['action_std_ratio_pct']:.1f}%, "
                        f"Kd_coverage={exploration_metrics['kd_coverage_pct']:.1f}%, "
                        f"target_std_ratio~{exploration_metrics['target_std_ratio']:.1f}, "
                        f"std_scale={self.agent.std_scale:.2f}"
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
                    f"🎯 [에피소드 1] 로봇제어PC 자체 PID 사용: Kp={pid_gains[0]:.2f}, Ki={pid_gains[1]:.2f}, Kd={pid_gains[2]:.3f}"
                )
                self._log(
                    "INFO",
                    f"🎯 에피소드 1 기준 PID: Kp={pid_gains[0]:.2f}, Ki={pid_gains[1]:.2f}, Kd={pid_gains[2]:.3f}",
                )
            else:
                # 2번째 에피소드부터: 이전 에피소드 종료 시 전송한 PID 사용
                assert (
                    self.pid_gains_next is not None
                ), f"에피소드 {ep+1}: 이전 에피소드에서 next PID가 설정되지 않았습니다!"
                pid_gains = self.pid_gains_next.copy()
                print(
                    f"🤖 [에피소드 {ep+1}] PID: Kp={pid_gains[0]:.2f}, Ki={pid_gains[1]:.2f}, Kd={pid_gains[2]:.3f}"
                )

                # ⭐ 에피소드 시작 신호: episode_done=False 전송 (플래그 리셋)
                print("📤 에피소드 시작 신호 전송 (episode_done=False)")
                self.comm.send_pid_once(
                    pid_gains[0],
                    pid_gains[1],
                    pid_gains[2],
                    timing_accurate=True,
                    episode_done=False,
                    learning_done=False,
                )

            # 모니터에 현재 PID 표시
            monitor.post_pid(pid_gains)
            self._run_log(
                f"[Ep {ep+1}] PID Kp={pid_gains[0]:.2f} Ki={pid_gains[1]:.2f} Kd={pid_gains[2]:.3f}"
            )
            # 3. PID 적용 대기
            time.sleep(0.1)

            # 5. 새로운 PID gain으로 제어된 실제 상태 관측
            actual_state, _ = self.comm.get_latest_state()
            if actual_state is not None:
                actual_initial_state = np.array(actual_state, dtype=np.float32)
                self._log(
                    "INFO", f"📊 실제 상태 관측: Force={actual_state[0]:.2f}N"
                )
            else:
                actual_initial_state = initial_state
                print("⚠️  [경고] 실제 상태 관측 실패, 추정 상태 사용")
                self._log("WARNING", "실제 상태 관측 실패, 추정 상태 사용")

            # 6. 에피소드 시간 동안 1kHz 데이터 수집
            self.episode_force_data = []
            self.episode_pi_output_data = []
            self.episode_state_history = []
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
                            monitor.post_pi_output(
                                t_global,
                                float(state[5]),
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
                        break

            # sander_active 상승 에지를 기준으로 데이터 수집/보상 계산 시작
            self._log(
                "INFO",
                f"📊 sander_active 상승 에지 대기 후 {self.cfg['EPISODE_SECONDS']:.0f}초 1kHz 데이터 수집 시작...",
            )
            wall_start_time = time.perf_counter()
            activation_seen = False  # False→True 전환 감지 여부
            activation_time = None   # sander_active 상승 시각
            collect_start_time = None
            collect_end_time = None
            effective_start_time = None  # 워밍업 이후 실데이터 시작 시각
            duration_start_time = None
            self.episode_start_time = None
            
            data_count = 0
            prev_error = 0.0
            prev_pi_output = 0.0
            last_monitor_sent = 0.0  # 모니터 전송용 타이머
            activation_timeout = self.cfg["EPISODE_SECONDS"] + 60.0  # 상승 에지 대기 최대치

            dt = 0.001  # 1ms
            t_next = None

            episode_invalid = False  # 통신 이상 등으로 무효 처리 플래그
            repeat_count = 0
            prev_force_rounded = None
            monitor_window_s = 0.5
            monitor_start_s = 0.05  # 활성화 후 50ms 경과 후부터 검사
            repeat_threshold = 20  # 소수점 6째 자리 동일 force 20개 연속(≈20ms)이면 오류

            data_valid = True  # 목표 미도달 등으로 데이터 무효 시 False
            while True:
                now = time.perf_counter()

                # 종료 시점 도달 시 루프 탈출
                if (
                    effective_start_time is not None
                    and collect_end_time is not None
                    and now >= collect_end_time
                ):
                    break

                state, sander_active = self.comm.get_latest_state()
                if state is None:
                    time.sleep(0.001)
                    continue

                # 🚦 상승 에지 감지 전: sander_active True 되면 타임라인/버퍼 리셋 후 시작
                if not activation_seen:
                    if sander_active:
                        activation_seen = True
                        activation_time = now
                        collect_start_time = activation_time + Constants.WARMUP_SKIP_SECONDS
                        effective_collect_duration = max(
                            0.0, self.cfg["EPISODE_SECONDS"] - Constants.WARMUP_SKIP_SECONDS
                        )
                        collect_end_time = collect_start_time + effective_collect_duration
                        duration_start_time = collect_start_time
                        self.episode_start_time = collect_start_time

                        # 워밍업 이전 데이터는 폐기하고 새로 시작
                        self.episode_state_history.clear()
                        self.episode_force_data.clear()
                        self.episode_pi_output_data.clear()
                        data_count = 0
                        prev_error = 0.0
                        prev_pi_output = 0.0
                        t_next = None
                        self._log(
                            "INFO",
                            f"✅ sander_active 상승 감지 → 워밍업 {Constants.WARMUP_SKIP_SECONDS}s 후 수집 시작",
                        )
                    else:
                        # 상승 에지 대기 타임아웃
                        if (now - wall_start_time) > activation_timeout:
                            print("⚠️ sander_active 상승 에지 미검출 - 에피소드 스킵")
                            data_valid = False
                            break
                        continue

                # 활성화 후 다시 False로 내려가면 로봇 초기화 상태이므로 수집 종료
                if activation_seen and not sander_active:
                    print("⚠️ sander_active 하강 감지 - 데이터 수집 종료")
                    break

                # 통신 이상 감지 (초기 0.5초 동안 소수점 6째 자리 동일 force 10회 연속)
                if activation_time is not None:
                    elapsed = now - activation_time
                    if monitor_start_s <= elapsed <= monitor_window_s:
                        force_r = round(float(state[0]), 6)
                        if (
                            prev_force_rounded is not None
                            and force_r == prev_force_rounded
                        ):
                            repeat_count += 1
                            if repeat_count >= repeat_threshold:
                                episode_invalid = True
                                break
                        else:
                            repeat_count = 0
                        prev_force_rounded = force_r

                # 워밍업 구간 동안에는 기록하지 않음
                if effective_start_time is None:
                    if now < collect_start_time:
                        continue
                    effective_start_time = collect_start_time
                    t_next = now

                self.episode_state_history.append(np.array(state, dtype=np.float32))
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
                        next_pid_for_reset[2],
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
                        "duration": time.perf_counter() - duration_start_time,
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
                                monitor.post_pi_output(
                                    t_global,
                                    float(state[5]),
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
                if t_next is None:
                    t_next = time.perf_counter()
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
                    monitor.post_pi_output(
                        t_global,
                        float(state[5]),
                    )
                    last_monitor_sent = now

            if duration_start_time is None:
                duration_start_time = wall_start_time

            if episode_invalid:
                msg = f"Ep {ep+1}: 통신 이상(동일 force 반복) 감지 → 에피소드 무효, 재시작"
                print(f"⚠️ {msg}")
                self._log("WARNING", msg)
                self._run_log(f"[Ep {ep+1}] invalid (comm repeat)")
                # 다음 시도에서 같은 PID로 재시작
                self.pid_gains_next = pid_gains.copy()
                # 로봇 초기화 시퀀스 트리거 (일반 에피소드 종료와 동일 플래그 사용)
                try:
                    self.comm.send_pid_once(
                        self.pid_gains_next[0],
                        self.pid_gains_next[1],
                        self.pid_gains_next[2],
                        timing_accurate=True,
                        episode_done=True,
                        learning_done=False,
                    )
                except Exception:
                    pass
                # 버퍼 정리
                self.episode_force_data.clear()
                self.episode_state_history.clear()
                self.episode_pi_output_data.clear()
                # 로봇 리셋 대기 (2초) - 모니터 업데이트만 수행
                wait_start = time.perf_counter()
                while (time.perf_counter() - wait_start) < 2.0:
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
                            monitor.post_pi_output(
                                t_global,
                                float(state[5]),
                            )
                            last_monitor_sent = now
                    time.sleep(0.01)
                # 에피소드 번호 유지하고 다시 시도
                continue

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
                initial_state_violation = np.array(actual_initial_state, dtype=np.float32)

                self.agent.store_transition(
                    initial_state_violation,
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
            
            if (not data_valid) or (not force_data):
                print("⚠️ 목표 미도달/데이터 없음으로 학습 건너뜀")
                self._log("WARNING", f"에피소드 {ep+1}: 목표 미도달 또는 데이터 없음, 업데이트 스킵")
                self._run_log(f"[Ep {ep+1}] skip (no data)")
                total_reward = Constants.REWARD_MIN
                final_metrics = {
                    "rmse": 0.0,
                    "overshoot": 0.0,
                    "settling_time": 0.0,
                    "band_time": 0.0,
                    "band_ratio": 0.0,
                    "reward_score": total_reward,
                    "r_centered": total_reward,
                    "r_baseline": 0.0,
                }
                self.agent.episode_rewards.append(total_reward)
                self.pid_gains_next = pid_gains.copy()
                monitor.post_reward(ep + 1, float(total_reward))
                continue
            
            # 세그먼트 분할
            num_segments = Constants.NUM_SEGMENTS  # 5
            total_samples = len(force_data)
            segment_len = total_samples // num_segments
            
            state_history = self.episode_state_history
            segments_data = []
            
            for i in range(num_segments):
                start_idx = i * segment_len
                end_idx = (i + 1) * segment_len if i < num_segments - 1 else total_samples
                
                seg_force = force_data[start_idx:end_idx]
                seg_pi = pi_output_data[start_idx:end_idx] if pi_output_data else []
                seg_len_s = len(seg_force) / (total_samples / self.cfg["EPISODE_SECONDS"])
                seg_states = state_history[start_idx:end_idx] if state_history else []
                
                # 세그먼트 보상 계산
                seg_reward, seg_metrics = self.calculate_episode_reward(
                    seg_force, seg_pi, self.cfg["TARGET_FORCE"], seg_len_s
                )
                
                # 세그먼트 상태 생성 (6차원)
                seg_state = self._build_segment_state(seg_states)
                
                segments_data.append({
                    "state": seg_state,
                    "reward": seg_reward,
                    "metrics": seg_metrics,
                    "segment_idx": i,
                })
            
            # 리플레이 버퍼에 세그먼트별 저장
            for i, seg in enumerate(segments_data):
                if i == 0:
                    current_state = np.array(actual_initial_state, dtype=np.float32)
                else:
                    current_state = segments_data[i-1]["state"]
                
                if i < num_segments - 1:
                    next_state = segments_data[i + 1]["state"]
                    done = False
                else:
                    next_state = seg["state"]
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

            if (ep + 1) % 10 == 0:
                try:
                    self._export_control_trace(ep + 1, force_data, state_history, pid_gains)
                except Exception as e:
                    self._log(
                        "ERROR",
                        f"Failed to export control trace for episode {ep + 1}: {e}",
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
            episode_duration = time.perf_counter() - duration_start_time
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
                        kp, ki, kd = pid_gains
                        fname = (
                            f"best_pid_agent_ep{ep+1}_"
                            f"Kp{kp:.2f}_Ki{ki:.2f}_Kd{kd:.3f}_"
                            f"rew{best_reward_after_min:.2f}.pth"
                        )
                        self.agent.save_model(
                            f"{model_save_dir}/{fname}"
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
                
                self.agent.update_parameters_one_step(
                    effective_batch_size, actual_updates
                )
                updates_done = actual_updates
            else:
                # 초기 탐색 단계 (학습 없이 데이터만 수집)
                updates_done = 0

            # 9. 요약 로그 (터미널)
            summary = (
                f"[Ep {ep+1}] "
                f"PID Kp={pid_gains[0]:.2f} Ki={pid_gains[1]:.2f} Kd={pid_gains[2]:.3f} | "
                f"R={total_reward:.2f} | "
                f"RMSE={final_metrics['rmse']:.2f} ov={final_metrics['overshoot']:.1f}% "
                f"band={final_metrics['band_ratio']:.2f} | "
                f"buf={buffer_size} upd={updates_done}"
                f"{' (no-learn)' if updates_done==0 else ''} "
                f"std={self.agent.std_scale:.2f}"
            )
            print(summary)
            self._log("INFO", summary)

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
                # 다음 에피소드 PID 게인 선택
                forced_explore = ep < Constants.FORCED_RANDOM_EPISODES
                use_random = False  # 100ep 이후엔 ε-greedy 비활성화
                if ep < Constants.FORCED_RANDOM_EPISODES:
                    use_random = ((ep + 1) % 10 == 0) and (random.random() < 0.05)
                if forced_explore:
                    next_pid_gains = self.agent.select_action_random()
                    self._log(
                        "INFO",
                        f"🎲 초기 강제 탐색: 에피소드 {ep+1}까지 랜덤 PID 사용"
                    )
                elif use_random:
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
                    f"🎯 다음 에피소드 PID: Kp={next_pid_gains[0]:.2f}, Ki={next_pid_gains[1]:.2f}, Kd={next_pid_gains[2]:.3f}"
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
                self.pid_gains_next[2],
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
                            monitor.post_pi_output(
                                t_global,
                                float(state[5]),
                            )
                            last_monitor_sent = now
                    time.sleep(0.01)  # CPU 부하 방지

            ep += 1

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

    def _build_segment_state(self, state_segment):
        """
        세그먼트별 상태 벡터 생성 (STATE_DIM 차원 = 6)
        세그먼트 마지막 샘플을 사용하여 실시간 상태를 표현한다.
        """
        if state_segment:
            return np.array(state_segment[-1], dtype=np.float32)
        return np.zeros(Constants.STATE_DIM, dtype=np.float32)

    def _export_control_trace(self, episode_num, force_series, state_history, pid_gains):
        """
        10 에피소드마다 현재 힘/목표 힘 궤적을 그래프로 저장하고 원본 CSV를 남긴다.
        """
        if not force_series:
            self._log(
                "WARNING",
                f"Skipping force trace export for episode {episode_num}: empty force data (target not reached or timeout)",
            )
            return

        # 타임축: 워밍업 제외한 길이(기본 10초) - NameError 방지용 명확 변수 사용
        episode_seconds = max(
            0.0, self.cfg["EPISODE_SECONDS"] - Constants.WARMUP_SKIP_SECONDS
        )
        if episode_seconds <= 0.0:
            episode_seconds = float(self.cfg["EPISODE_SECONDS"])
        sample_count = len(force_series)
        time_axis = np.linspace(
            0.0, episode_seconds, sample_count, endpoint=False, dtype=np.float32
        )

        current_force = np.asarray(force_series, dtype=np.float32)

        if state_history and len(state_history) >= sample_count:
            target_force = np.asarray(
                [state[1] for state in state_history[:sample_count]],
                dtype=np.float32,
            )
        else:
            target_force = np.full(sample_count, self.cfg["TARGET_FORCE"], dtype=np.float32)

        kp, ki, kd = pid_gains
        episode_tag = f"ep_{episode_num:04d}_Kp{kp:.2f}_Ki{ki:.2f}_Kd{kd:.3f}"
        csv_path = os.path.join(self.control_log_dir, f"{episode_tag}_force_trace.csv")
        png_path = os.path.join(self.control_log_dir, f"{episode_tag}_force_trace.png")

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["time_s", "current_force_N", "target_force_N"])
            for t, cf, tf in zip(time_axis, current_force, target_force):
                writer.writerow([f"{t:.6f}", f"{cf:.6f}", f"{tf:.6f}"])

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        fig = Figure(figsize=(10, 5))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        current_abs = np.abs(current_force)
        target_abs = np.abs(target_force)
        ax.plot(time_axis, current_abs, color="black", linewidth=1.5, label="Current Force")
        ax.plot(time_axis, target_abs, color="red", linewidth=1.2, linestyle="--", label="Target Force")

        ax.set_title(f"Episode {episode_num} Force Trace", fontsize=14)
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("Force [N]", fontsize=12)
        ax.set_xlim(0.0, episode_seconds)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)

        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

        self._log(
            "INFO",
            f"Saved control trace for episode {episode_num}: {os.path.basename(png_path)} / {os.path.basename(csv_path)}",
        )
        self._run_log(f"[Ep {episode_num}] saved control_log ({os.path.basename(png_path)})")
