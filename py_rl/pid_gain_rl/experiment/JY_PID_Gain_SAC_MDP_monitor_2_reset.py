# PID Gain Optimization SAC Agent for Pneumatic Polishing System

import os
import random
import time
import socket
import struct
import threading
import signal
import sys
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 백엔드를 Agg로 변경 (GUI 불필요)
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from multiprocessing import Process, Queue
from queue import Empty, Full

# ========== RLRealtimeMonitor: 실시간 모니터링 (단일 윈도우) ==========
class RLRealtimeMonitor:
    def __init__(self, title="PID Gain RL Monitor", hz=10, rolling_window=60.0):
        self.hz = max(1, int(hz))
        self.rolling_window = float(rolling_window)
        self.q = Queue(maxsize=256)
        self.proc = None
        self.title = title
    def start(self):
        self.proc = Process(target=self._run, daemon=True)
        self.proc.start()

    def stop(self, timeout=2.0):
        if self.proc is None:
            return
        try:
            self.q.put_nowait({"type": "quit"})
        except Full:
            pass
        self.proc.join(timeout=timeout)

    def reset_force_buffers(self):
        try:
            self.q.put_nowait({"type": "reset_force"})
        except Full:
            pass

    def post_force(self, t_sec: float, current_f: float, desired_f: float):
        msg = {"type": "force", "t": float(t_sec), "cf": float(current_f), "df": float(desired_f)}
        try:
            self.q.put_nowait(msg)
        except Full:
            pass  # 최신만 유지

    def post_reward(self, episode: int, reward: float):
        msg = {"type": "reward", "ep": int(episode), "rew": float(reward)}
        try:
            self.q.put_nowait(msg)
        except Full:
            pass
    def _run(self):
        import matplotlib
        tk_ok = True
        try:
            import tkinter  # noqa: F401
            matplotlib.use("TkAgg")
        except Exception:
            matplotlib.use("Agg")
            tk_ok = False

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as np
        import matplotlib.gridspec as gridspec
        import time
        
        tbuf, cbuf, dbuf = [], [], []
        ep_idx, ep_rew = [], []

        if not tk_ok:
            print("[Monitor] TkAgg unavailable (headless). Realtime window is disabled.")
            last_note = 0.0
            while True:
                try:
                    msg = self.q.get(timeout=1.0)
                except Empty:
                    if time.time() - last_note > 5.0:
                        print("[Monitor] Headless mode: plotting disabled.")
                        last_note = time.time()
                    continue
                tp = msg.get("type")
                if tp == "quit":
                    break
            return
        fig = plt.figure(figsize=(9, 7.2))
        fig.suptitle(self.title)
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.0, 1.0], hspace=0.35)
        
        axF = fig.add_subplot(gs[0, 0])
        ln_c, = axF.plot([], [], label="current_force [N]")
        ln_d, = axF.plot([], [], linestyle="--", label="desired_force [N]")
        axF.set_xlim(0.0, self.rolling_window)
        axF.set_ylabel("force [N]")
        axF.set_xlabel("time [s]")
        axF.grid(True)
        axF.legend(loc="best")
        
        axR = fig.add_subplot(gs[1, 0])
        ln_r, = axR.plot([], [], marker="o", linewidth=1.0)
        axR.set_xlabel("episode")
        axR.set_ylabel("episode reward")
        axR.grid(True)

        interval_ms = int(1000 / self.hz)

        def on_timer(_frame):
            nonlocal tbuf, cbuf, dbuf
            while True:
                try:
                    msg = self.q.get_nowait()
                except Empty:
                    break

                tp = msg.get("type")
                if tp == "quit":
                    plt.close(fig)
                    return
                elif tp == "reset_force":
                    pass
                elif tp == "force":
                    t_new = float(msg["t"])
                    tbuf.append(t_new)
                    cbuf.append(float(msg["cf"]))
                    dbuf.append(float(msg["df"]))
                    while len(tbuf) > 0 and (t_new - tbuf[0]) > self.rolling_window:
                        tbuf.pop(0)
                        cbuf.pop(0)
                        dbuf.pop(0)
                elif tp == "reward":
                    ep_idx.append(int(msg["ep"]))
                    ep_rew.append(float(msg["rew"]))
            
            if len(tbuf) >= 2:
                t = np.asarray(tbuf, dtype=float)
                c = np.asarray(cbuf, dtype=float)
                d = np.asarray(dbuf, dtype=float)
                ln_c.set_data(t, c)
                ln_d.set_data(t, d)
                axF.relim(); axF.autoscale_view()
                t_max = t[-1]
                t_min = max(0.0, t_max - self.rolling_window)
                axF.set_xlim(t_min, t_max)
            
            if len(ep_idx) >= 1:
                ln_r.set_data(ep_idx, ep_rew)
                axR.relim(); axR.autoscale_view()
                axR.set_xlim(0, max(10, ep_idx[-1] + 1))

        anim = FuncAnimation(fig, on_timer, interval=interval_ms, blit=False)
        try:
            plt.show()
        except Exception:
            pass

# ========== 유틸리티 클래스 ==========
class Logger:
    @staticmethod
    def log(level, message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        icons = {"INFO": "ℹ️","SUCCESS": "✅","WARNING": "⚠️","ERROR": "❌","DEBUG": "🔍"}
        print(f"[{timestamp}] {icons.get(level, 'ℹ️')} {message}")

class DataSaver:
    @staticmethod
    def save_all_data(env, current_episode=None, force=True):
        try:
            if current_episode is not None:
                env.rlogger.flush_if_needed(current_episode, force=force, episode_rewards=env.agent.episode_rewards)
            else:
                env.rlogger.flush_if_needed(env.cfg["EPISODES"], force=force, episode_rewards=env.agent.episode_rewards)
        except Exception as e:
            Logger.log("ERROR", f"reward breakdown flush 실패: {e}")
        try:
            Logger.log("INFO", "📊 제어 성능 지표 저장 중...")
            env.cplogger.save_performance_summary()
            env.cplogger.generate_plots()
            Logger.log("INFO", "✅ 제어 성능 지표 저장 완료!")
        except Exception as e:
            Logger.log("ERROR", f"제어 성능 지표 저장 실패: {e}")
        Logger.log("INFO", "✅ 데이터 저장 완료!")

# ========== 상수 정의 ==========
class Constants:
    DEFAULT_HIDDEN_DIM = 256
    DEFAULT_LR = 1e-3  # 학습률 증가: 더 적극적인 탐색
    DEFAULT_GAMMA = 0.99
    DEFAULT_TAU = 0.01
    DEFAULT_PID_RANGE = {"Kp": (40.0, 120.0), "Ki": (70.0, 200.0), "Kd": (0.0, 0.1)}  # P,I 범위 확대, D=0~0.1 학습
    DEFAULT_RECV_FREQ = 1000
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_EPISODES = 500
    DEFAULT_EPISODE_SECONDS = 40.0  # 에피소드 길이 증가: relaxation 관찰을 위해
    DEFAULT_TARGET_FORCE = -45.0  # FT 센서 좌표계: 압축력이 음수
    DEFAULT_UPDATES_PER_EPISODE = 16  # 업데이트 횟수 감소: 탐색 활성화
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = 8888
    DEFAULT_RECV_TIMEOUT = 0.5
    DEFAULT_RECV_LOOP_TIMEOUT = 0.05
    DEFAULT_COMM_FAIL_MAX = 3
    DEFAULT_COMM_RETRY_DELAY = 0.1
    DEFAULT_MAX_REWARDS_HISTORY = 1000
    DEFAULT_REPLAY_BUFFER_SIZE = 4000
    DEFAULT_MODEL_SAVE_DIR = "/home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl/saved_agents"
    DEFAULT_LOG_DIR = "/home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl/experiment_logs"
    WAIT_MESSAGE_INTERVAL = 1.0
    DEFAULT_FORCE_VALUE = -30.0
    TAU_RMSE = 3.0
    TAU_SETTLE = 5.0
    TAU_VAR = 0.1
    TAU_U = 1.0
    TAU_DU = 1.0
    POTENTIAL_GAMMA = 0.99
    SHAPING_WARMUP_S = 0.5
    REWARD_WEIGHT_PROGRESS = 0.05
    REWARD_MIN = -100.0
    REWARD_MAX = 50.0
    BAND_TOLERANCE_N = 1.5
    SETTLING_BAND_TOLERANCE = 0.5
    SETTLING_HOLD_TIME_S = 1.0
    SAFETY_FORCE_LIMIT = 100.0
    PI_OUTPUT_MAX = 0.4
    PI_OUTPUT_SAT_THRESHOLD = 0.95

# ========== CONFIG 설정 ==========
_BASE_CONFIG = {
    "STATE_DIM": 12,
    "ACTION_DIM": 3,  # P, I, D 모두 학습 (D=0~1)
    "HIDDEN": Constants.DEFAULT_HIDDEN_DIM,
    "LR": Constants.DEFAULT_LR,
    "GAMMA": Constants.DEFAULT_GAMMA,
    "TAU": Constants.DEFAULT_TAU,
    "AUTO_ENTROPY": True,
    "PID_RANGE": Constants.DEFAULT_PID_RANGE,
    "EPISODE_SECONDS": Constants.DEFAULT_EPISODE_SECONDS,
    "TARGET_FORCE": Constants.DEFAULT_TARGET_FORCE,
    "UPDATES_PER_EPISODE": Constants.DEFAULT_UPDATES_PER_EPISODE,
    "RECV_FREQ_HZ": Constants.DEFAULT_RECV_FREQ,
    "BATCH_SIZE": Constants.DEFAULT_BATCH_SIZE,
    "HOST": Constants.DEFAULT_HOST,
    "PORT": Constants.DEFAULT_PORT,
    "RECV_TIMEOUT_SEC": Constants.DEFAULT_RECV_TIMEOUT,
    "RECV_LOOP_TIMEOUT_SEC": Constants.DEFAULT_RECV_LOOP_TIMEOUT,
    "COMM_FAIL_MAX": Constants.DEFAULT_COMM_FAIL_MAX,
    "COMM_RETRY_DELAY": Constants.DEFAULT_COMM_RETRY_DELAY,
    "EPISODES": Constants.DEFAULT_EPISODES,
    "MODEL_SAVE_DIR": Constants.DEFAULT_MODEL_SAVE_DIR,
    "LOG_DIR": Constants.DEFAULT_LOG_DIR,
    "MAX_EPISODE_REWARDS_HISTORY": Constants.DEFAULT_MAX_REWARDS_HISTORY,
    "REPLAY_BUFFER_SIZE": Constants.DEFAULT_REPLAY_BUFFER_SIZE,
}

def create_config(recv_freq_hz=None, episode_seconds=None):
    """
    설정 생성 함수
    Args:
        recv_freq_hz: 수신 주파수 (Hz)
        episode_seconds: 에피소드 길이 (초)
    """
    config = _BASE_CONFIG.copy()
    
    if recv_freq_hz is not None:
        if recv_freq_hz <= 0 or recv_freq_hz > 10000:
            raise ValueError(f"수신 주파수는 0과 10000 사이여야 합니다: {recv_freq_hz}")
        config["RECV_FREQ_HZ"] = recv_freq_hz
        config["RECV_INTERVAL_SEC"] = 1.0 / recv_freq_hz
    
    if episode_seconds is not None:
        if episode_seconds <= 0:
            raise ValueError(f"에피소드 길이는 0보다 커야 합니다: {episode_seconds}")
        config["EPISODE_SECONDS"] = episode_seconds
    
    return config

CONFIG = create_config()

# ========== 에피소드 길이 변경 유틸리티 ==========
def change_episode_length(config, new_length_seconds):
    """
    에피소드 길이를 동적으로 변경하는 함수
    Args:
        config: 현재 설정 딕셔너리
        new_length_seconds: 새로운 에피소드 길이 (초)
    Returns:
        업데이트된 설정 딕셔너리
    """
    if new_length_seconds <= 0:
        raise ValueError(f"에피소드 길이는 0보다 커야 합니다: {new_length_seconds}")
    
    config["EPISODE_SECONDS"] = new_length_seconds
    print(f"🔄 에피소드 길이 변경: {new_length_seconds}초")
    print(f"📊 목표 데이터 개수: {int(new_length_seconds * 1000)}개")
    return config

# ========== PID Gain 유틸리티 ==========
def scale_action_to_pid(action, pid_range):
    def scale_single(v, lo, hi):
        return lo + (v + 1.0) * 0.5 * (hi - lo)
    return np.array([
        scale_single(action[0], *pid_range["Kp"]),
        scale_single(action[1], *pid_range["Ki"]),
        scale_single(action[2], *pid_range["Kd"]),  # D값도 학습 (0~1)
    ], dtype=np.float32)

def create_initial_state(force_data, target_force=-45.0, previous_pid_gains=None, historical_errors=None, episode_history=None):
    if not force_data:
        if previous_pid_gains is not None:
            return _estimate_state_from_previous_pid(previous_pid_gains, target_force, historical_errors, episode_history)
        else:
            print("🆕 [첫 에피소드] 기본 상태로 시작 (데이터 없음)")
            return np.array([
                target_force, target_force, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)
    
    force_arr = np.asarray(force_data, dtype=np.float32)
    current_force = force_arr[-1]
    error = current_force - target_force
    all_errors = force_arr - target_force
    
    if all_errors.size > 0:
        recent_error_avg = float(np.mean(np.abs(all_errors)))
        recent_error_std = float(np.std(all_errors)) if all_errors.size > 1 else 0.0
    else:
        recent_error_avg, recent_error_std = 0.0, 0.0
    
    if force_arr.size >= 2:
        error_dot = float((force_arr[-1] - force_arr[-2]) / 0.001)
    else:
        error_dot = 0.0
    
    error_int = float(np.sum(np.abs(all_errors)) * 0.001)
    performance_trend = avg_recent_performance = pid_variance = episode_count = 0.0
    
    if episode_history is not None and len(episode_history) > 0:
        recent_rewards = [ep['reward'] for ep in episode_history[-5:]]
        recent_pids = [ep['pid_gains'] for ep in episode_history[-5:]]
        if len(recent_rewards) >= 2:
            performance_trend = recent_rewards[-1] - recent_rewards[-2]
            avg_recent_performance = np.mean(recent_rewards)
        if len(recent_pids) >= 2:
            pid_changes = np.array(recent_pids[-1]) - np.array(recent_pids[-2])
            pid_variance = np.var(pid_changes)
        episode_count = len(episode_history)
    
    return np.array([
        current_force, target_force, error, error_dot, error_int, 0.0,
        recent_error_avg, recent_error_std, performance_trend,
        avg_recent_performance, pid_variance, episode_count
    ], dtype=np.float32)
def _estimate_state_from_previous_pid(previous_pid_gains, target_force, historical_errors=None, episode_history=None):
    Kp, Ki, Kd = previous_pid_gains
    if episode_history is not None and len(episode_history) > 0:
        recent_rewards = [ep['reward'] for ep in episode_history[-5:]]
        recent_pids = [ep['pid_gains'] for ep in episode_history[-5:]]
        if len(recent_rewards) >= 2:
            performance_trend = recent_rewards[-1] - recent_rewards[-2]
            avg_recent_performance = np.mean(recent_rewards)
        else:
            performance_trend = 0.0
            avg_recent_performance = recent_rewards[0] if recent_rewards else 0.0
        if len(recent_pids) >= 2:
            pid_changes = np.array(recent_pids[-1]) - np.array(recent_pids[-2])
        else:
            pid_changes = np.array([0.0, 0.0, 0.0])
        pid_variance = np.var(pid_changes)
        episode_count = len(episode_history)
        if avg_recent_performance > 0:
            avg_error, error_std = -target_force * 0.05, target_force * 0.02
        else:
            avg_error, error_std = -target_force * 0.15, target_force * 0.08
    else:
        avg_error, error_std = -target_force * 0.1, target_force * 0.05
        performance_trend = avg_recent_performance = pid_variance = episode_count = 0.0
    
    estimated_force = target_force + avg_error
    return np.array([
        estimated_force, target_force, avg_error, 0.0, 0.0, 0.0,
        abs(avg_error), error_std, performance_trend,
        avg_recent_performance, pid_variance, episode_count
    ], dtype=np.float32)

# ========== SAC Models ==========
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-5, log_std_max=2):  # 더욱 넓은 탐색 범위
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        """최적화된 가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc4 = nn.Linear(hidden_dim, 1)
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc4 = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = F.relu(self.q1_fc3(q1))
        q1 = self.q1_fc4(q1)
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = F.relu(self.q2_fc3(q2))
        q2 = self.q2_fc4(q2)
        return q1, q2

# ========== Replay Buffer ==========
class ReplayBuffer:
    def __init__(self, capacity=None):
        self.buffer = deque(maxlen=capacity or Constants.DEFAULT_REPLAY_BUFFER_SIZE)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (state, action.reshape(-1, 1) if action.ndim == 1 else action, reward, next_state, done)
    
    def __len__(self):
        return len(self.buffer)

# ========== SAC Agent ==========
class PIDGainSACAgent:
    def __init__(self, cfg=None):
        if cfg is None:
            raise ValueError("cfg 파라미터가 필요합니다")
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s_dim, a_dim, hidden = cfg["STATE_DIM"], cfg["ACTION_DIM"], cfg["HIDDEN"]
        self.gamma, self.tau = cfg["GAMMA"], cfg["TAU"]
        self.alpha = 0.05
        self.auto_entropy_tuning = cfg["AUTO_ENTROPY"]
        self.actor = Actor(s_dim, a_dim, hidden).to(self.device)
        self.critic = Critic(s_dim, a_dim, hidden).to(self.device)
        self.critic_target = Critic(s_dim, a_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg["LR"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg["LR"])
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.tensor([a_dim], device=self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg["LR"])
        self.replay = ReplayBuffer(cfg.get("REPLAY_BUFFER_SIZE", Constants.DEFAULT_REPLAY_BUFFER_SIZE))
        self.total_steps = 0
        self.episode_rewards = []
        self.max_rewards_history = cfg.get("MAX_EPISODE_REWARDS_HISTORY", Constants.DEFAULT_MAX_REWARDS_HISTORY)

    def select_action(self, state, evaluate=False):
        """
        PID gain 액션 선택 (에피소드당 한 번)
        Args:
            state: 12차원 상태 벡터 [0-5: 로봇제어PC 전송, 6-11: 강화학습PC 계산]
            evaluate: 평가 모드 여부
        Returns:
            pid_gains: [Kp, Ki, Kd] 실제 PID gain 값들
            log_prob: 로그 확률 (학습용)
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
                log_prob = None
            else:
                action, log_prob = self.actor.sample(state)
        
        action_np = action.cpu().numpy().flatten()
        pid_gains = scale_action_to_pid(action_np, self.cfg["PID_RANGE"])
        
        if log_prob is not None:
            log_prob = log_prob.cpu().numpy()
            return pid_gains, log_prob
        else:
            return pid_gains, None
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        PID gain transition 저장 (에피소드당 한 개)
        Args:
            state: 초기 상태 (12차원) [0-5: 로봇제어PC 전송, 6-11: 강화학습PC 계산]
            action: PID gain 액션 [Kp, Ki, Kd] (3차원 그대로 사용)
            reward: 에피소드 총보상
            next_state: 최종 상태 (요약 또는 zero)
            done: 에피소드 종료 여부 (항상 True)
        """
        # PID gain을 [Kp, Ki, Kd] 모두 사용 (3차원)
        norm_action = self._normalize_pid_action(action)
        self.replay.push(state, norm_action, reward, next_state, done)

    def _normalize_pid_action(self, pid_action):
        """PID gain을 [-1, 1] 범위로 정규화 (Kp, Ki, Kd 모두)"""
        def normalize_single(v, lo, hi):
            return 2.0 * (v - lo) / (hi - lo) - 1.0

        return np.array([
            normalize_single(pid_action[0], *self.cfg["PID_RANGE"]["Kp"]),
            normalize_single(pid_action[1], *self.cfg["PID_RANGE"]["Ki"]),
            normalize_single(pid_action[2], *self.cfg["PID_RANGE"]["Kd"]),
        ], dtype=np.float32)

    def update_parameters_one_step(self, batch_size=None, num_updates=128):
        """
        한 스텝 MDP에 최적화된 SAC 업데이트
        Args:
            batch_size: 배치 크기 (None이면 replay buffer 크기에 맞춰 동적 조정)
            num_updates: 업데이트 횟수
        """
        # 동적 배치 크기: min(설정 배치 크기, 현재 버퍼 크기)
        bs = min(batch_size or self.cfg["BATCH_SIZE"], len(self.replay))
        
        # 최소 2개 이상 있어야 학습 가능
        if len(self.replay) < 2: 
            return

        for _ in range(num_updates):
            s, a, r, ns, d = self.replay.sample(bs)
            s = torch.FloatTensor(s).to(self.device)
            a = torch.FloatTensor(a).to(self.device)
            r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
            ns = torch.FloatTensor(ns).to(self.device)
            d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

            # 한 스텝 MDP이므로 y = r (부트스트랩 없음)
            with torch.no_grad():
                y = r

            # Critic 업데이트
            q1, q2 = self.critic(s, a)
            q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            self.critic_opt.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)  # Gradient clipping 증가
            self.critic_opt.step()

            # Actor 업데이트
            pi, logp = self.actor.sample(s)
            q1_pi, q2_pi = self.critic(s, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            pi_loss = ((self.alpha * logp) - min_q_pi).mean()
            self.actor_opt.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)  # Gradient clipping 증가
            self.actor_opt.step()

            # 엔트로피 자동 조절
            if self.auto_entropy_tuning:
                logp_entropy = logp.squeeze(1)
                a_loss = -(self.log_alpha * (logp_entropy + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad()
                a_loss.backward()
                self.alpha_opt.step()
                self.alpha = self.log_alpha.exp()

            # 타겟 네트워크 소프트 업데이트
            with torch.no_grad():
                for tp, lp in zip(self.critic_target.parameters(), self.critic.parameters()):
                    tp.data.copy_(self.tau * lp.data + (1 - self.tau) * tp.data)

    def save_model(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_steps": self.total_steps,
            "episode_rewards": self.episode_rewards,
        }, path)
        print(f"💾 Saved: {path}")

# =========================
# TCP Communicator
# =========================
class PIDGainCommunicator:
    """
    로봇 제어 PC와의 TCP 통신 관리
    - PID gain 전송 (에피소드당 한 번)
    - 실시간 상태 데이터 수신 (1kHz)
    - 연결 상태 모니터링
    """
    def __init__(self, host, port, recv_timeout, recv_loop_timeout=0.05, cfg=None):
        self.host, self.port = host, port 
        self.recv_timeout = recv_timeout
        self.recv_loop_timeout = recv_loop_timeout
        self.cfg = cfg
        self.socket = None
        self.conn = None
        self.connected = False
        
        self.CPP_TO_PY_PACKET_FORMAT = ">HffffffBH"
        self.CPP_TO_PY_PACKET_SIZE = 29
        self.CPP_TO_PY_SOF = 0xAAAA
        # self.PY_TO_CPP_PACKET_FORMAT = ">HfBBBH"  # SOF, rl_residual, timing_accurate, episode_done, learning_done, checksum (미사용)
        # self.PY_TO_CPP_PACKET_SIZE = 11  # SOF(2) + rl_residual(4) + timing_accurate(1) + episode_done(1) + learning_done(1) + checksum(2) = 11 bytes (미사용)
        # self.PY_TO_CPP_SOF = 0xBBBB  # (미사용)
        
        # PID gain 전송용 패킷 포맷
        self.PID_PACKET_FORMAT = ">HfffBBBH"  # SOF, Kp, Ki, Kd, timing_accurate, episode_done, learning_done, checksum
        self.PID_PACKET_SIZE = 19  # SOF(2) + Kp(4) + Ki(4) + Kd(4) + timing(1) + ep_done(1) + learn_done(1) + checksum(2) = 19 bytes
        self.PID_SOF = 0xBBBB  # 잔차학습과 동일한 SOF 사용
        self.latest_state = None
        self.latest_sander_active = False
        self.receive_thread = None
        self.is_receiving = False
        self.state_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.packets_received = 0
        self.packets_sent = 0
        self.connection_start_time = None
        self.last_packet_time = None
        self.consecutive_failures = 0
        self.old_data_warning_logged = False  # 오래된 데이터 경고 중복 방지

    def _log(self, level, message):
        Logger.log(level, message)

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.settimeout(1.0)
            self._log("INFO", f"로봇제어PC 연결 대기 중 {self.host}:{self.port} ...")
            while True:
                try:
                    conn, addr = self.socket.accept()
                    break
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    self._log("WARNING", "사용자에 의해 연결 취소됨")
                    return False
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            conn.settimeout(self.recv_timeout)
            self._log("SUCCESS", f"연결 성공: {addr}")
            self.conn = conn
            self.connected = True
            self.connection_start_time = time.perf_counter()
            self.start_receiving()
            return True
        except KeyboardInterrupt:
            self._log("WARNING", "사용자에 의해 연결 취소됨")
            return False
        except Exception as e:
            self._log("ERROR", f"연결 오류: {e}")
            return False
        
    def start_receiving(self):
        self.is_receiving = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        self._log("INFO", f"{self.cfg['RECV_FREQ_HZ']}Hz 수신 스레드 시작")

    def _receive_loop(self):
        next_receive_time = time.perf_counter()
        recv_interval = self.cfg["RECV_INTERVAL_SEC"]
        while self.is_receiving:
            current_time = time.perf_counter()
            if current_time >= next_receive_time:
                next_receive_time += recv_interval
                try:
                    self.conn.settimeout(self.recv_loop_timeout)
                    data = self._recv_exact(self.CPP_TO_PY_PACKET_SIZE)
                    if data:
                        state, sander_active = self._process_packet(data)
                        if state is not None:
                            with self.state_lock:
                                self.latest_state = state
                                self.latest_sander_active = sander_active
                                self.last_packet_time = time.perf_counter()
                            self.consecutive_failures = 0
                except socket.timeout:
                    pass
                except Exception as e:
                    self.consecutive_failures += 1
                    self._log("WARNING", f"수신 루프 오류 ({self.consecutive_failures}회): {e}")
                    if self.consecutive_failures >= 5:
                        self._log("ERROR", "연속 수신 실패로 수신 루프 중단")
                        break
                    time.sleep(self.cfg["RECV_INTERVAL_SEC"])
            else:
                time.sleep(0.001)
        self._log("INFO", "수신 루프 종료")

    def _recv_exact(self, nbytes):
        data = b''
        while len(data) < nbytes:
            chunk = self.conn.recv(nbytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def _process_packet(self, data):
        try:
            if len(data) != self.CPP_TO_PY_PACKET_SIZE:
                self._log("WARNING", f"예상 {self.CPP_TO_PY_PACKET_SIZE}B, 수신 {len(data)}B")
                return None, False
            try:
                (sof, current_force, target_force, force_error, force_error_dot, 
                 force_error_int, pi_output, sander_active, 
                 received_checksum) = struct.unpack(">HffffffBH", data)
            except struct.error as e:
                self._log("ERROR", f"패킷 언팩 실패: {e}")
                return None, False
            if sof != self.CPP_TO_PY_SOF:
                self._log("WARNING", f"SOF 불일치: {hex(sof)} (예상: {hex(self.CPP_TO_PY_SOF)})")
                return None, False
            calculated_checksum = self.calculate_crc16(data[:-2])
            if received_checksum != calculated_checksum:
                self._log("ERROR", f"체크섬 오류: 수신:{received_checksum} 계산:{calculated_checksum}")
                return None, False
            state = np.array([
                current_force,      # 0
                target_force,       # 1 
                force_error,        # 2
                force_error_dot,    # 3
                force_error_int,    # 4
                pi_output,          # 5
            ], dtype=np.float32)
            sander_active = bool(sander_active)
            
            
            with self.stats_lock:
                self.packets_received += 1
            return state, sander_active
        except Exception as e:
            self._log("ERROR", f"패킷 처리 오류: {e}")
            return None, False
        
    def calculate_crc16(self, data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc = crc >> 1
        return crc
    
    def get_latest_state(self):
        with self.state_lock:
            if self.latest_state is not None:
                current_time = time.perf_counter()
                if (self.last_packet_time and 
                    current_time - self.last_packet_time > 2.0):
                    # 오래된 데이터 경고는 한 번만 출력
                    if not self.old_data_warning_logged:
                        self._log("WARNING", f"오래된 데이터 감지: {current_time - self.last_packet_time:.2f}초 전")
                        self.old_data_warning_logged = True
                else:
                    # 데이터가 정상이면 경고 플래그 리셋
                    self.old_data_warning_logged = False
                    
                if hasattr(self, 'last_logged_sander_active') and self.last_logged_sander_active != self.latest_sander_active:
                    self._log("DEBUG", f"RL 플래그 변경: {self.last_logged_sander_active} -> {self.latest_sander_active}")
                    self.last_logged_sander_active = self.latest_sander_active
                elif not hasattr(self, 'last_logged_sander_active'):
                    self.last_logged_sander_active = self.latest_sander_active
                    self._log("DEBUG", f"초기 RL 플래그: {self.latest_sander_active}")
                return self.latest_state.copy(), self.latest_sander_active
        return None, False
    
        
    def send_pid_once(self, kp, ki, kd, timing_accurate=True, episode_done=False, learning_done=False):
        """
        PID gain 전송
        - 첫 에피소드: 에피소드 시작 시 전송
        - 이후 에피소드: 이전 에피소드 종료 시 다음 에피소드 PID를 미리 전송
        Args:
            kp, ki, kd: PID gain 값들
            timing_accurate: 타이밍 정확성
            episode_done: 에피소드 종료 플래그 (True면 다음 에피소드 PID 포함)
            learning_done: 학습 종료 플래그
        """
        try:
            payload = struct.pack(">HfffBBB", 
                                  self.PID_SOF, 
                                  float(kp), 
                                  float(ki), 
                                  float(kd),
                                  bool(timing_accurate), 
                                  bool(episode_done),
                                  bool(learning_done))
            checksum = self.calculate_crc16(payload)
            final_packet = struct.pack(self.PID_PACKET_FORMAT, 
                                     self.PID_SOF, 
                                     float(kp), 
                                     float(ki), 
                                     float(kd),
                                     bool(timing_accurate), 
                                     bool(episode_done),
                                     bool(learning_done),
                                     checksum)
            self.conn.sendall(final_packet)
            with self.stats_lock:
                self.packets_sent += 1
            self._log("INFO", f"📡 PID gain 전송: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}")
            return True
        except Exception as e:
            self._log("ERROR", f"PID gain 전송 오류: {e}")
            return False
        
    def send_reset(self):
        try:
            reset_data = struct.pack(">HBxxxH", 0xBBBB, 1, 0)
            checksum = self.calculate_crc16(reset_data[:-2])
            reset_packet = struct.pack(">HBxxxH", 0xBBBB, 1, checksum)
            self.conn.sendall(reset_packet)
            return True
        except Exception as e:
            self._log("ERROR", f"리셋 전송 오류: {e}")
            return False
        
    def get_communication_stats(self):
        uptime = time.perf_counter() - self.connection_start_time if self.connection_start_time else 0
        with self.stats_lock:
            packets_received = self.packets_received
            packets_sent = self.packets_sent
        return {
            "uptime_seconds": uptime,
            "packets_received": packets_received,
            "packets_sent": packets_sent,
            "receive_rate_hz": packets_received / uptime if uptime > 0 else 0,
            "send_rate_hz": packets_sent / uptime if uptime > 0 else 0,
        }
    def print_communication_stats(self):
        stats = self.get_communication_stats()
        self._log("INFO", "\n📊 === 통신 통계 ===")
        self._log("INFO", f"⏱️  가동 시간: {stats['uptime_seconds']:.1f}s")
        self._log("INFO", f"📥 수신된 패킷: {stats['packets_received']}")
        self._log("INFO", f"📤 송신된 패킷: {stats['packets_sent']}")
        self._log("INFO", f"📥 수신률: {stats['receive_rate_hz']:.1f} Hz")
        self._log("INFO", f"📤 송신률: {stats['send_rate_hz']:.1f} Hz")
        self._log("INFO", "=" * 40)

    def close(self):
        try:
            self.is_receiving = False
            if self.receive_thread and self.receive_thread.is_alive():
                self.receive_thread.join(timeout=1.0)
            if self.conn: 
                self.conn.close()
            if self.socket: 
                self.socket.close()
        finally:
            self.connected = False
            self._log("INFO", "통신 종료")

# =========================
# Environment
# =========================
class PIDGainOptimizationEnvironment:
    """
    PID Gain 최적화 환경
    - 에피소드 실행 및 관리
    - 보상 계산 (18개 제어공학 지표)
    - 데이터 수집 및 저장
    - 학습 진행 모니터링
    """
    def __init__(self, cfg=None):
        if cfg is None:
            raise ValueError("cfg 파라미터가 필요합니다")
        self.cfg = cfg
        self.agent = PIDGainSACAgent(cfg)
        self.comm = PIDGainCommunicator(cfg["HOST"], cfg["PORT"], cfg["RECV_TIMEOUT_SEC"], cfg["RECV_LOOP_TIMEOUT_SEC"], cfg)
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
        self.pid_gains_next = None  # 다음 에피소드에 실제로 적용할 PID (미리 전송한 값)
        self.historical_errors = []  # 이전 에피소드들의 에러 통계
        self.episode_count = 0  # 에피소드 카운터
        
        # ==== ADDED: 이전 5개 에피소드 성능 히스토리 (최적값) ====
        self.episode_history = []  # 최근 5개 에피소드의 (PID, 성능) 기록
        self.max_history = 5  # 최대 5개 에피소드 히스토리 유지 (최적값)
        
        # ==== [ADD] RELAX detection params ====
        self.relax_guard_s = 2.0          # PID on + 2.0s 이후부터 감시
        self.relax_window_s = 2.5         # 연속 판정 창 (s)
        self.relax_pressure_thr = 0.38    # MPa (고압 유지 기준)
        self.relax_slope_thr = -2.5       # N/s (힘 하강 기울기 기준: 1초에 2.5N 이상 감소)
        self.relax_low_pressure_thr = 0.02 # MPa (저압 유지 기준)
        self.relax_force_target_tol = 5.0  # N (목표값 수렴 허용 오차)
        self.relax_detected = False

        # 최근 창(2.5s) 버퍼: 1kHz 기준 2500 샘플
        self._relax_win_N = int(self.relax_window_s * 1000)
        self.relax_force_win = deque(maxlen=self._relax_win_N)
        self.relax_time_win  = deque(maxlen=self._relax_win_N)
        self.relax_pi_win    = deque(maxlen=self._relax_win_N)
        self._relax_pid_on_ts = None
        
    def _log(self, level, message):
        Logger.log(level, message)
    
    # ==== [ADD] RELAX utils ====
    def _relax_reset_windows(self):
        """RELAX 감지 윈도우 초기화"""
        self.relax_force_win.clear()
        self.relax_time_win.clear()
        self.relax_pi_win.clear()
        self.relax_detected = False
        self._relax_pid_on_ts = None

    def _relax_update_and_check(self, now_ts: float) -> bool:
        """
        RELAX 감지 로직:
        A1: pi_output >= relax_pressure_thr 이 창 구간의 ≥95% (고압 + 오차 증가)
        A2: desired force와 current force의 절대값 오차가 지속적으로 커질 때
        A3: pi_output <= relax_low_pressure_thr 이 창 구간의 ≥95% (저압 + 수렴 안 됨)
        
        guard 시간(PID on + 2.0s)과 창 길이(2.5s) 조건을 모두 만족해야 판정.
        
        ⚠️ 센서 좌표계 고려: 누르는 방향이 음수 (예: -45N)
        → Relaxation 시: -45N → -44N → -43N (절댓값이 감소)
        """
        # guard 시간 미도달
        if self._relax_pid_on_ts is None or (now_ts - self._relax_pid_on_ts) < self.relax_guard_s:
            return False

        # 창 길이 부족
        if len(self.relax_time_win) < self._relax_win_N:
            return False

        # --- A1: 고압 유지율 (기존 조건) ---
        pi_arr = np.asarray(self.relax_pi_win, dtype=np.float64)
        high_ratio = float(np.mean(pi_arr >= self.relax_pressure_thr))
        a1 = (high_ratio >= 0.95)

        # --- A2: desired force와 current force의 절대값 오차가 지속적으로 커질 때 ---
        # 센서 좌표계: 누르는 방향 = 음수 → 절댓값 사용
        t = np.asarray(self.relax_time_win, dtype=np.float64)
        f_abs = np.abs(np.asarray(self.relax_force_win, dtype=np.float64))  # 절댓값 변환
        target_force_abs = abs(self.cfg['TARGET_FORCE'])
        
        # 절대값 오차 계산: |current_force| - |target_force|
        abs_error = np.abs(f_abs - target_force_abs)
        
        t0 = t.mean()
        tt = t - t0
        denom = (tt**2).sum()
        if denom <= 1e-12:
            return False
        
        # 절대값 오차의 기울기 계산 (N/s)
        error_slope = (tt * (abs_error - abs_error.mean())).sum() / denom
        
        # Relaxation: 절대값 오차가 증가 (error_slope > 0) + 기준치 이상
        # 오차가 1초에 0.01N 이상 증가하는 경우를 relaxation으로 판단 (매우 민감하게)
        a2 = (error_slope >= 0.01)

        # --- A3: 저압 유지 + 목표값 수렴 안 됨 (새 조건) ---
        low_ratio = float(np.mean(pi_arr <= self.relax_low_pressure_thr))
        a3_low_pressure = (low_ratio >= 0.95)
        
        # 현재 힘이 목표값으로 수렴하지 않음 (절댓값 기준)
        target_force_abs = abs(self.cfg['TARGET_FORCE'])
        force_abs = np.abs(np.asarray(self.relax_force_win, dtype=np.float64))
        
        # 창 내에서 목표값과의 최소 차이
        min_force_diff = np.min(np.abs(force_abs - target_force_abs))
        a3_not_converged = (min_force_diff >= self.relax_force_target_tol)
        
        a3 = a3_low_pressure and a3_not_converged

        # A1&A2 (기존) 또는 A3 (새 조건) 중 하나라도 만족하면 RELAX
        return bool((a1 and a2) or a3)
    
    def calculate_episode_reward(self, force_data, pi_output_data, target_force=None, episode_len_s=None):
        """
        연속형 지수 스코어 기반 보상 함수 (0.000 ~ 1.000 범위)
        - 작은 성능 차이도 소수점으로 명확히 구별
        - 클립 최소화로 해상도 유지
        
        Args:
            force_data: 에피소드 동안의 힘 데이터 리스트
            pi_output_data: 에피소드 동안의 PI 출력 데이터 리스트
            target_force: 목표 힘 (None이면 설정에서 가져옴)
            episode_len_s: 에피소드 길이 (None이면 설정에서 가져옴)
        Returns:
            total_reward: 에피소드 총보상
            metrics: 성능 지표 딕셔너리
        """
        if not force_data:
            return 0.0, {}
        
        # ----- 동적 설정값 가져오기 -----
        if target_force is None:
            target_force = self.cfg['TARGET_FORCE']
        if episode_len_s is None:
            episode_len_s = self.cfg['EPISODE_SECONDS']
        
        # ----- 데이터/기본값 -----
        force_array = np.array(force_data, dtype=np.float64)
        n_samples = len(force_array)
        fs_hz = n_samples / max(episode_len_s, 1e-6)
        dt = 1.0 / max(fs_hz, 1e-6)
        errors = force_array - target_force
        T_abs = max(abs(target_force), 1.0)
        
        # ----- 핵심 지표 -----
        rmse = float(np.sqrt(np.mean(errors**2)))
        
        # 음수 타깃(-) 오버슈트: 더 음수(min)로 내려가면 overshoot
        if target_force < 0:
            extreme_force = float(np.min(force_array))
            overshoot_pct = max(0.0, (target_force - extreme_force) / T_abs)
        else:
            extreme_force = float(np.max(force_array))
            overshoot_pct = max(0.0, (extreme_force - target_force) / T_abs)
        
        # 밴드 유지율 (±1.5N)
        tol_main = Constants.BAND_TOLERANCE_N
        in_band = (np.abs(errors) <= tol_main)
        band_ratio = float(np.mean(in_band)) if n_samples > 0 else 0.0
        
        # 정착시간: warmup 이후에만 체크
        SHAPING_WARMUP_S = Constants.SHAPING_WARMUP_S
        start_idx = int(fs_hz * SHAPING_WARMUP_S)
        tol_settling = max(Constants.SETTLING_BAND_TOLERANCE, 0.01 * T_abs)
        hold_samples = int(fs_hz * Constants.SETTLING_HOLD_TIME_S)
        in_settling = (np.abs(errors) <= tol_settling)
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
        
        # ----- 연속형 지수 스코어 (클립 최소화) -----
        TAU_RMSE = Constants.TAU_RMSE
        TAU_SETTLE = Constants.TAU_SETTLE
        TAU_VAR = Constants.TAU_VAR
        TAU_U = Constants.TAU_U
        TAU_DU = Constants.TAU_DU
        
        # 0~1 사이의 "좋을수록 1" 지수 점수들
        S_acc = float(np.exp(-rmse / max(TAU_RMSE, 1e-6)))
        S_settle = float(np.exp(-settling_time_s / max(TAU_SETTLE, 1e-6)))
        S_stab = float(np.exp(-variance_n / max(TAU_VAR, 1e-6)))
        S_eff_u = float(np.exp(-u_rms_n / max(TAU_U, 1e-6)))
        S_smooth = float(np.exp(-du_rms_n / max(TAU_DU, 1e-6)))
        
        # 밴드 여유도: 밴드 안에서 얼마나 타이트하게 유지했는지 (연속형)
        margin = np.maximum(0.0, tol_main - np.abs(errors)) / max(tol_main, 1e-6)
        S_bandq = float(np.mean(margin)) if n_samples > 0 else 0.0
        S_nosat = float(1.0 - sat_ratio)
        
        # overshoot는 0~1 (연속 페널티로 따로 가중)
        P_overshoot = overshoot_pct
        
        # ----- 가중합 (연속형) -----
        W_ACC, W_BANDQ, W_FSET = 0.30, 0.30, 0.18
        W_STAB, W_EFFU, W_SMO, W_NOSAT = 0.08, 0.05, 0.04, 0.02
        W_OVERSHOOT = 0.10
        
        base = (
            W_ACC * S_acc +
            W_BANDQ * S_bandq +
            W_FSET * S_settle+
            W_STAB * S_stab +
            W_EFFU * S_eff_u +
            W_SMO * S_smooth+
            W_NOSAT * S_nosat -
            W_OVERSHOOT * P_overshoot
        )
        
        # 진행도 shaping(얇게)
        W_PROGRESS = Constants.REWARD_WEIGHT_PROGRESS
        reward_lin = base + W_PROGRESS * progress
        
        # 성공 보너스: 연속형(0~0.15)
        success_gain = 0.15 * (band_ratio**1.2) * float(np.exp(-overshoot_pct / 0.03))
        reward_lin += success_gain
        
        # 안전 위반 시 큰 페널티
        if abs(extreme_force) > Constants.SAFETY_FORCE_LIMIT:
            reward = 0.0  # 저장 안 되도록
            print(f"⚠️ [안전 위반] 극한 힘: {extreme_force:.1f}N (절댓값 > {Constants.SAFETY_FORCE_LIMIT}N)")
        else:
            # 최종 보상: 0~1에 살짝만 클립 (해상도는 유지)
            reward = float(np.clip(reward_lin, 0.0, 1.0))
        
        # ----- 메트릭 -----
        metrics = {
            # 기본 지표 (기존 호환)
            "rmse": rmse,
            "overshoot": overshoot_pct * 100.0,
            "settling_time": settling_time_s,
            "band_time": band_ratio * episode_len_s,
            "out_of_band_time": (1.0 - band_ratio) * episode_len_s,
            "band_ratio": band_ratio,
            "variance_n": variance_n,
            "u_rms_n": u_rms_n,
            "du_rms_n": du_rms_n,
            "sat_ratio": sat_ratio,
            # 연속형 스코어 구성요소
            "band_quality": S_bandq,
            "S_acc": S_acc,
            "S_settle": S_settle,
            "S_stab": S_stab,
            "S_eff_u": S_eff_u,
            "S_smooth": S_smooth,
            "S_nosat": S_nosat,
            "P_overshoot": P_overshoot,
            # Potential-based shaping
            "progress_shaping": progress,
            "phi_start": float(-abs_e[0] / T_abs) if len(abs_e) > 0 else 0.0,
            "phi_end": float(-abs_e[-1] / T_abs) if len(abs_e) > 0 else 0.0,
            "warmup_s": float(SHAPING_WARMUP_S),
            # 보너스 및 최종
            "success_gain": success_gain,
            "reward_linear": reward_lin,
            "final_reward": reward,
            # 기타
            "max_force": extreme_force,
            "sampling_freq_hz": fs_hz,
            "n_samples": n_samples,
        }
        
        return reward, metrics

    def generate_episode_reward_graph(self, save_to_rlogger_folder=True):
        if not hasattr(self, 'agent') or not self.agent.episode_rewards:
            self._log("WARNING", "생성할 보상 데이터가 없습니다")
            return
        try:
            episode_rewards = self.agent.episode_rewards
            episodes = list(range(1, len(episode_rewards) + 1))
            plt.figure(figsize=(12, 6))
            plt.plot(episodes, episode_rewards, 'b-', linewidth=2, marker='o', markersize=4)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Episode Reward', fontsize=12)
            plt.title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            if len(episode_rewards) > 1:
                avg_reward = np.mean(episode_rewards)
                plt.axhline(y=avg_reward, color='r', linestyle='--', alpha=0.7, 
                           label=f'Average: {avg_reward:.2f}')
                plt.legend()
            
            # RewardBreakdownLogger 폴더에 저장 (기본값)
            if save_to_rlogger_folder and hasattr(self, 'rlogger'):
                filename = os.path.join(self.rlogger.log_dir, "episode_rewards.png")
            else:
                # 기존 방식 (LOG_DIR에 저장)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.cfg['LOG_DIR']}/episode_rewards_{timestamp}.png"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
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
            self._log("WARNING", f"안전 위반: 힘 {current_force:.1f}N > {self.safety_force_limit}N")
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
                print(f"\r{' ' * 80}\r🎯 RL 활성화! ({wait_duration:.1f}s 대기)")
                self._log("INFO", f"RL 활성화 ({wait_duration:.1f}s)")
                break
            if state is not None:
                current_force = state[0]
                print(f"\r⏳ 대기 중 Force:{current_force:6.1f}N", end='', flush=True)
            time.sleep(1.0)
            if time.perf_counter() - wait_start > 300:
                print()
                self._log("WARNING", "RL 활성화 타임아웃 (5분)")
                return
        
        episode_stats = []
        best_reward = -float('inf')
        best_pid_gains = None
        
        # =========================
        # [A] 학습 시작 직후 1회: 모니터 시작
        # =========================
        monitor = RLRealtimeMonitor(
            title="PID Gain RL Monitor",
            hz=10,               # 실시간 갱신 10 Hz
            rolling_window=60.0  # 롤링 윈도우 60초 (연속 모니터링)
        )
        monitor.start()
        print("📊 [Monitor] 실시간 모니터 시작됨 (10 Hz, 롤링 윈도우 60초)")
        
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
                    self.cfg['TARGET_FORCE'], 
                    self.previous_pid_gains, 
                    self.historical_errors,
                    self.episode_history
                )
            else:
                # 첫 에피소드: 기준 PID gain으로 상태 추정
                print("🆕 첫 에피소드 - 기준 PID (80, 130, 0)로 상태 추정")
                base_pid = np.array([80.0, 130.0, 0.0], dtype=np.float32)
                initial_state = create_initial_state(
                    [], 
                    self.cfg['TARGET_FORCE'], 
                    base_pid, 
                    []
                )
            
            # 2. PID gain 사용 및 에피소드 시작 신호 전송
            if ep == 0:
                # 첫 에피소드: 로봇제어PC 자체 PID 사용 (P=80, I=130, D=0)
                pid_gains = np.array([80.0, 130.0, 0.0], dtype=np.float32)
                print(f"🎯 [에피소드 1] 로봇제어PC 자체 PID 사용: Kp={pid_gains[0]:.0f}, Ki={pid_gains[1]:.0f}, Kd={pid_gains[2]:.0f}")
                self._log("INFO", f"🎯 에피소드 1 기준 PID: Kp={pid_gains[0]:.0f}, Ki={pid_gains[1]:.0f}, Kd={pid_gains[2]:.0f}")
            else:
                # 2번째 에피소드부터: 이전 에피소드 종료 시 전송한 PID 사용
                assert self.pid_gains_next is not None, f"에피소드 {ep+1}: 이전 에피소드에서 next PID가 설정되지 않았습니다!"
                pid_gains = self.pid_gains_next.copy()
                print(f"🤖 [에피소드 {ep+1}] PID: Kp={pid_gains[0]:.2f}, Ki={pid_gains[1]:.2f}, Kd={pid_gains[2]:.2f}")
                
                # ⭐ 에피소드 시작 신호: episode_done=False 전송 (플래그 리셋)
                print(f"📤 에피소드 시작 신호 전송 (episode_done=False)")
                self.comm.send_pid_once(
                    pid_gains[0], pid_gains[1], pid_gains[2],
                    timing_accurate=True, episode_done=False, learning_done=False
                )
            
            # 3. PID 적용 대기
            time.sleep(0.1)
            
            # 5. 새로운 PID gain으로 제어된 실제 상태 관측
            actual_state, _ = self.comm.get_latest_state()
            if actual_state is not None:
                # 실제 관측된 상태로 업데이트
                actual_initial_state = create_initial_state([actual_state[0]], self.cfg['TARGET_FORCE'])
                self._log("INFO", f"📊 실제 상태 관측: Force={actual_state[0]:.2f}N")
            else:
                # 관측 실패 시 추정 상태 사용
                actual_initial_state = initial_state
                print("⚠️  [경고] 실제 상태 관측 실패, 추정 상태 사용")
                self._log("WARNING", "실제 상태 관측 실패, 추정 상태 사용")
            
            # 6. 에피소드 시간 동안 1kHz 데이터 수집
            self.episode_force_data = []
            self.episode_pi_output_data = []
            self.current_pid_gains = pid_gains.copy()
            
            # ⭐ sander_active 상승 에지 대기 (False -> True 전환 감지)
            # 첫 에피소드는 이미 True 상태이므로 상승 에지 대기 건너뜀
            if ep == 0:
                print(f"✅ [첫 에피소드] 이미 sander_active=True 상태 → 바로 시작")
                self._log("INFO", "✅ 첫 에피소드: 이미 활성화 상태, 바로 시작")
            else:
                # 2번째 에피소드부터: 상승 에지 대기 (모니터링 지속)
                print(f"⏳ sander_active 상승 에지 대기 중 (모니터링 지속)...")
                self._log("INFO", "⏳ sander_active 상승 에지 대기 중...")
                wait_start = time.perf_counter()
                prev_active = None
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
                                float(abs(self.cfg['TARGET_FORCE'])),
                            )
                            last_monitor_sent = now
                    
                    # 상승 에지 (False -> True) 포착
                    if (prev_active is False) and (sander_active is True):
                        wait_duration = time.perf_counter() - wait_start
                        print(f"✅ sander_active 상승 에지 감지! (False→True, {wait_duration:.2f}s 대기) → 에피소드 시작")
                        self._log("INFO", f"✅ sander_active 상승 에지 감지 ({wait_duration:.2f}s)")
                        break
                    
                    prev_active = sander_active
                    time.sleep(0.001)
                    
                    # 타임아웃 (30초)
                    if time.perf_counter() - wait_start > 30.0:
                        print("⚠️ sander_active 대기 타임아웃 (30초) - 에피소드 강제 시작")
                        self._log("WARNING", "sander_active 대기 타임아웃 - 에피소드 강제 시작")
                        break
            
            # ⭐ sander_active 상승 에지부터 동적 길이 데이터 수집
            self._log("INFO", f"📊 {self.cfg['EPISODE_SECONDS']:.0f}초 1kHz 데이터 수집 시작...")
            start_time = time.perf_counter()
            self.episode_start_time = start_time  # 에피소드 시작 시간 기록
            
            # ==== [HOOK] RELAX window reset & PID-on time ====
            self._relax_reset_windows()
            self._relax_pid_on_ts = time.perf_counter()
            
            data_count = 0
            prev_error = 0.0
            prev_pi_output = 0.0
            last_monitor_sent = 0.0  # 모니터 전송용 타이머
            
            # 주기 고정 방식으로 1kHz 정확도 향상
            dt = 0.001  # 1ms
            t_next = time.perf_counter()
            
            while (time.perf_counter() - start_time) < self.cfg["EPISODE_SECONDS"]:
                state, sander_active = self.comm.get_latest_state()
                if state is None:
                    time.sleep(0.001)
                    continue
                
                self.episode_force_data.append(state[0])  # 힘 데이터 수집
                self.episode_pi_output_data.append(state[5])  # PI 출력 데이터 수집
                data_count += 1
                
                # ==== [HOOK] Update RELAX sliding windows ====
                now_ts = time.perf_counter()
                self.relax_force_win.append(float(state[0]))
                self.relax_time_win.append(now_ts)
                self.relax_pi_win.append(float(state[5]))  # pi_output≈MPa

                # 창 기반 RELAX 판정 (A1&A2 또는 A3) → True면 에피소드 즉시 폐기
                if self._relax_update_and_check(now_ts):
                    self.relax_detected = True
                    self._log("WARNING", f"RELAX detected: high-pressure+decreasing OR low-pressure+stuck ({self.relax_window_s:.1f}s window)")
                    break
                
                # 실시간 제어 지표 데이터 수집
                current_time = time.perf_counter() - self.episode_start_time
                self.cplogger.add_data_point(
                    time=current_time,
                    force=state[0],
                    target=self.cfg['TARGET_FORCE'],
                    control_effort=np.sum(np.abs(pid_gains)),
                    pi_output=state[5],
                    pid_gains=pid_gains  # PID gain 정보 추가
                )
                
                # RewardBreakdownLogger에 스텝 로그 추가 (state가 None이 아님을 보장)
                if hasattr(self, 'rlogger'):
                    error = abs(state[0] - self.cfg['TARGET_FORCE'])
                    prog = np.exp(-error/5.0)
                    in_band_now = error <= 0.05 * self.cfg['TARGET_FORCE']
                    edot_abs = abs(prev_error - error) / 0.001 if data_count > 1 else 0.0
                    du_abs = abs(state[5] - prev_pi_output) if data_count > 1 else 0.0
                    self.rlogger.log_step(ep+1, data_count, prog, in_band_now, edot_abs, du_abs, 0.0, False)
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
                        float(abs(self.cfg['TARGET_FORCE'])),  # desired_force
                    )
                    last_monitor_sent = now
            
            print(f"📈 [수집] 완료: {data_count}개 데이터 (목표: {int(self.cfg['EPISODE_SECONDS'] * 1000)})")
            self._log("INFO", f"📈 수집된 데이터 포인트: {data_count}개 (목표: {int(self.cfg['EPISODE_SECONDS'] * 1000)})")
            
            # ==== [BRANCH] RELAX episode: skip learning and logs ====
            if self.relax_detected:
                self._log("WARNING", "🗑️ RELAX episode: skip reward/replay/logs; reset and continue.")
                print(f"🗑️ [RELAX] 에피소드 {ep+1} 폐기: 학습/기록 생략, 로봇 리셋 후 다음 에피소드")
                # 기록/학습/히스토리/지표/CSV/플롯 전부 생략

                # 내부 상태만 초기화 (로봇 리셋은 episode_done=True로 처리)
                self.episode_force_data = []
                self.episode_pi_output_data = []

                # 다음 에피소드 PID 산출(정상 히스토리 기반) 또는 유지
                if ep < episodes - 1:
                    base_pid_for_next = (self.previous_pid_gains if self.previous_pid_gains is not None
                                         else np.array([80.0, 130.0, 0.0], dtype=np.float32))
                    next_initial_state = create_initial_state(
                        [], self.cfg['TARGET_FORCE'], base_pid_for_next,
                        self.historical_errors, self.episode_history
                    )
                    next_pid_gains, _ = self.agent.select_action(next_initial_state, evaluate=False)
                    self.pid_gains_next = next_pid_gains.copy()
                    self._log("INFO", f"🎯 [RELAX→next] PID: Kp={next_pid_gains[0]:.2f}, Ki={next_pid_gains[1]:.2f}, Kd={next_pid_gains[2]:.2f}")
                    print(f"🎯 [다음 에피소드] PID: Kp={next_pid_gains[0]:.2f}, Ki={next_pid_gains[1]:.2f}, Kd={next_pid_gains[2]:.2f}")
                else:
                    self.pid_gains_next = (self.previous_pid_gains if self.previous_pid_gains is not None
                                           else np.array([80.0, 130.0, 0.0], dtype=np.float32))

                # (옵션) 로봇 PC에 다음 PID 미리 전송(episode_done=True)
                self.comm.send_pid_once(
                    self.pid_gains_next[0], self.pid_gains_next[1], self.pid_gains_next[2],
                    timing_accurate=True, episode_done=True, learning_done=False
                )
                print(f"📤 [RELAX] 다음 에피소드 PID 전송 완료")
                continue
            
            # 5. 에피소드 총보상 계산
            episode_reward, metrics = self.calculate_episode_reward(
                self.episode_force_data, 
                self.episode_pi_output_data,
                self.cfg['TARGET_FORCE'],
                self.cfg['EPISODE_SECONDS']
            )
            print(f"🏆 [결과] 보상: {episode_reward:.2f}, RMSE: {metrics['rmse']:.2f}, 오버슈트: {metrics['overshoot']:.1f}%")
            
            # =========================
            # [C] 에피소드 종료 시: reward 점 업데이트 (한 번)
            # =========================
            monitor.post_reward(ep + 1, float(episode_reward))
            
            # 6. Transition 저장 (한 스텝 MDP) - 실제 관측된 초기 상태 사용
            final_state = np.zeros(self.cfg["STATE_DIM"], dtype=np.float32)  # 최종 상태는 zero로 설정
            self.agent.store_transition(actual_initial_state, pid_gains, episode_reward, final_state, True)
            
            # 7. 통계 업데이트 (학습 전에 먼저 업데이트)
            episode_duration = time.perf_counter() - start_time
            episode_stat = {
                "episode": ep + 1,
                "duration": episode_duration,
                "pid_gains": pid_gains.copy(),
                "reward": episode_reward,
                "metrics": metrics
            }
            episode_stats.append(episode_stat)
            self.agent.episode_rewards.append(episode_reward)
            
            # RewardBreakdownLogger 플러시 (에피소드 경계에서) - 그래프 생성은 최종에만
            if hasattr(self, 'rlogger'):
                self.rlogger.flush_if_needed(ep+1, force=False, episode_rewards=self.agent.episode_rewards)
            
            # 최고 성능 PID gain 저장 (동일 점수도 저장)
            if episode_reward >= best_reward:
                best_reward = episode_reward
                best_pid_gains = pid_gains.copy()
                self.agent.save_model(f"{model_save_dir}/best_pid_agent_episode_{ep+1}_reward_{best_reward:.2f}.pth")
                print(f"💾 [저장] 최고 성능 에이전트 저장: 에피소드 {ep+1}, 보상 {best_reward:.2f}")
            
            # 8. 학습 (모든 에피소드에서 시도, replay buffer에 2개 미만이면 자동 건너뜀)
            # ⭐ 중요: 학습을 먼저 수행한 후 다음 PID를 계산해야 학습된 네트워크 사용!
            # 동적 업데이트 횟수 조정 (과적합 방지)
            buffer_size = len(self.agent.replay)
            if buffer_size >= 2:
                # 배치 크기는 자동으로 replay buffer 크기에 맞춰 조정됨
                max_updates = min(self.cfg["UPDATES_PER_EPISODE"], 
                                max(1, buffer_size // 2))  # 최소 2개씩 샘플링
                actual_updates = max(1, max_updates)
                print(f"🧠 [학습] 강화학습 업데이트 중... (에피소드 {ep+1}, {actual_updates}회, 배치크기: {min(128, buffer_size)})")
                self.agent.update_parameters_one_step(
                    None,  # None이면 동적 배치 크기 사용
                    actual_updates
                )
                print(f"✅ [학습] 신경망 업데이트 완료! 다음 에피소드는 학습된 네트워크 사용")
            else:
                print(f"📊 [에피소드 {ep+1}] 데이터 수집 완료 (학습 건너뜀: replay buffer={buffer_size}개, 최소 2개 필요)")
            
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
                'episode': ep + 1,
                'pid_gains': pid_gains.copy(),
                'reward': episode_reward,
                'metrics': metrics
            }
            self.episode_history.append(episode_record)
            
            # 최근 5개 에피소드만 유지
            if len(self.episode_history) > self.max_history:
                self.episode_history.pop(0)
            
            print(f"📊 [히스토리] 에피소드 {ep+1} 기록: 보상={episode_reward:.2f}, PID={pid_gains}")
            print(f"📈 [히스토리] 총 {len(self.episode_history)}개 에피소드 기록 유지 (최대 5개)")
            
            # 에러 통계 업데이트 (최근 10개 에피소드만 유지)
            if len(self.episode_force_data) > 0:
                episode_errors = [f - self.cfg['TARGET_FORCE'] for f in self.episode_force_data]
                avg_error = np.mean(episode_errors)
                self.historical_errors.append(avg_error)
                
                # 최근 10개 에피소드만 유지
                if len(self.historical_errors) > 10:
                    self.historical_errors.pop(0)
                
                print(f"📈 [에러] 평균 에러: {avg_error:.2f}N, 히스토리={len(self.historical_errors)}개")
            else:
                print("⚠️  [경고] 에피소드 데이터 없음, 에러 히스토리 업데이트 건너뜀")
            
            print(f"💾 [저장] 다음 에피소드용 PID: Kp={self.previous_pid_gains[0]:.2f}, Ki={self.previous_pid_gains[1]:.2f}, Kd={self.previous_pid_gains[2]:.2f}")
            
            # 11. 에피소드별 제어 지표 저장
            print(f"💾 [저장] 에피소드 지표 저장 중...")
            self.cplogger.save_episode_metrics(ep + 1)
            self.cplogger.reset_episode_data()
            
            # 12. 다음 에피소드 PID 게인 미리 계산 (마지막 에피소드가 아닌 경우)
            if ep < episodes - 1:
                # 다음 에피소드를 위한 초기 상태 생성
                next_initial_state = create_initial_state(
                    [], 
                    self.cfg['TARGET_FORCE'], 
                    pid_gains,  # 현재 에피소드의 PID (실제 사용된 것)
                    self.historical_errors,
                    self.episode_history
                )
                # 다음 에피소드 PID 게인 선택
                next_pid_gains, _ = self.agent.select_action(next_initial_state, evaluate=False)
                self.pid_gains_next = next_pid_gains.copy()  # ✅ 저장! (다음 에피소드에서 사용)
                print(f"🎯 [다음 에피소드] PID 계산 및 저장: Kp={next_pid_gains[0]:.2f}, Ki={next_pid_gains[1]:.2f}, Kd={next_pid_gains[2]:.2f}")
                self._log("INFO", f"🎯 다음 에피소드 PID: Kp={next_pid_gains[0]:.2f}, Ki={next_pid_gains[1]:.2f}, Kd={next_pid_gains[2]:.2f}")
            else:
                # 마지막 에피소드인 경우 현재 PID 사용
                next_pid_gains = pid_gains
                self.pid_gains_next = next_pid_gains.copy()
            
            # 13. 에피소드 완료 신호 전송 (다음 에피소드 PID 게인과 함께)
            print(f"📤 [전송] 에피소드 {ep+1} 완료 신호 + 다음 PID 전송 중...")
            episode_done_success = self.comm.send_pid_once(
                self.pid_gains_next[0], self.pid_gains_next[1], self.pid_gains_next[2],  # ✅ 저장된 값 전송
                timing_accurate=True, episode_done=True, learning_done=False
            )
            if episode_done_success:
                print(f"✅ [전송] 에피소드 {ep+1} 완료 신호 + 다음 PID 전송 성공")
                self._log("INFO", f"📤 에피소드 {ep+1} 완료 신호 + 다음 PID 전송 성공")
            else:
                print(f"❌ [오류] 에피소드 {ep+1} 완료 신호 + 다음 PID 전송 실패")
                self._log("ERROR", f"에피소드 {ep+1} 완료 신호 + 다음 PID 전송 실패")
            
            
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
                                float(abs(self.cfg['TARGET_FORCE'])),
                            )
                            last_monitor_sent = now
                    time.sleep(0.01)  # CPU 부하 방지
        
        # 12. 최종 결과
        self._log("INFO", "\n🎯 PID Gain 최적화 완료!")
        self._log("INFO", f"✅ {episodes}개 에피소드 완료")
        self._log("INFO", f"🏆 최고 보상: {best_reward:.2f}")
        self._log("INFO", f"🎯 최적 PID: Kp={best_pid_gains[0]:.2f}, Ki={best_pid_gains[1]:.2f}, Kd={best_pid_gains[2]:.2f}")
        
        # 13. 에피소드 요약 CSV 저장
        summary_csv = os.path.join(self.cplogger.control_perf_dir, "episode_summary.csv")
        with open(summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "Kp", "Ki", "Kd", "reward", "rmse", "overshoot", "settling_time", "band_time", "out_of_band_time"])
            for s in episode_stats:
                m = s["metrics"]
                w.writerow([s["episode"], *s["pid_gains"], s["reward"], m["rmse"], m["overshoot"], m["settling_time"], m["band_time"], m["out_of_band_time"]])
        self._log("INFO", f"📄 에피소드 요약 저장: {summary_csv}")
        
        # 14. 데이터 저장
        DataSaver.save_all_data(self, episodes, force=True)
        
        # 15. 학습 완료 신호 전송
        print(f"📤 [전송] 모든 에피소드 완료 - 학습 종료 신호 전송 중...")
        learning_done_success = self.comm.send_pid_once(0, 0, 0, True, False, True)  # learning_done=True
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

# =========================
# Signal Handler for Safe Exit
# =========================
def signal_handler(signum, frame):
    print(f"\n⚠️ Received signal {signum}. Shutting down gracefully...")
    if 'env' in globals():
        try:
            # ==== ADDED: 강제 종료 시 learning_done=True 전송 ====
            print("📡 강화학습 강제 종료 신호 전송 중...")
            try:
                success = env.comm.send_pid_once(0.0, 0.0, 0.0, True, False, True)  # learning_done=True
                if success:
                    print("✅ 강화학습 강제 종료 신호 전송 성공")
                else:
                    print("⚠️ 강화학습 강제 종료 신호 전송 실패")
            except Exception as e:
                print(f"⚠️ 강화학습 강제 종료 신호 전송 오류: {e}")
            
            print("📈 데이터 저장 중...")
            # ==== ADDED: 강제 종료 시에도 지금까지 쌓인 로우를 저장/그림 ====
            try:
                # 현재까지 완료된 에피소드 수로 flush (강제 실행)
                current_episode = len(env.agent.episode_rewards)
                env.rlogger.flush_if_needed(current_episode, force=True, episode_rewards=env.agent.episode_rewards)
            except Exception as e:
                Logger.log("ERROR", f"reward breakdown flush 실패: {e}")
            
            # ==== ADDED: 제어 성능 지표 저장 ====
            try:
                print("📊 제어 성능 지표 저장 중...")
                env.cplogger.save_performance_summary()
                env.cplogger.generate_plots()
                print("✅ 제어 성능 지표 저장 완료!")
            except Exception as e:
                print(f"⚠️ 제어 성능 지표 저장 실패: {e}")
            
            # ==== ADDED: Learning Done 폴더에 파일들 복사 ====
            # 이미 learning_done 폴더 안에 생성되므로 복사 불필요
            
            
            Logger.log("INFO", "✅ 데이터 저장 완료!")
        except Exception as e:
            Logger.log("ERROR", f"❌ 데이터 저장 실패: {e}")
    sys.exit(0)

# =========================
# STORAGE CLASSES
# =========================
class ControlPerformanceLogger:
    """
    확장된 제어공학 지표를 계산하고 저장하는 클래스
    - 기본 지표: RMSE, SSE, Rise Time, Settling Time, Overshoot, Control Effort
    - 추가 지표: IAE, ISE, ITAE, ITSE, Input RMS, Total Variation, Saturation Time, Success Rate
    - 논문용 고품질 그래프 생성 (Times New Roman 폰트)
    """
    def __init__(self, log_dir):
        self.base_log_dir = log_dir
        # 실행별 고유 폴더 생성
        # log_dir는 이미 learning_done_YYMMDD_HHhMMm 형태
        self.log_dir = log_dir
        self.control_perf_dir = os.path.join(self.log_dir, "control_performance")
        os.makedirs(self.control_perf_dir, exist_ok=True)
        
        # 폰트 설정 (논문용 Times New Roman)
        self._setup_fonts()
        
        # 기본 데이터 저장용 리스트들
        self.time_data = []
        self.force_data = []
        self.target_data = []
        self.error_data = []
        self.control_effort_data = []
        self.pi_output_data = []
        
        # 추가 지표용 데이터 저장
        self.pid_gains_history = []  # PID gain 변화 추적용
        self.input_data = []  # 제어 입력 데이터
        
        # 에피소드별 지표 저장
        self.episode_metrics = []
        
        print(f"📁 Control Performance 저장 폴더: {self.control_perf_dir}")
    
    def _setup_fonts(self):
        """폰트 설정 (기본 크기)"""
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.family'] = 'Times New Roman'
            # 기본 크기 사용 (matplotlib 기본값)
        except Exception as e:
            pass  # 폰트 설정 실패해도 그래프는 생성됨
            print("기본 폰트 사용")

    def add_data_point(self, time, force, target, control_effort, pi_output, pid_gains=None):
        """실시간 데이터 포인트 추가 (1kHz에서 호출)
        Args:
            time: 시간 (초)
            force: 현재 힘 (N)
            target: 목표 힘 (N)
            control_effort: 제어 노력 (PID gain 합) - 사용하지 않음
            pi_output: PID 출력 (실제 제어 입력)
            pid_gains: PID gain 값들 [Kp, Ki, Kd] (선택사항)
        """
        self.time_data.append(time)
        self.force_data.append(force)
        self.target_data.append(target)
        self.error_data.append(abs(force - target))
        # 실제 제어 입력으로 pi_output 사용
        self.control_effort_data.append(abs(pi_output))
        self.pi_output_data.append(pi_output)
        
        # PID gain 정보 저장 (추가 지표 계산용)
        if pid_gains is not None:
            self.pid_gains_history.append(pid_gains.copy())
            # 제어 입력으로 실제 pi_output 사용
            self.input_data.append(np.sum(np.abs(pid_gains)))
        else:
            self.pid_gains_history.append([0.0, 0.0, 0.0])
            self.input_data.append(0.0)

    def calculate_episode_metrics(self, episode_num):
        """에피소드별 확장된 제어공학 지표 계산
        Returns:
            dict: 18개 제어공학 지표 (기본 6개 + 추가 12개)
        """
        if not self.time_data:
            return None
            
        # 기본 지표들 (기존)
        basic_metrics = {
            'episode': episode_num,
            'rmse': self._calculate_rmse(),
            'steady_state_error': self._calculate_steady_state_error(),
            'rise_time': self._calculate_rise_time(),
            'settling_time': self._calculate_settling_time(),
            'overshoot': self._calculate_overshoot(),
            'control_effort': self._calculate_control_effort(),
        }
        
        # 추가 지표들 (논문용)
        additional_metrics = {
            # 추종 성능 지표
            'iae': self._calculate_iae(),           # Integral Absolute Error
            'ise': self._calculate_ise(),           # Integral Square Error
            'itae': self._calculate_itae(),         # Integral Time Absolute Error
            'itse': self._calculate_itse(),         # Integral Time Square Error
            
            # 제어 노력 지표
            'input_rms': self._calculate_input_rms(),  # Input RMS
            'total_variation': self._calculate_total_variation(),  # Total Variation
            'out_of_band_time': self._calculate_out_of_band_time(),  # Out of Band Time
            
            # 안정성 지표
            'success_rate': self._calculate_success_rate(),  # Success Rate
            'error_variance': self._calculate_error_variance(),  # Error Variance
            'peak_count': self._calculate_peak_count(),  # Peak Count
            
            # 학습 진행도 지표
            'learning_progress': self._calculate_learning_progress()
        }
        
        # 모든 지표 통합
        metrics = {**basic_metrics, **additional_metrics}
        
        self.episode_metrics.append(metrics)
        return metrics

    def _calculate_rmse(self):
        """RMSE 계산"""
        if not self.error_data:
            return None
        return np.sqrt(np.mean(np.square(self.error_data)))

    def _calculate_steady_state_error(self):
        """Steady State Error 계산 (마지막 10% 구간의 평균 절대 오차)"""
        if not self.error_data:
            return None
        last_10_percent = max(1, int(len(self.error_data) * 0.1))
        return np.mean(self.error_data[-last_10_percent:])

    def _calculate_rise_time(self):
        """Stabilization Time 계산 (목표값 ±5% 밴드 내 최초 진입 시간)"""
        if not self.force_data or not self.target_data or len(self.force_data) < 10:
            return None
            
        target = self.target_data[0]  # -45N
        force_array = np.array(self.force_data)
        time_array = np.array(self.time_data)
        
        # ±5% 밴드 정의
        band = abs(target) * 0.05  # ±2.25N
        target_min = target - band
        target_max = target + band
        
        # 밴드 내 진입 지점 찾기
        in_band = (force_array >= target_min) & (force_array <= target_max)
        band_indices = np.where(in_band)[0]
        
        if len(band_indices) > 0:
            # 첫 번째로 밴드에 진입한 시간
            return float(time_array[band_indices[0]] - time_array[0])
        
        return None

    def _calculate_settling_time(self):
        """Settling Time 계산 (연속 유지 기준) - 에피소드 보상과 동일한 기준"""
        if not self.force_data or not self.target_data:
            return None
            
        target = self.target_data[0]
        band = 0.05 * target  # ±5% 밴드
        force_array = np.array(self.force_data)
        time_array = np.array(self.time_data)
        
        # 연속 유지 구간 찾기 (2초 연속 유지)
        within = np.abs(force_array - target) <= band
        hold_duration = int(2.0 * 1000)  # 2초 연속 유지 (1kHz 기준)
        
        run_length = 0
        settling_time = None
        for k, in_band in enumerate(within):
            if in_band:
                run_length += 1
                if run_length >= hold_duration:
                    settling_time = max(0.0, (k - hold_duration) / 1000.0)
                    break
            else:
                run_length = 0
        
        return float(settling_time) if settling_time is not None else None

    def _calculate_overshoot(self):
        """Overshoot 계산 (목표값을 넘어선 최대 편차)"""
        if not self.force_data or not self.target_data:
            return None
            
        target = self.target_data[0]  # -45N
        force_array = np.array(self.force_data)
        
        # 목표값보다 더 나쁜 방향으로의 최대 편차 계산
        if target < 0:  # 음수 목표값 (압축력)
            # 더 큰 음수값 (더 큰 압축력)을 찾음
            max_overshoot = np.min(force_array)  # 가장 작은 값 (가장 큰 음수)
            if max_overshoot < target:
                return float(((target - max_overshoot) / abs(target)) * 100)
        else:  # 양수 목표값
            max_overshoot = np.max(force_array)
            if max_overshoot > target:
                return float(((max_overshoot - target) / target) * 100)
        
        return 0.0

    def _calculate_control_effort(self):
        """Control Effort 계산 (실제 제어 입력의 RMS)"""
        if not self.pi_output_data:
            return None
        return float(np.sqrt(np.mean(np.array(self.pi_output_data)**2)))

    def _calculate_learning_progress(self):
        """Learning Progress 계산 (에피소드별 성공률 개선)"""
        if len(self.episode_metrics) < 3:
            return None
        
        success_rates = [ep['success_rate'] for ep in self.episode_metrics if ep['success_rate'] is not None]
        if len(success_rates) < 3:
            return None
            
        # 최근 3개 에피소드의 성공률 추세 (양수일수록 개선)
        recent_success = np.mean(success_rates[-3:])
        earlier_success = np.mean(success_rates[:-3]) if len(success_rates) > 3 else success_rates[0]
        
        return float(recent_success - earlier_success)
    
    # =========================
    # 추가 제어공학 지표 계산 메서드들
    # =========================
    
    def _calculate_iae(self):
        """IAE (Integral Absolute Error) 계산 - 연마 공정에서 편차 누적"""
        if not self.error_data or not self.time_data:
            return None
        dt = np.mean(np.diff(self.time_data)) if len(self.time_data) > 1 else 0.001
        return float(np.sum(np.abs(self.error_data)) * dt)
    
    def _calculate_ise(self):
        """ISE (Integral Square Error) 계산 - 큰 오차에 민감"""
        if not self.error_data or not self.time_data:
            return None
        dt = np.mean(np.diff(self.time_data)) if len(self.time_data) > 1 else 0.001
        return float(np.sum(np.square(self.error_data)) * dt)
    
    def _calculate_itae(self):
        """ITAE (Integral Time Absolute Error) 계산 - 후반 안정성 강조"""
        if not self.error_data or not self.time_data:
            return None
        dt = np.mean(np.diff(self.time_data)) if len(self.time_data) > 1 else 0.001
        time_array = np.array(self.time_data)
        return float(np.sum(np.abs(self.error_data) * time_array) * dt)
    
    def _calculate_itse(self):
        """ITSE (Integral Time Square Error) 계산 - 시간 가중 제곱 오차"""
        if not self.error_data or not self.time_data:
            return None
        dt = np.mean(np.diff(self.time_data)) if len(self.time_data) > 1 else 0.001
        time_array = np.array(self.time_data)
        return float(np.sum(np.square(self.error_data) * time_array) * dt)
    
    def _calculate_input_rms(self):
        """Input RMS 계산 - PID gain 합의 RMS 값 (제어 노력 분리)"""
        if not self.input_data:
            return None
        arr = np.asarray(self.input_data, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(arr))))
    
    def _calculate_total_variation(self):
        """Total Variation 계산 - 실제 제어 출력 변화 총량 (밸브 마모와 직결)"""
        if len(self.pi_output_data) < 2:
            return None
        return float(np.sum(np.abs(np.diff(self.pi_output_data))))
    
    def _calculate_out_of_band_time(self):
        """Out of Band Time 계산 - 목표 범위 밖 체류 시간"""
        if not self.force_data or not self.target_data or not self.time_data:
            return None
        target = self.target_data[0]
        tolerance = 1.0  # ±1N 오차 범위
        force_array = np.array(self.force_data)
        time_array = np.array(self.time_data)
        
        out_of_range = np.abs(force_array - target) > tolerance
        if len(out_of_range) > 0:
            dt = np.mean(np.diff(time_array)) if len(time_array) > 1 else 0.001
            return float(np.sum(out_of_range) * dt)
        return 0.0
    
    def _calculate_success_rate(self):
        """Success Rate 계산 - 목표 범위 내 유지 비율"""
        if not self.force_data or not self.target_data or not self.time_data:
            return None
        target = self.target_data[0]  # -45N
        tolerance = abs(target) * 0.02  # ±2% 오차 범위 (±0.9N) - 더 엄격한 기준
        in_band = np.abs(np.array(self.force_data) - target) <= tolerance
        return float(np.sum(in_band) / len(in_band))
    
    def _calculate_error_variance(self):
        """Error Variance 계산 - 오차 분산 (안정성 지표)"""
        if not self.error_data:
            return None
        return float(np.var(self.error_data))
    
    def _calculate_peak_count(self):
        """Peak Count 계산 - 피크 수 (링잉 정도)"""
        if len(self.force_data) < 3:
            return None
        
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(self.force_data, height=np.mean(self.force_data))
            return len(peaks)
        except ImportError:
            # scipy가 없는 경우 간단한 피크 검출
            force_array = np.array(self.force_data)
            mean_force = np.mean(force_array)
            peaks = 0
            
            # 간단한 피크 검출 알고리즘
            for i in range(1, len(force_array) - 1):
                if (force_array[i] > force_array[i-1] and 
                    force_array[i] > force_array[i+1] and 
                    force_array[i] > mean_force):
                    peaks += 1
            return peaks

    def save_episode_metrics(self, episode_num):
        """에피소드별 지표를 CSV로 저장"""
        metrics = self.calculate_episode_metrics(episode_num)
        if metrics is None:
            return
            
        # 개별 지표별 CSV 저장
        for metric_name, value in metrics.items():
            if metric_name == 'episode' or value is None:
                continue
                
            csv_path = os.path.join(self.control_perf_dir, f"{metric_name}.csv")
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["episode", metric_name])
                writer.writerow([episode_num, value])

    def save_performance_summary(self):
        """전체 성능 요약 저장 (18개 제어공학 지표)"""
        if not self.episode_metrics:
            return
            
        summary_path = os.path.join(self.control_perf_dir, "performance_summary.csv")
        
        with open(summary_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Mean", "Std", "Min", "Max", "Unit", "Description"])
            
            # 모든 지표별 통계 계산 (기본 + 추가)
            all_metrics = [
                # 기본 지표들
                'rmse', 'steady_state_error', 'rise_time', 'settling_time', 'overshoot', 'control_effort',
                # 추가 추종 성능 지표
                'iae', 'ise', 'itae', 'itse',
                # 제어 노력 지표
                'input_rms', 'total_variation', 'out_of_band_time',
                # 안정성 지표
                'success_rate', 'error_variance', 'peak_count',
                # 학습 진행도 지표
                'learning_progress'
            ]
            
            for metric_name in all_metrics:
                values = [ep[metric_name] for ep in self.episode_metrics if ep[metric_name] is not None]
                
                if values:
                    writer.writerow([
                        metric_name,
                        f"{np.mean(values):.4f}",
                        f"{np.std(values):.4f}",
                        f"{np.min(values):.4f}",
                        f"{np.max(values):.4f}",
                        self._get_metric_unit(metric_name),
                        self._get_metric_description(metric_name)
                    ])
        
        print(f"📊 성능 요약 저장 완료: {summary_path}")

    def _get_metric_unit(self, metric_name):
        """지표별 단위 반환"""
        units = {
            # 기본 지표들
            'rmse': 'N',
            'steady_state_error': 'N', 
            'rise_time': 's',
            'settling_time': 's',
            'overshoot': '%',
            'control_effort': 'arb.',
            'learning_progress': 'N/episode',
            # 추가 추종 성능 지표
            'iae': 'N·s',
            'ise': 'N²·s',
            'itae': 'N·s²',
            'itse': 'N²·s²',
            # 제어 노력 지표
            'input_rms': 'N',
            'total_variation': 'N',
            'out_of_band_time': 's',
            # 안정성 지표
            'success_rate': '-',
            'error_variance': 'N²',
            'peak_count': 'count'
        }
        return units.get(metric_name, '')
    
    def _get_metric_description(self, metric_name):
        """지표별 설명 반환"""
        descriptions = {
            # 기본 지표들
            'rmse': 'Root Mean Square Error - 제어 정확도',
            'steady_state_error': 'Steady State Error - 정상상태 오차',
            'rise_time': 'Rise Time - 상승시간 (10%→90%)',
            'settling_time': 'Settling Time - 정착시간 (±5%)',
            'overshoot': 'Overshoot - 오버슈트 (%)',
            'control_effort': 'Control Effort - 제어 노력',
            'learning_progress': 'Learning Progress - 학습 진행도',
            # 추가 추종 성능 지표
            'iae': 'Integral Absolute Error - 절대 오차 적분',
            'ise': 'Integral Square Error - 제곱 오차 적분',
            'itae': 'Integral Time Absolute Error - 시간 가중 절대 오차 적분',
            'itse': 'Integral Time Square Error - 시간 가중 제곱 오차 적분',
            # 제어 노력 지표
            'input_rms': 'Input RMS - 제어 입력 RMS',
            'total_variation': 'Total Variation - 총 변화량 (밸브 마모)',
            'out_of_band_time': 'Out of Band Time - 밴드 이탈 시간',
            # 안정성 지표
            'success_rate': 'Success Rate - 성공률 (목표 범위 유지)',
            'error_variance': 'Error Variance - 오차 분산 (안정성)',
            'peak_count': 'Peak Count - 피크 수 (링잉 정도)'
        }
        return descriptions.get(metric_name, '')

    def generate_plots(self):
        """각 지표별 시각화 생성 (18개 제어공학 지표)"""
        if not self.episode_metrics:
            return
            
        print("📈 논문용 고품질 그래프 생성 중...")
        
        # 모든 지표들 (기본 + 추가)
        all_metrics = [
            # 기본 지표들
            'rmse', 'steady_state_error', 'rise_time', 'settling_time', 'overshoot', 'control_effort',
            # 추가 추종 성능 지표
            'iae', 'ise', 'itae', 'itse',
            # 제어 노력 지표
            'input_rms', 'total_variation', 'out_of_band_time',
            # 안정성 지표
            'success_rate', 'error_variance', 'peak_count',
            # 학습 진행도 지표
            'learning_progress'
        ]
        
        for metric_name in all_metrics:
            self._plot_metric(metric_name)
        
        # 추가로 종합 대시보드 생성
        self._generate_comprehensive_dashboard()
        
        # Step 축 지표들도 생성 (논문용)
        self._generate_step_based_plots()
        
        print(f"✅ 총 {len(all_metrics)}개 지표 그래프 생성 완료")

    def _plot_metric(self, metric_name):
        """개별 지표 시각화 (논문용 고품질)"""
        values = [ep[metric_name] for ep in self.episode_metrics if ep[metric_name] is not None]
        episodes = [ep['episode'] for ep in self.episode_metrics if ep[metric_name] is not None]
        
        if not values:
            return
            
        # 폰트 설정 재적용 (각 그래프마다)
        self._setup_fonts()
            
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, values, 'b-', linewidth=3, marker='o', markersize=6, 
                markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=1)
        plt.xlabel('Episode Number', fontweight='bold')
        plt.ylabel(f'{metric_name.upper()}', fontweight='bold')
        plt.title(f'{metric_name.upper()} Over Episodes', fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 평균선 추가
        if len(values) > 1:
            avg_value = np.mean(values)
            std_value = np.std(values)
            plt.axhline(y=avg_value, color='r', linestyle='--', alpha=0.8, linewidth=2,
                       label=f'Mean: {avg_value:.4f}±{std_value:.4f}')
            plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # 축 범위 조정
        plt.xlim(min(episodes) - 0.5, max(episodes) + 0.5)
        
        png_path = os.path.join(self.control_perf_dir, f"{metric_name}.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  📊 {metric_name.upper()} 그래프 저장: {png_path}")
    
    def _generate_comprehensive_dashboard(self):
        """종합 대시보드 생성 (논문용)"""
        if len(self.episode_metrics) < 2:
            return
        
        # 폰트 설정 재적용
        self._setup_fonts()
        
        # 4x4 서브플롯 생성
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('PID Gain Optimization Performance Dashboard', fontweight='bold')
        
        # 주요 지표들 선택
        key_metrics = [
            'rmse', 'iae', 'ise', 'itae',
            'rise_time', 'settling_time', 'overshoot', 'success_rate',
            'input_rms', 'total_variation', 'out_of_band_time', 'error_variance',
            'control_effort', 'peak_count', 'learning_progress'
        ]
        
        for i, metric_name in enumerate(key_metrics):
            row, col = i // 4, i % 4
            ax = axes[row, col]
            
            values = [ep[metric_name] for ep in self.episode_metrics if ep[metric_name] is not None]
            episodes = [ep['episode'] for ep in self.episode_metrics if ep[metric_name] is not None]
            
            if values:
                ax.plot(episodes, values, 'b-', linewidth=2, marker='o', markersize=4)
                ax.set_title(f'{metric_name.upper()}', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # 평균선 추가
                if len(values) > 1:
                    avg_value = np.mean(values)
                    ax.axhline(y=avg_value, color='r', linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric_name.upper()}', fontweight='bold')
        
        # 빈 서브플롯 숨기기
        for i in range(len(key_metrics), 16):
            row, col = i // 4, i % 4
            axes[row, col].set_visible(False)
        
        dashboard_path = os.path.join(self.control_perf_dir, "comprehensive_dashboard.png")
        plt.tight_layout()
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 종합 대시보드 저장: {dashboard_path}")
    
    def _generate_step_based_plots(self):
        """Step 축 지표 그래프 생성 (논문용 - 에피소드 내부 시간적 추세)"""
        if not self.time_data or not self.force_data:
            return
        
        print("📈 Step 축 지표 그래프 생성 중...")
        
        # 폰트 설정 재적용
        self._setup_fonts()
        
        # 1. Force Tracking Curve (목표힘 vs 실제힘)
        self._plot_force_tracking_curve()
        
        # 2. Error Time Series (순간 오차)
        self._plot_error_time_series()
        
        # 3. Control Input Time Series (제어 입력)
        self._plot_control_input_series()
        
        # 4. Reward Breakdown (보상 구성 요소)
        self._plot_reward_breakdown()
        
        # 5. Step 축 종합 대시보드
        self._generate_step_dashboard()
        
        print("✅ Step 축 지표 그래프 생성 완료")
    
    def _plot_force_tracking_curve(self):
        """Force Tracking Curve (목표힘 vs 실제힘)"""
        if not self.force_data or not self.target_data or not self.time_data:
            return
        
        plt.figure(figsize=(14, 8))
        time_array = np.array(self.time_data)
        force_array = np.array(self.force_data)
        target_array = np.array(self.target_data)
        
        plt.plot(time_array, target_array, 'r--', linewidth=3, label='Target Force', alpha=0.8)
        plt.plot(time_array, force_array, 'b-', linewidth=2, label='Actual Force', alpha=0.9)
        tolerance = 0.05 * target_array[0] if len(target_array) > 0 else 2.25
        plt.fill_between(time_array, target_array - tolerance, target_array + tolerance, 
                        alpha=0.2, color='green', label='±5% Tolerance Band')
        
        plt.xlabel('Time (s)', fontweight='bold')
        plt.ylabel('Force (N)', fontweight='bold')
        plt.title('Force Tracking Performance (Step-based)', fontweight='bold')
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        png_path = os.path.join(self.control_perf_dir, "force_tracking_curve.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  📊 Force Tracking Curve 저장: {png_path}")
    
    def _plot_error_time_series(self):
        """Error Time Series (순간 오차)"""
        if not self.error_data or not self.time_data:
            return
        
        plt.figure(figsize=(14, 8))
        time_array = np.array(self.time_data)
        error_array = np.array(self.error_data)
        target_array = np.array(self.target_data)
        
        plt.plot(time_array, error_array, 'r-', linewidth=2, label='Absolute Error')
        tolerance = 0.05 * target_array[0] if len(target_array) > 0 else 2.25
        plt.axhline(y=tolerance, color='g', linestyle='--', alpha=0.7, label='±5% Tolerance')
        plt.axhline(y=-tolerance, color='g', linestyle='--', alpha=0.7)
        plt.fill_between(time_array, -tolerance, tolerance, alpha=0.1, color='green', label='Tolerance Band')
        
        plt.xlabel('Time (s)', fontweight='bold')
        plt.ylabel('Force Error (N)', fontweight='bold')
        plt.title('Force Error Time Series (Step-based)', fontweight='bold')
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        png_path = os.path.join(self.control_perf_dir, "error_time_series.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  📊 Error Time Series 저장: {png_path}")
    
    def _plot_control_input_series(self):
        """Control Input Time Series (제어 입력)"""
        if not self.input_data or not self.time_data:
            return
        
        plt.figure(figsize=(14, 8))
        time_array = np.array(self.time_data)
        input_array = np.array(self.input_data)
        
        plt.plot(time_array, input_array, 'purple', linewidth=2, label='Control Input (PID Gain Sum)')
        plt.xlabel('Time (s)', fontweight='bold')
        plt.ylabel('Control Input', fontweight='bold')
        plt.title('Control Input Time Series (Step-based)', fontweight='bold')
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        png_path = os.path.join(self.control_perf_dir, "control_input_series.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  📊 Control Input Series 저장: {png_path}")
    
    def _plot_reward_breakdown(self):
        """Reward Breakdown (보상 구성 요소) - Step 단위"""
        if not self.time_data or not self.force_data or not self.target_data:
            return
        
        # Step 단위 보상 구성 요소 계산
        time_array = np.array(self.time_data)
        force_array = np.array(self.force_data)
        target_array = np.array(self.target_data)
        error_array = np.abs(force_array - target_array)
        
        # 1. Progress Reward (목표에 가까워질수록 높은 보상)
        progress_reward = np.exp(-error_array / 5.0)  # 오차가 작을수록 높은 보상
        
        # 2. In-band Reward (±5% 범위 내에 있을 때 보상)
        tolerance = target_array[0] * 0.05
        in_band = np.abs(force_array - target_array) <= tolerance
        in_band_reward = in_band.astype(float)
        
        # 3. Error Penalty (오차에 대한 페널티)
        error_penalty = -error_array / 10.0
        
        # 4. Stability Reward (안정성 보상)
        if len(error_array) > 1:
            error_derivative = np.abs(np.diff(error_array, prepend=error_array[0]))
            stability_reward = np.exp(-error_derivative / 2.0)
        else:
            stability_reward = np.ones_like(error_array)
        
        plt.figure(figsize=(16, 10))
        
        # 서브플롯 1: Progress Reward
        plt.subplot(2, 2, 1)
        plt.plot(time_array, progress_reward, 'b-', linewidth=2)
        plt.title('Progress Reward (Step-based)', fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Progress Reward')
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 2: In-band Reward
        plt.subplot(2, 2, 2)
        plt.plot(time_array, in_band_reward, 'g-', linewidth=2)
        plt.title('In-band Reward (Step-based)', fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('In-band Reward')
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 3: Error Penalty
        plt.subplot(2, 2, 3)
        plt.plot(time_array, error_penalty, 'r-', linewidth=2)
        plt.title('Error Penalty (Step-based)', fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Error Penalty')
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 4: Stability Reward
        plt.subplot(2, 2, 4)
        plt.plot(time_array, stability_reward, 'purple', linewidth=2)
        plt.title('Stability Reward (Step-based)', fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Stability Reward')
        plt.grid(True, alpha=0.3)
        
        png_path = os.path.join(self.control_perf_dir, "reward_breakdown_step.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  📊 Reward Breakdown 저장: {png_path}")
    
    def _generate_step_dashboard(self):
        """Step 축 종합 대시보드"""
        if not self.time_data or not self.force_data:
            return
        
        # 폰트 설정 재적용
        self._setup_fonts()
        
        # 2x2 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Step-based Performance Dashboard', fontweight='bold')
        
        time_array = np.array(self.time_data)
        force_array = np.array(self.force_data)
        target_array = np.array(self.target_data)
        error_array = np.array(self.error_data)
        
        # 1. Force Tracking
        axes[0, 0].plot(time_array, target_array, 'r--', linewidth=2, label='Target')
        axes[0, 0].plot(time_array, force_array, 'b-', linewidth=1.5, label='Actual')
        axes[0, 0].set_title('Force Tracking', fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Force (N)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error Time Series
        axes[0, 1].plot(time_array, error_array, 'r-', linewidth=1.5)
        target_array = np.array(self.target_data)
        tolerance = 0.05 * target_array[0] if len(target_array) > 0 else 2.25
        axes[0, 1].axhline(y=tolerance, color='g', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=-tolerance, color='g', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Error Time Series', fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Error (N)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Control Input
        if self.input_data:
            input_array = np.array(self.input_data)
            axes[1, 0].plot(time_array, input_array, 'purple', linewidth=1.5)
            axes[1, 0].set_title('Control Input', fontweight='bold')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Input')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error Distribution
        axes[1, 1].hist(error_array, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        target_array = np.array(self.target_data)
        tolerance = 0.05 * target_array[0] if len(target_array) > 0 else 2.25
        axes[1, 1].axvline(x=tolerance, color='r', linestyle='--', alpha=0.7, label='±5% Tolerance')
        axes[1, 1].axvline(x=-tolerance, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Error Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Error (N)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        dashboard_path = os.path.join(self.control_perf_dir, "step_dashboard.png")
        plt.tight_layout()
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 Step 축 대시보드 저장: {dashboard_path}")

    def reset_episode_data(self):
        """에피소드 데이터 초기화 (모든 데이터 변수 포함)"""
        # 기본 데이터
        self.time_data.clear()
        self.force_data.clear()
        self.target_data.clear()
        self.error_data.clear()
        self.control_effort_data.clear()
        self.pi_output_data.clear()
        
        # 추가 지표용 데이터
        self.pid_gains_history.clear()
        self.input_data.clear()

# =========================
# Learning Done Logger
# =========================
class LearningDoneLogger:
    """
    학습 완료 시 전체 로깅을 관리하는 클래스
    - 에피소드 보상 데이터 복사
    - 보상 분석 데이터 복사
    - 학습 결과 통합 저장
    """
    def __init__(self, log_dir):
        # 타임스탬프 기반 learning_done 폴더 한 번만 생성
        now = datetime.now()
        timestamp = now.strftime("%y%m%d_%Hh%Mm")
        self.log_dir = os.path.join(log_dir, f"learning_done_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"📁 Learning Done 폴더: {self.log_dir}")

    # 복사 함수 제거 - 이미 learning_done 폴더 안에 생성되므로 복사 불필요

# =========================
# Reward Breakdown Logger
# =========================
class RewardBreakdownLogger:
    """
    스텝 단위 보상 분석 로깅
    - 실시간 보상 구성 요소 수집
    - 에피소드별 보상 통계 생성
    - CSV 저장 및 PNG 시각화
    - Ctrl+C/학습 완료 시 자동 저장
    """
    def __init__(self, log_dir):
        # learning_done 폴더 내부의 reward_breakdown 서브폴더 사용 (중복 방지)
        self.log_dir = os.path.join(log_dir, "reward_breakdown")
        os.makedirs(self.log_dir, exist_ok=True)
        self.rows = []  # 버퍼
        self.csv_path = os.path.join(self.log_dir, "reward_breakdown.csv")
        self.episode_rewards_path = os.path.join(self.log_dir, "episode_rewards.csv")
        print(f"📁 Reward breakdown 저장 폴더: {self.log_dir}")

    def log_step(self, episode, step, prog, in_band_now, edot_abs, du_abs, reward, is_her):
        self.rows.append({
            "episode": episode,
            "step": step,
            "prog": float(prog),
            "in_band_now": int(in_band_now),
            "edot_abs": float(edot_abs),
            "du_abs": float(du_abs),
            "reward": float(reward),
            "is_her": int(is_her)
        })

    def save_episode_rewards(self, episode_rewards):
        """에피소드별 보상을 CSV로 저장"""
        with open(self.episode_rewards_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])  # 헤더
            for i, reward in enumerate(episode_rewards, 1):
                writer.writerow([i, float(reward)])

    def generate_episode_reward_graph(self, episode_rewards):
        """에피소드별 보상 그래프를 PNG로 저장"""
        if not episode_rewards:
            return
        
        try:
            episodes = list(range(1, len(episode_rewards) + 1))
            plt.figure(figsize=(12, 6))
            plt.plot(episodes, episode_rewards, 'b-', linewidth=2, marker='o', markersize=4)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Episode Reward', fontsize=12)
            plt.title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            if len(episode_rewards) > 1:
                avg_reward = np.mean(episode_rewards)
                plt.axhline(y=avg_reward, color='r', linestyle='--', alpha=0.7, 
                           label=f'Average: {avg_reward:.2f}')
                plt.legend()
            
            filename = os.path.join(self.log_dir, "episode_rewards.png")
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   📈 PNG: episode_rewards.png")
        except Exception as e:
            print(f"   ⚠️ 에피소드 리워드 그래프 생성 실패: {e}")

    def _write_csv_append(self):
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["episode","step","prog","in_band_now","edot_abs","du_abs","reward","is_her"]
            )
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.rows)

    def save_reward_breakdown_csv(self):
        """reward_breakdown 데이터를 CSV로 저장"""
        if not self.rows:
            return
            
        csv_path = os.path.join(self.log_dir, "reward_breakdown.csv")
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["episode","step","prog","in_band_now","edot_abs","du_abs","reward","is_her"]
            )
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"   📊 CSV: reward_breakdown.csv")

    def _plot_png(self, start_ep, end_ep):
        # start_ep~end_ep 사이의 데이터만 사용
        data = [r for r in self.rows if start_ep <= r["episode"] <= end_ep]
        if not data:
            return
        # 에피소드별 평균
        ep_keys = sorted(set(r["episode"] for r in data))
        avg_prog = []
        avg_in_band = []
        avg_edot = []
        avg_du = []
        avg_R = []
        for ep in ep_keys:
            items = [r for r in data if r["episode"] == ep]
            avg_prog.append(np.mean([r["prog"] for r in items]))
            avg_in_band.append(np.mean([r["in_band_now"] for r in items]))
            avg_edot.append(np.mean([r["edot_abs"] for r in items]))
            avg_du.append(np.mean([r["du_abs"] for r in items]))
            avg_R.append(np.mean([r["reward"] for r in items]))

        # 1) prog
        plt.figure(figsize=(11,4))
        plt.plot(ep_keys, avg_prog, linewidth=2, marker="o")
        plt.title(f"Average prog per episode ({start_ep}-{end_ep})")
        plt.xlabel("Episode"); plt.ylabel("prog")
        plt.grid(True, alpha=0.3)
        out1 = os.path.join(self.log_dir, f"reward_breakdown_prog_ep{start_ep}-{end_ep}.png")
        plt.tight_layout(); plt.savefig(out1, dpi=200); plt.close()

        # 2) in_band_now
        plt.figure(figsize=(11,4))
        plt.plot(ep_keys, avg_in_band, linewidth=2, marker="o")
        plt.title(f"Average in_band_now per episode ({start_ep}-{end_ep})")
        plt.xlabel("Episode"); plt.ylabel("in_band_now (ratio)")
        plt.grid(True, alpha=0.3)
        out2 = os.path.join(self.log_dir, f"reward_breakdown_inband_ep{start_ep}-{end_ep}.png")
        plt.tight_layout(); plt.savefig(out2, dpi=200); plt.close()

        # 3) edot_abs
        plt.figure(figsize=(11,4))
        plt.plot(ep_keys, avg_edot, linewidth=2, marker="o")
        plt.title(f"Average |de/dt| per episode ({start_ep}-{end_ep})")
        plt.xlabel("Episode"); plt.ylabel("|de/dt|")
        plt.grid(True, alpha=0.3)
        out3 = os.path.join(self.log_dir, f"reward_breakdown_edot_ep{start_ep}-{end_ep}.png")
        plt.tight_layout(); plt.savefig(out3, dpi=200); plt.close()

        # 4) du_abs
        plt.figure(figsize=(11,4))
        plt.plot(ep_keys, avg_du, linewidth=2, marker="o")
        plt.title(f"Average |Δu| per episode ({start_ep}-{end_ep})")
        plt.xlabel("Episode"); plt.ylabel("|Δu|")
        plt.grid(True, alpha=0.3)
        out4 = os.path.join(self.log_dir, f"reward_breakdown_du_ep{start_ep}-{end_ep}.png")
        plt.tight_layout(); plt.savefig(out4, dpi=200); plt.close()

        # 5) reward
        plt.figure(figsize=(11,4))
        plt.plot(ep_keys, avg_R, linewidth=2, marker="o")
        plt.title(f"Average reward per episode ({start_ep}-{end_ep})")
        plt.xlabel("Episode"); plt.ylabel("reward")
        plt.grid(True, alpha=0.3)
        out5 = os.path.join(self.log_dir, f"reward_breakdown_reward_ep{start_ep}-{end_ep}.png")
        plt.tight_layout(); plt.savefig(out5, dpi=200); plt.close()

    def flush_if_needed(self, current_episode, force=False, episode_rewards=None):
        """
        CSV 저장 + PNG 시각화를 수행.
        force=True이면 언제든 실행, False이면 CSV만 저장 (PNG 생성 안 함).
        episode_rewards: 에피소드별 보상 리스트 (선택사항)
        """
        # 데이터가 없으면 실행하지 않음
        if not self.rows:
            return
            
        # CSV는 항상 저장 (메모리 절약)
        self.save_reward_breakdown_csv()
        
        # force=True일 때만 PNG 생성 (최종에만)
        if force:
            # 에피소드별 보상 저장 및 그래프 생성
            if episode_rewards is not None:
                self.save_episode_rewards(episode_rewards)
                self.generate_episode_reward_graph(episode_rewards)
            
            # PNG 생성 (전체 데이터)
            start_ep = min(row["episode"] for row in self.rows)
            end_ep = max(row["episode"] for row in self.rows)
            self._plot_png(start_ep, end_ep)
            
            print(f"✅ Reward breakdown 저장: {self.log_dir}")
        
        # 메모리 절약을 위해 rows 유지 (전체 그래프용)

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # ========== 설정 변경 포인트 ==========
    RECV_FREQUENCY_HZ = 1000
    EPISODE_LENGTH_SECONDS = 40.0  # 에피소드 길이
    # ====================================
    
    config = create_config(RECV_FREQUENCY_HZ, EPISODE_LENGTH_SECONDS)
    print("🚀 PID GAIN OPTIMIZATION VERSION: JY_PID_Gain_SAC_1_test.py")
    print(f"📡 수신 주파수: {RECV_FREQUENCY_HZ}Hz (간격: {config['RECV_INTERVAL_SEC']:.3f}초)")
    print(f"🎯 목표 힘: {config['TARGET_FORCE']}N 고정")
    print(f"⏱️ 에피소드 길이: {config['EPISODE_SECONDS']}초")
    print("=" * 60)
    
    # 재현성을 위한 시드 설정
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("🎲 재현성 시드 설정 완료 (42)")
    env = PIDGainOptimizationEnvironment(config)
    try:
        print(f"🚀 Starting PID Gain optimization training...")
        env.run_pid_optimization_training(config["EPISODES"])
        print("✅ Training completed successfully!")
        try:
            print("📈 데이터 저장 중...")
            # ==== ADDED: 정상 종료 시에도 강제 플러시 ====
            try:
                env.rlogger.flush_if_needed(config["EPISODES"], force=True, episode_rewards=env.agent.episode_rewards)
            except Exception as e:
                Logger.log("ERROR", f"reward breakdown flush 실패: {e}")
            
            # ==== ADDED: 모든 데이터 저장 ====
            DataSaver.save_all_data(env)
        except Exception as e:
            Logger.log("ERROR", f"❌ 데이터 저장 실패: {e}")
    except KeyboardInterrupt:
        Logger.log("WARNING", "Interrupted by user (Ctrl+C).")
        print("\n⚠️ 사용자가 Ctrl+C로 중단했습니다. 안전하게 종료 중...")
        # ==== ADDED: KeyboardInterrupt 시 learning_done=True 전송 ====
        try:
            print("📡 강화학습 중단 신호 전송 중...")
            success = env.comm.send_pid_once(0.0, 0.0, 0.0, True, False, True)  # learning_done=True
            if success:
                print("✅ 강화학습 중단 신호 전송 성공")
                Logger.log("INFO", "📤 학습 중단 신호 전송 성공")
            else:
                print("⚠️ 강화학습 중단 신호 전송 실패")
                Logger.log("ERROR", "학습 중단 신호 전송 실패")
        except Exception as e:
            print(f"⚠️ 강화학습 중단 신호 전송 오류: {e}")
            Logger.log("ERROR", f"학습 중단 신호 전송 오류: {e}")
        
        # ==== ADDED: 모든 데이터 저장 ====
        DataSaver.save_all_data(env)
        
        env.comm.close()
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        Logger.log("ERROR", f"학습 중 오류 발생: {e}")
        # ==== ADDED: 예외 발생 시 learning_done=True 전송 ====
        try:
            print("📡 강화학습 오류 종료 신호 전송 중...")
            success = env.comm.send_pid_once(0.0, 0.0, 0.0, True, False, True)  # learning_done=True
            if success:
                print("✅ 강화학습 오류 종료 신호 전송 성공")
                Logger.log("INFO", "📤 학습 오류 종료 신호 전송 성공")
            else:
                print("⚠️ 강화학습 오류 종료 신호 전송 실패")
                Logger.log("ERROR", "학습 오류 종료 신호 전송 실패")
        except Exception as e2:
            print(f"⚠️ 강화학습 오류 종료 신호 전송 오류: {e2}")
            Logger.log("ERROR", f"학습 오류 종료 신호 전송 오류: {e2}")
        
        # ==== ADDED: 모든 데이터 저장 ====
        DataSaver.save_all_data(env)
        
        env.comm.close()
    finally:
        Logger.log("INFO", "🔚 Training program terminated.")
