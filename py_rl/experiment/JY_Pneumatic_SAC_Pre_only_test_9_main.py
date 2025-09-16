# Residual SAC Agent for Pneumatic Polishing System

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
# ==== ADDED for CSV ====
import csv  # <-- CSV 저장용 추가
# =========================
# UTILITIES
# =========================
class Logger:
    """공통 로깅 유틸리티"""
    @staticmethod
    def log(level, message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_icons = {"INFO": "ℹ️","SUCCESS": "✅","WARNING": "⚠️","ERROR": "❌","DEBUG": "🔍"}
        icon = level_icons.get(level, "ℹ️")
        print(f"[{timestamp}] {icon} {message}")

class DataSaver:
    """공통 데이터 저장 유틸리티"""
    @staticmethod
    def save_all_data(env, current_episode=None, force=True):
        """모든 데이터를 저장하는 공통 함수"""
        try:
            # Reward breakdown flush
            if current_episode is not None:
                env.rlogger.flush_if_needed(current_episode, force=force, episode_rewards=env.agent.episode_rewards)
            else:
                env.rlogger.flush_if_needed(env.cfg["EPISODES"], force=force, episode_rewards=env.agent.episode_rewards)
        except Exception as e:
            Logger.log("ERROR", f"reward breakdown flush 실패: {e}")
        
        # 제어 성능 지표 저장
        try:
            Logger.log("INFO", "📊 제어 성능 지표 저장 중...")
            env.cplogger.save_performance_summary()
            env.cplogger.generate_plots()
            Logger.log("INFO", "✅ 제어 성능 지표 저장 완료!")
        except Exception as e:
            Logger.log("ERROR", f"제어 성능 지표 저장 실패: {e}")
        
        # Learning Done 폴더에 파일들 복사
        try:
            Logger.log("INFO", "📁 Learning Done 폴더에 파일들 복사 중...")
            env.ldlogger.copy_episode_rewards(env.agent.episode_rewards, env.rlogger.log_dir)
            env.ldlogger.copy_reward_breakdown(env.rlogger.log_dir)
            Logger.log("INFO", "✅ Learning Done 폴더 복사 완료!")
        except Exception as e:
            Logger.log("ERROR", f"Learning Done 폴더 복사 실패: {e}")
        
        Logger.log("INFO", "✅ 데이터 저장 완료!")

# =========================
# CONSTANTS
# =========================
# 상수 정의
class Constants:
    # =========================
    # 네트워크 관련 상수
    # =========================
    DEFAULT_HIDDEN_DIM = 256          # 신경망 은닉층 크기 (Actor/Critic 공통)
    DEFAULT_LR = 1e-3                 # 학습률 (Learning Rate) - 옵티마이저 업데이트 크기
    DEFAULT_GAMMA = 0.99              # 할인 인수 - 미래 보상의 현재 가치 비율
    DEFAULT_TAU = 0.005               # 소프트 업데이트 계수 - 타겟 네트워크 업데이트 속도
    
    # =========================
    # 액션 관련 상수
    # =========================
    DEFAULT_R_MIN = -0.05             # 최소 잔여 압력 (MPa) - 공압 딜레이 고려
    DEFAULT_R_MAX = 0.05              # 최대 잔여 압력 (MPa) - 공압 딜레이 고려
    
    # =========================
    # 주파수 관련 상수
    # =========================
    DEFAULT_SEND_FREQ = 100           # 송신 주파수 (Hz) - 로봇 제어 PC로 명령 전송
    DEFAULT_RECV_FREQ = 1000          # 수신 주파수 (Hz) - 로봇 제어 PC에서 상태 수신
    DEFAULT_TICK_TOL = 0.005          # 송신 주기 허용 오차 (초) - 100Hz에서 5ms
    
    # =========================
    # 학습 관련 상수
    # =========================
    DEFAULT_BATCH_SIZE = 128          # 배치 크기 - 한 번에 학습할 경험 개수
    DEFAULT_REPLAY_WARMUP = 21000     # 리플레이 버퍼 워밍업 - 학습 시작 전 최소 경험 수 (30초간 데이터)
    DEFAULT_UPDATE_FREQ = 10          # 네트워크 업데이트 주기 (Hz) - 1초에 10번 학습
    DEFAULT_EPISODES = 250            # 총 에피소드 수 - 전체 학습 횟수
    DEFAULT_MAX_STEPS = 3000          # 에피소드당 최대 스텝 수 - 한 에피소드 최대 길이
    DEFAULT_HER_SAMPLES = 12          # HER 샘플 수 - 실패한 경험을 재활용하는 개수
    
    # =========================
    # 네트워킹 관련 상수
    # =========================
    DEFAULT_HOST = "0.0.0.0"          # 서버 호스트 - 모든 IP에서 접속 허용
    DEFAULT_PORT = 8888               # 통신 포트 번호
    DEFAULT_RECV_TIMEOUT = 0.5        # 수신 타임아웃 (초) - 데이터 수신 대기 시간
    DEFAULT_RECV_LOOP_TIMEOUT = 0.05  # 수신 루프 타임아웃 (초) - 내부 루프 대기 시간
    DEFAULT_COMM_FAIL_MAX = 3         # 최대 통신 실패 횟수 - 연속 실패 시 경고
    DEFAULT_COMM_RETRY_DELAY = 0.1    # 통신 재시도 지연 (초) - 실패 후 재시도 간격
    
    # =========================
    # 메모리 관련 상수
    # =========================
    DEFAULT_MAX_REWARDS_HISTORY = 1000        # 최대 보상 기록 수 - 메모리 절약을 위한 제한
    DEFAULT_REPLAY_BUFFER_SIZE = 2000000      # 리플레이 버퍼 크기 - 저장할 경험의 최대 개수
    
    # =========================
    # 경로 관련 상수
    # =========================
    DEFAULT_MODEL_SAVE_DIR = "/home/katech/Robot-Polishing-RL-system/py_rl/saved_agents"  # 모델 저장 경로
    DEFAULT_LOG_DIR = "/home/katech/Robot-Polishing-RL-system/py_rl/experiment_logs"      # 로그 저장 경로
    
    # =========================
    # 기타 상수
    # =========================
    WAIT_MESSAGE_INTERVAL = 0.04      # 대기 메시지 출력 간격 (초) - 상태 표시 주기
    DEFAULT_FORCE_VALUE = -30.0       # 기본 힘 값 (N) - 데이터 없을 때 사용
    MAX_RETRIES = 3                   # 최대 재시도 횟수 - 일반적인 재시도 제한
    SUCCESS_REWARD = 10.0             # 성공 보상 - 목표 달성 시 추가 보상
    BAND_TOLERANCE = 5.0              # 밴드 허용 오차 (N) - 목표 힘 주변 허용 범위
    
    # =========================
    # 보상함수 관련 상수
    # =========================
    # 보상 가중치 (수렴>유지>안정성 순서로 중요도)
    REWARD_WEIGHT_PROGRESS = 2.0      # 진행도 가중치 - 목표에 가까워지는 정도
    REWARD_WEIGHT_BAND = 0.6          # 밴드 내 유지 가중치 - 허용 범위 내에 있는 정도
    REWARD_WEIGHT_STICK = 0.6         # 연속 밴드 유지 가중치 - 밴드 내에서 계속 유지
    REWARD_WEIGHT_EDOT = 0.02         # 힘 변화율 페널티 가중치 - 힘의 급격한 변화 방지
    REWARD_WEIGHT_DU = 0.04           # 액션 변화 페널티 가중치 - 액션의 급격한 변화 방지
    
    # 보상 범위 및 임계값
    REWARD_MIN = -15.0                # 최소 보상값 - 보상 하한선 (안정성)
    REWARD_MAX = 15.0                 # 최대 보상값 - 보상 상한선 (안정성)
    TAU_REQUIRED_SECONDS = 10         # 밴드 유지 요구 시간 (초) - 성공 판정 기준
    
    # 밴드 관련 상수
    BAND_TOLERANCE_N = 5.0            # 밴드 허용 오차 (N) - 목표 힘 ±5N 범위
    BAND_SUCCESS_MESSAGE = "🎯 ±{:.1f}N 밴드 내 {:.1f}s 유지 성공"  # 성공 메시지 템플릿

# =========================
# CONFIG
# =========================
# 기본 설정
_BASE_CONFIG = {
    # Neural Network
    "STATE_DIM": 6,
    "ACTION_DIM": 1,
    "HIDDEN": Constants.DEFAULT_HIDDEN_DIM,
    "LR": Constants.DEFAULT_LR,
    "GAMMA": Constants.DEFAULT_GAMMA,
    "TAU": Constants.DEFAULT_TAU,
    "AUTO_ENTROPY": True,
    # Residual limits (MPa) - 공압 딜레이 고려하여 범위 축소
    "R_MIN": Constants.DEFAULT_R_MIN,
    "R_MAX": Constants.DEFAULT_R_MAX,
    # Scheduling - 송신/수신 주파수 설정
    "SEND_FREQ_HZ": Constants.DEFAULT_SEND_FREQ,  # 송신 주파수 (Hz)
    "RECV_FREQ_HZ": Constants.DEFAULT_RECV_FREQ,  # 수신 주파수 (Hz) - 로봇 제어 PC에서 받는 주파수
    "TICK_TOL": Constants.DEFAULT_TICK_TOL,  # 🚀 수정: 100Hz에서 5ms 허용오차 (기존 20ms → 5ms)
    # Training
    "BATCH_SIZE": Constants.DEFAULT_BATCH_SIZE,
    "REPLAY_WARMUP": Constants.DEFAULT_REPLAY_WARMUP,
    "UPDATE_FREQ_HZ": Constants.DEFAULT_UPDATE_FREQ,
    # Networking
    "HOST": Constants.DEFAULT_HOST,
    "PORT": Constants.DEFAULT_PORT,
    "RECV_TIMEOUT_SEC": Constants.DEFAULT_RECV_TIMEOUT,
    "RECV_LOOP_TIMEOUT_SEC": Constants.DEFAULT_RECV_LOOP_TIMEOUT,
    "COMM_FAIL_MAX": Constants.DEFAULT_COMM_FAIL_MAX,
    "COMM_RETRY_DELAY": Constants.DEFAULT_COMM_RETRY_DELAY,
    # Episode
    "EPISODES": Constants.DEFAULT_EPISODES,
    "MAX_EPISODE_STEPS": Constants.DEFAULT_MAX_STEPS,
    # Model saving
    "MODEL_SAVE_DIR": Constants.DEFAULT_MODEL_SAVE_DIR,
    # Logging paths
    "LOG_DIR": Constants.DEFAULT_LOG_DIR,
    # Memory management
    "MAX_EPISODE_REWARDS_HISTORY": Constants.DEFAULT_MAX_REWARDS_HISTORY,
    # HER settings
    "HER_SAMPLES": Constants.DEFAULT_HER_SAMPLES,  # HER 샘플 개수
}

def create_config(send_freq_hz=None, recv_freq_hz=None):
    """송신/수신 주파수를 기반으로 CONFIG 생성"""
    config = _BASE_CONFIG.copy()
    
    if send_freq_hz is not None:
        if send_freq_hz <= 0 or send_freq_hz > 1000:
            raise ValueError(f"송신 주파수는 0과 1000 사이여야 합니다: {send_freq_hz}")
        config["SEND_FREQ_HZ"] = send_freq_hz
    
    if recv_freq_hz is not None:
        if recv_freq_hz <= 0 or recv_freq_hz > 10000:
            raise ValueError(f"수신 주파수는 0과 10000 사이여야 합니다: {recv_freq_hz}")
        config["RECV_FREQ_HZ"] = recv_freq_hz
    
    if config["RECV_FREQ_HZ"] < config["SEND_FREQ_HZ"]:
        raise ValueError(f"수신 주파수({config['RECV_FREQ_HZ']}Hz)는 송신 주파수({config['SEND_FREQ_HZ']}Hz)보다 높거나 같아야 합니다")
    
    config["TICK_SEC"] = 1.0 / config["SEND_FREQ_HZ"]
    config["RECV_INTERVAL_SEC"] = 1.0 / config["RECV_FREQ_HZ"]
    return config

# 기본 CONFIG 생성
CONFIG = create_config()

# =========================
# SAC Models
# =========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
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
        log_prob_sum = log_prob.sum(1, keepdim=True)
        return action, log_prob_sum
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        return q1, q2

# =========================
# Replay Buffer
# =========================    
class ReplayBuffer:
    def __init__(self, capacity=Constants.DEFAULT_REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        if action.ndim == 1:
            action = action.reshape(-1, 1)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ==== ADDED: Control Performance Logger ====
class ControlPerformanceLogger:
    """
    9개 핵심 제어공학 지표를 계산하고 저장하는 클래스
    """
    def __init__(self, log_dir):
        self.base_log_dir = log_dir
        # 실행별 고유 폴더 생성
        now = datetime.now()
        timestamp = now.strftime("%y%m%d_%Hh%Mm")
        self.log_dir = os.path.join(log_dir, f"learning_done_{timestamp}")
        self.control_perf_dir = os.path.join(self.log_dir, "control_performance")
        os.makedirs(self.control_perf_dir, exist_ok=True)
        
        # 데이터 저장용 리스트들
        self.time_data = []
        self.force_data = []
        self.target_data = []
        self.error_data = []
        self.control_effort_data = []
        self.pi_output_data = []
        
        # 에피소드별 지표 저장
        self.episode_metrics = []
        
        print(f"📁 Control Performance 저장 폴더: {self.control_perf_dir}")

    def add_data_point(self, time, force, target, control_effort, pi_output):
        """실시간 데이터 포인트 추가 (100Hz에서 호출)"""
        self.time_data.append(time)
        self.force_data.append(force)
        self.target_data.append(target)
        self.error_data.append(abs(force - target))
        self.control_effort_data.append(control_effort)
        self.pi_output_data.append(pi_output)

    def calculate_episode_metrics(self, episode_num):
        """에피소드별 9개 핵심 지표 계산"""
        if not self.time_data:
            return None
            
        metrics = {
            'episode': episode_num,
            'rmse': self._calculate_rmse(),
            'sse': self._calculate_sse(),
            'rise_time': self._calculate_rise_time(),
            'settling_time': self._calculate_settling_time(),
            'overshoot': self._calculate_overshoot(),
            'control_effort': self._calculate_control_effort(),
            'residual_effectiveness': self._calculate_residual_effectiveness(),
            'pi_rl_synergy': self._calculate_pi_rl_synergy(),
            'learning_progress': self._calculate_learning_progress()
        }
        
        self.episode_metrics.append(metrics)
        return metrics

    def _calculate_rmse(self):
        """RMSE 계산"""
        if not self.error_data:
            return None
        return np.sqrt(np.mean(np.square(self.error_data)))

    def _calculate_sse(self):
        """SSE 계산 (마지막 10% 구간의 평균 오차)"""
        if not self.error_data:
            return None
        last_10_percent = max(1, int(len(self.error_data) * 0.1))
        return np.mean(self.error_data[-last_10_percent:])

    def _calculate_rise_time(self):
        """Rise Time 계산 (10% → 90%)"""
        if not self.force_data or not self.target_data:
            return None
            
        target = self.target_data[0]
        target_10 = target * 0.1
        target_90 = target * 0.9
        
        force_array = np.array(self.force_data)
        time_array = np.array(self.time_data)
        
        idx_10 = np.where(force_array >= target_10)[0]
        idx_90 = np.where(force_array >= target_90)[0]
        
        if len(idx_10) > 0 and len(idx_90) > 0:
            return float(time_array[idx_90[0]] - time_array[idx_10[0]])
        return None

    def _calculate_settling_time(self):
        """Settling Time 계산 (±5% 오차 범위)"""
        if not self.force_data or not self.target_data:
            return None
            
        target = self.target_data[0]
        tolerance = target * 0.05
        force_array = np.array(self.force_data)
        time_array = np.array(self.time_data)
        
        settled_indices = np.where(np.abs(force_array - target) <= tolerance)[0]
        if len(settled_indices) > 0:
            return float(time_array[settled_indices[0]])
        return None

    def _calculate_overshoot(self):
        """Overshoot 계산"""
        if not self.force_data or not self.target_data:
            return None
            
        target = self.target_data[0]
        max_force = np.max(self.force_data)
        
        if max_force > target:
            return float(((max_force - target) / target) * 100)
        return 0.0

    def _calculate_control_effort(self):
        """Control Effort 계산 (RL residual의 총합)"""
        if not self.control_effort_data:
            return None
        return float(np.sum(np.abs(self.control_effort_data)))

    def _calculate_residual_effectiveness(self):
        """Residual Effectiveness 계산 (RL residual과 오차의 상관계수)"""
        if len(self.control_effort_data) < 2 or len(self.error_data) < 2:
            return None
        try:
            correlation = np.corrcoef(self.control_effort_data, self.error_data)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except (ValueError, np.linalg.LinAlgError, IndexError):
            return 0.0

    def _calculate_pi_rl_synergy(self):
        """PI-RL Synergy 계산 (PI 출력과 RL residual의 상관계수)"""
        if len(self.pi_output_data) < 2 or len(self.control_effort_data) < 2:
            return None
        try:
            correlation = np.corrcoef(self.pi_output_data, self.control_effort_data)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except (ValueError, np.linalg.LinAlgError, IndexError):
            return 0.0

    def _calculate_learning_progress(self):
        """Learning Progress 계산 (에피소드별 RMSE 개선)"""
        if len(self.episode_metrics) < 2:
            return None
        
        rmse_values = [ep['rmse'] for ep in self.episode_metrics if ep['rmse'] is not None]
        if len(rmse_values) < 2:
            return None
            
        # 선형 회귀 기울기 (음수일수록 개선)
        x = np.arange(len(rmse_values))
        slope = np.polyfit(x, rmse_values, 1)[0]
        return float(slope)

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
        """전체 성능 요약 저장"""
        if not self.episode_metrics:
            return
            
        summary_path = os.path.join(self.control_perf_dir, "performance_summary.csv")
        
        with open(summary_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Mean", "Std", "Min", "Max", "Unit"])
            
            # 각 지표별 통계 계산
            for metric_name in ['rmse', 'sse', 'rise_time', 'settling_time', 'overshoot', 
                              'control_effort', 'residual_effectiveness', 'pi_rl_synergy', 'learning_progress']:
                values = [ep[metric_name] for ep in self.episode_metrics if ep[metric_name] is not None]
                
                if values:
                    writer.writerow([
                        metric_name,
                        f"{np.mean(values):.4f}",
                        f"{np.std(values):.4f}",
                        f"{np.min(values):.4f}",
                        f"{np.max(values):.4f}",
                        self._get_metric_unit(metric_name)
                    ])

    def _get_metric_unit(self, metric_name):
        """지표별 단위 반환"""
        units = {
            'rmse': 'N',
            'sse': 'N', 
            'rise_time': 's',
            'settling_time': 's',
            'overshoot': '%',
            'control_effort': 'MPa·s',
            'residual_effectiveness': '-',
            'pi_rl_synergy': '-',
            'learning_progress': 'N/episode'
        }
        return units.get(metric_name, '')

    def generate_plots(self):
        """각 지표별 시각화 생성"""
        if not self.episode_metrics:
            return
            
        for metric_name in ['rmse', 'sse', 'rise_time', 'settling_time', 'overshoot', 
                          'control_effort', 'residual_effectiveness', 'pi_rl_synergy', 'learning_progress']:
            self._plot_metric(metric_name)

    def _plot_metric(self, metric_name):
        """개별 지표 시각화"""
        values = [ep[metric_name] for ep in self.episode_metrics if ep[metric_name] is not None]
        episodes = [ep['episode'] for ep in self.episode_metrics if ep[metric_name] is not None]
        
        if not values:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, values, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel(f'{metric_name.upper()}', fontsize=12)
        plt.title(f'{metric_name.upper()} Over Episodes', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 평균선 추가
        if len(values) > 1:
            avg_value = np.mean(values)
            plt.axhline(y=avg_value, color='r', linestyle='--', alpha=0.7, 
                       label=f'Average: {avg_value:.4f}')
            plt.legend()
        
        png_path = os.path.join(self.control_perf_dir, f"{metric_name}.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

    def reset_episode_data(self):
        """에피소드 데이터 초기화"""
        self.time_data.clear()
        self.force_data.clear()
        self.target_data.clear()
        self.error_data.clear()
        self.control_effort_data.clear()
        self.pi_output_data.clear()

# ==== ADDED: Learning Done Logger ====
class LearningDoneLogger:
    """
    학습 완료 시 전체 로깅을 관리하는 클래스
    """
    def __init__(self, log_dir):
        self.base_log_dir = log_dir
        now = datetime.now()
        timestamp = now.strftime("%y%m%d_%Hh%Mm")
        self.log_dir = os.path.join(log_dir, f"learning_done_{timestamp}")
        self.reward_breakdown_dir = os.path.join(self.log_dir, "reward_breakdown")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.reward_breakdown_dir, exist_ok=True)
        
        print(f"📁 Learning Done 저장 폴더: {self.log_dir}")

    def copy_episode_rewards(self, episode_rewards, rlogger_dir):
        """episode_rewards 파일들을 learning_done 폴더로 복사"""
        # CSV 복사
        src_csv = os.path.join(rlogger_dir, "episode_rewards.csv")
        dst_csv = os.path.join(self.log_dir, "episode_rewards.csv")
        if os.path.exists(src_csv):
            import shutil
            shutil.copy2(src_csv, dst_csv)
        
        # PNG 복사
        src_png = os.path.join(rlogger_dir, "episode_rewards.png")
        dst_png = os.path.join(self.log_dir, "episode_rewards.png")
        if os.path.exists(src_png):
            import shutil
            shutil.copy2(src_png, dst_png)

    def copy_reward_breakdown(self, rlogger_dir):
        """reward_breakdown 파일들을 learning_done/reward_breakdown 폴더로 복사"""
        import shutil
        import glob
        
        # reward_breakdown 관련 파일들 찾기
        pattern = os.path.join(rlogger_dir, "reward_breakdown*")
        files = glob.glob(pattern)
        
        for src_file in files:
            filename = os.path.basename(src_file)
            dst_file = os.path.join(self.reward_breakdown_dir, filename)
            shutil.copy2(src_file, dst_file)

# ==== ADDED: Reward Breakdown Logger ====
class RewardBreakdownLogger:
    """
    스텝 단위 보상 항목 로깅을 메모리에 모아두고
    Ctrl+C나 강화학습 완료 시 CSV 저장 + PNG 시각화를 수행
    """
    def __init__(self, log_dir):
        self.base_log_dir = log_dir
        # 실행별 고유 폴더 생성 (오늘날짜_시작시간 형식)
        now = datetime.now()
        timestamp = now.strftime("%y%m%d_%Hh%Mm")
        self.log_dir = os.path.join(log_dir, f"reward_breakdown_{timestamp}")
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
        force=True이면 언제든 실행, False이면 실행하지 않음.
        episode_rewards: 에피소드별 보상 리스트 (선택사항)
        """
        # 강제 실행이 아니면 실행하지 않음 (자동 저장 비활성화)
        if not force:
            return
        
        # 데이터가 없으면 실행하지 않음
        if not self.rows:
            return
            
        # CSV Append
        self._write_csv_append()
        
        # ==== ADDED: reward_breakdown CSV 저장 ====
        self.save_reward_breakdown_csv()
        
        # 에피소드별 보상 저장 (제공된 경우)
        if episode_rewards is not None:
            self.save_episode_rewards(episode_rewards)
            # 에피소드별 보상 그래프도 생성
            self.generate_episode_reward_graph(episode_rewards)
        
        # PNG 생성 (전체 데이터)
        start_ep = min(row["episode"] for row in self.rows)
        end_ep = max(row["episode"] for row in self.rows)
        self._plot_png(start_ep, end_ep)
        
        # 저장 완료 메시지
        print(f"✅ Reward breakdown 저장 완료: {self.log_dir}")
        print(f"   📊 CSV: reward_breakdown.csv")
        if episode_rewards is not None:
            print(f"   📈 CSV: episode_rewards.csv")
        print(f"   📈 PNG: reward_breakdown_*_ep{start_ep}-{end_ep}.png (5개 파일)")
        
        # CSV에 저장했으니 rows를 비워도 되지만,
        # 전체 기간 그래프를 원할 수도 있어 유지 선택 가능.
        # 여기서는 메모리 절약 위해 비움.
        self.rows.clear()

# =========================
# Residual SAC Agent
# =========================   
class ResidualSACAgent:
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
        self.replay = ReplayBuffer()
        self.total_steps = 0
        self.episode_rewards = []
        self.max_rewards_history = cfg.get("MAX_EPISODE_REWARDS_HISTORY", 1000)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, log_prob = self.actor.sample(state)
        action = action.cpu().numpy().flatten()
        return float(action[0] * (self.cfg["R_MAX"]))
    
    def store_transition(self, state, action, reward, next_state, done):
        norm_action = action / self.cfg["R_MAX"]
        self.replay.push(state, norm_action, reward, next_state, done)

    def update_parameters(self, batch_size=None):
        bs = batch_size or self.cfg["BATCH_SIZE"]
        if len(self.replay) < bs: return

        s, a, r, ns, d = self.replay.sample(bs)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        with torch.no_grad():
            na, nlogp = self.actor.sample(ns)
            q1n, q2n = self.critic_target(ns, na)
            min_qn = torch.min(q1n, q2n) - self.alpha * nlogp
            y = r + (1 - d) * self.gamma * min_qn

        q1, q2 = self.critic(s, a)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        pi, logp = self.actor.sample(s)
        q1_pi, q2_pi = self.critic(s, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = ((self.alpha * logp) - min_q_pi).mean()
        self.actor_opt.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        if self.auto_entropy_tuning:
            logp_entropy = logp.squeeze(1)
            a_loss = -(self.log_alpha * (logp_entropy + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); a_loss.backward(); self.alpha_opt.step()
            self.alpha = self.log_alpha.exp()

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
class ResidualRLCommunicator:
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
        self.PY_TO_CPP_PACKET_FORMAT = ">HfBBBH"  # SOF, rl_residual, timing_accurate, episode_done, learning_done, checksum
        self.PY_TO_CPP_PACKET_SIZE = 11  # SOF(2) + rl_residual(4) + timing_accurate(1) + episode_done(1) + learning_done(1) + checksum(2) = 11 bytes
        self.PY_TO_CPP_SOF = 0xBBBB
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
    
    def send_residual(self, rl_residual, timing_accurate, episode_done, learning_done=False):
        try:
            data_part = struct.pack(">HfBBB", 
                                  self.PY_TO_CPP_SOF, 
                                  float(rl_residual), 
                                  bool(timing_accurate), 
                                  bool(episode_done),
                                  bool(learning_done))
            checksum = self.calculate_crc16(data_part)
            final_packet = struct.pack(self.PY_TO_CPP_PACKET_FORMAT, 
                                     self.PY_TO_CPP_SOF, 
                                     float(rl_residual), 
                                     bool(timing_accurate), 
                                     bool(episode_done),
                                     bool(learning_done),
                                     checksum)
            self.conn.sendall(final_packet)
            with self.stats_lock:
                self.packets_sent += 1
            return True
        except Exception as e:
            self._log("ERROR", f"residual 전송 오류: {e}")
            return False
        
    def send_reset(self):
        try:
            reset_data = struct.pack(">HBxxxH", 0xCCCC, 1, 0)
            checksum = self.calculate_crc16(reset_data[:-2])
            reset_packet = struct.pack(">HBxxxH", 0xCCCC, 1, checksum)
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
class PneumaticPolishingEnvironment:
    def __init__(self, cfg=None):
        if cfg is None:
            raise ValueError("cfg 파라미터가 필요합니다")
        self.cfg = cfg
        self.agent = ResidualSACAgent(cfg)
        self.comm = ResidualRLCommunicator(cfg["HOST"], cfg["PORT"], cfg["RECV_TIMEOUT_SEC"], cfg["RECV_LOOP_TIMEOUT_SEC"], cfg)
        self.prev_residual = 0.0
        self.episode_step = 0
        self.max_episode_steps = cfg["MAX_EPISODE_STEPS"]
        self.current_episode_reward = 0.0
        self.best_episode_reward = -float("inf")
        self.best_agent_episode = -1
        self.rl_inactive_count = 0
        self.max_rl_inactive_steps = 250
        self.rl_active_in_episode = False
        self.total_rl_active_steps = 0
        self.fail_count = 0
        self.FAIL_MAX = cfg["COMM_FAIL_MAX"]
        self.last_log_time = None
        self.previous_target_force = 0.0
        self.last_valid_state = None
        self.last_sander_active = False
        self.update_interval = 1.0 / cfg["UPDATE_FREQ_HZ"]
        self.last_update_time = None
        
        # ==== ADDED: 에피소드 전환 감지용 변수들 ====
        self.episode_transition_detected = False  # 에피소드 전환 감지 플래그
        self.waiting_for_episode_start = False    # 새 에피소드 시작 대기 중인지
        self.episode_end_confirmed = False       # 에피소드 종료 확인됨
        
        # --- HER Reward 전용 변수 ---
        self.band_tol_N = Constants.BAND_TOLERANCE_N
        self.tau_req_s = Constants.TAU_REQUIRED_SECONDS
        self.tau_req_steps = int(self.tau_req_s * self.cfg["SEND_FREQ_HZ"])
        self.band_timer = 0
        self._last_e = None

        # reward weights (수렴>유지>안정성)
        self.w_prog = Constants.REWARD_WEIGHT_PROGRESS
        self.w_band = Constants.REWARD_WEIGHT_BAND
        self.w_stick = Constants.REWARD_WEIGHT_STICK
        self.w_edot = Constants.REWARD_WEIGHT_EDOT
        self.w_du = Constants.REWARD_WEIGHT_DU
        self.R_success = Constants.SUCCESS_REWARD

        # HER 전용 상태 변수 (원래 보상과 독립적으로 계산)
        self._last_e_her = None
        self.prev_residual_her = 0.0  # HER 전용 residual 변수

        # |Δu| 계산 전용 변수 (prev_residual과 분리)
        self.prev_action_for_du = 0.0          # |Δu| 계산 기준(원래 보상)
        self.prev_action_for_du_her = 0.0      # |Δu| 계산 기준(HER 보상)

        # ==== ADDED: reward breakdown logger ====
        self.rlogger = RewardBreakdownLogger(self.cfg["LOG_DIR"])
        
        # ==== ADDED: control performance logger ====
        self.cplogger = ControlPerformanceLogger(self.cfg["LOG_DIR"])
        
        # ==== ADDED: learning done logger ====
        self.ldlogger = LearningDoneLogger(self.cfg["LOG_DIR"])
        
    def _log(self, level, message):
        Logger.log(level, message)

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
    
    def limit_residual(self, r):
        # 공압 딜레이 고려하여 즉시 전송 (slew rate 제한 해제)
        r = float(np.clip(r, self.cfg["R_MIN"], self.cfg["R_MAX"]))
        # 이전 값 업데이트 (로깅용)
        self.prev_residual = r
        return r
    
    # ---- HER Reward Methods ----
    def _update_band_timer(self, state):
        e = abs(state[0] - state[1])
        if e <= self.band_tol_N:
            self.band_timer += 1
        else:
            self.band_timer = 0

    def compute_reward_her(self, s, a_residual, s_next, g_force, terminal_success=False, return_components=False):
        """
        HER-친화적 보상함수
        """
        curF = float(s[0])
        curF_next = float(s_next[0])
        e = abs(curF - g_force)
        e_next = abs(curF_next - g_force)

        edot_abs = abs(float(s[3]))
        du_abs = abs(a_residual - self.prev_action_for_du)

        if self._last_e is None:
            prog = 0.0
        else:
            prog = (self._last_e - e)
        self._last_e = e

        R = 0.0
        R += self.w_prog * prog
        in_band_now = 1.0 if e <= self.band_tol_N else 0.0
        in_band_next = 1.0 if e_next <= self.band_tol_N else 0.0
        R += self.w_band * in_band_now
        R += self.w_stick * (in_band_now * in_band_next)
        R -= self.w_edot * edot_abs
        R -= self.w_du * du_abs
        if terminal_success:
            R += self.R_success

        R = float(np.clip(R, Constants.REWARD_MIN, Constants.REWARD_MAX))

        if return_components:
            return R, {
                "prog": float(prog),
                "in_band_now": float(in_band_now),
                "edot_abs": float(edot_abs),
                "du_abs": float(du_abs)
            }
        return R

    def compute_reward_her_independent(self, s, a_residual, s_next, g_force, terminal_success=False, return_components=False):
        """
        HER 전용 독립적 보상함수 (원래 보상과 상태 공유하지 않음)
        """
        curF = float(s[0])
        curF_next = float(s_next[0])
        e = abs(curF - g_force)
        e_next = abs(curF_next - g_force)

        edot_abs = abs(float(s[3]))
        du_abs = abs(a_residual - self.prev_action_for_du_her)

        # HER 전용 진행도 계산 (독립적)
        if self._last_e_her is None:
            prog = 0.0
        else:
            prog = (self._last_e_her - e)
        self._last_e_her = e

        R = 0.0
        R += self.w_prog * prog
        in_band_now = 1.0 if e <= self.band_tol_N else 0.0
        in_band_next = 1.0 if e_next <= self.band_tol_N else 0.0
        R += self.w_band * in_band_now
        R += self.w_stick * (in_band_now * in_band_next)
        R -= self.w_edot * edot_abs
        R -= self.w_du * du_abs
        if terminal_success:
            R += self.R_success

        R = float(np.clip(R, Constants.REWARD_MIN, Constants.REWARD_MAX))

        if return_components:
            return R, {
                "prog": float(prog),
                "in_band_now": float(in_band_now),
                "edot_abs": float(edot_abs),
                "du_abs": float(du_abs)
            }
        return R

    def is_done(self, state):
        if self.episode_step >= self.max_episode_steps:
            return True, False
        if state[0] > 100.0:
            self._log("WARNING", f"안전: 힘 과다 {state[0]:.1f}N > 100N")
            return True, False
        self._update_band_timer(state)
        if self.band_timer >= self.tau_req_steps:
            self._log("SUCCESS", Constants.BAND_SUCCESS_MESSAGE.format(self.band_tol_N, self.tau_req_s))
            return True, True
        return False, False

    # ---- 학습 주기 제어 ----
    def should_update_now(self):
        now = time.perf_counter()
        if self.last_update_time is None:
            self.last_update_time = now
            return True
        if now - self.last_update_time >= self.update_interval:
            self.last_update_time = now
            return True
        return False
    
    # ---- RL activity monitor ----
    def check_rl_status(self, sander_active):
        if sander_active:
            self.rl_inactive_count = 0
            self.rl_active_in_episode = True
            self.total_rl_active_steps += 1
            
            # ==== ADDED: 에피소드 시작 감지 (0→1 전환) ====
            if self.waiting_for_episode_start and not self.last_sander_active:
                self._log("INFO", "🎬 새 에피소드 시작 감지! (sander_active: 0→1)")
                self.waiting_for_episode_start = False
                self.episode_transition_detected = True
                self.episode_end_confirmed = False
            
            return "active"
        else:
            self.rl_inactive_count += 1
            
            # ==== ADDED: 에피소드 종료 감지 (1→0 전환) ====
            if self.last_sander_active and not sander_active:
                self._log("INFO", "🏁 에피소드 종료 감지! (sander_active: 1→0)")
                self.episode_end_confirmed = True
                self.waiting_for_episode_start = True
                return "episode_end"
            
            if self.rl_inactive_count >= self.max_rl_inactive_steps:
                self._log("WARNING", "RL 비활성 너무 오래 → 에피소드 종료")
                return "terminate"
            return "inactive"

    def end_episode_fast_with_reliable_flag(self, ep, target_force):
        self._log("INFO", f"🎯 에피소드 {ep+1} 완료 (단계 {self.episode_step})")
        success = self.comm.send_residual(0.0, True, True, False)  # episode_done=True, learning_done=False
        if success:
            self._log("INFO", f"📡 Episode done 신호 전송 성공")
        else:
            self._log("WARNING", f"⚠️ Episode done 신호 전송 실패")
        confirmation_start = time.perf_counter()
        max_confirmation_time = 0.5
        while (time.perf_counter() - confirmation_start) < max_confirmation_time:
            state, _ = self.comm.get_latest_state()
            if state is not None and abs(state[1] - target_force) > 1.0:
                elapsed = time.perf_counter() - confirmation_start
                self._log("INFO", f"✅ 에피소드 종료 확인! {target_force:.1f}N → {state[1]:.1f}N ({elapsed:.3f}s 소요)")
                break
            time.sleep(self.cfg["RECV_INTERVAL_SEC"])
        else:
            self._log("INFO", f"⚡ Episode done 신호 전송 후 에피소드 종료 (500ms 타임아웃) - 다음 에피소드로 진행")
        success = self.comm.send_residual(0.0, True, False, False)  # episode_done=False, learning_done=False
        return True

    def reset_episode(self):
        self.prev_residual = 0.0
        self.episode_step = 0
        self.current_episode_reward = 0.0
        self.rl_inactive_count = 0
        self.rl_active_in_episode = False
        self.last_log_time = None
        self.last_valid_state = None
        self.last_sander_active = False
        self.last_update_time = None
        self.band_timer = 0
        self._last_e = None
        self._last_e_her = None
        self.prev_residual_her = 0.0
        # |Δu| 계산 전용 변수 초기화
        self.prev_action_for_du = 0.0
        self.prev_action_for_du_her = 0.0
        
        # ==== ADDED: 에피소드 전환 관련 변수 초기화 ====
        self.episode_transition_detected = False
        self.waiting_for_episode_start = False
        self.episode_end_confirmed = False
        state, _ = self.comm.get_latest_state()
        if state is not None:
            self.previous_target_force = state[1]
            self._log("INFO", f"에피소드 목표 힘: {self.previous_target_force:.1f}N")
        ok = self.comm.send_reset()
        if ok:
            self._log("INFO", "\n--- 에피소드 리셋 ---")
            self._log("INFO", "로봇PC: 1kHz PI 실행 중, 각 틱마다 RL residual(보유) 추가.")
        else:
            self._log("WARNING", "리셋 신호 실패 (계속 진행).")
        return ok

    # ---- main loop ----
    def run_training(self, episodes=None):
        episodes = episodes or self.cfg["EPISODES"]
        if not self.comm.connect():
            self._log("ERROR", "로봇PC 연결 실패")
            return
        model_save_dir = self.cfg["MODEL_SAVE_DIR"]
        os.makedirs(model_save_dir, exist_ok=True)
        self._log("INFO", f"📁 모델 저장 디렉토리: {model_save_dir}")
        self._log("INFO", "🚀 최적화된 Residual RL 학습 시작 - 버전 8")
        self._log("INFO", f"📡 송신: {self.cfg['SEND_FREQ_HZ']}Hz residual 출력 ({self.cfg['TICK_SEC']:.3f}초 간격)")
        self._log("INFO", f"📥 수신: {self.cfg['RECV_FREQ_HZ']}Hz 상태 수신 ({self.cfg['RECV_INTERVAL_SEC']:.3f}초 간격)")
        self._log("INFO", f"⚡ 최적화: {self.cfg['SEND_FREQ_HZ']}Hz 안정적 송신을 위한 빠른 에피소드 전환")
        self._log("INFO", f"⏱️  에피소드: {self.cfg['MAX_EPISODE_STEPS']} 단계 ({self.cfg['SEND_FREQ_HZ']}Hz에서 {self.cfg['MAX_EPISODE_STEPS'] * self.cfg['TICK_SEC']:.1f}초)")
        self._log("INFO", "\n🔄 RL 활성화 대기 중...")
        wait_start_time = time.perf_counter()
        while True:
            state, sander_active = self.comm.get_latest_state()
            if sander_active:
                wait_duration = time.perf_counter() - wait_start_time
                self._log("INFO", f"🎯 RL 활성화! ({wait_duration:.1f}s 대기)")
                break
            if state is not None:
                current_force = state[0]
                target_force = state[1]
                Logger.log("INFO", f"⏳ Waiting... Current Force: {current_force:.1f}N, Target: {target_force:.1f}N")
            time.sleep(Constants.WAIT_MESSAGE_INTERVAL)
            if time.perf_counter() - wait_start_time > 300:
                self._log("WARNING", "\n⚠️ RL 활성화 타임아웃 (5분)")
                return
        episode_stats = []
        for ep in range(episodes):
            self._log("INFO", f"\n🎬 === 에피소드 {ep+1}/{episodes} 시작 ===")
            
            # ==== ADDED: 새 에피소드 시작 대기 (sander_active=1 감지) ====
            if ep > 0:  # 첫 번째 에피소드가 아닌 경우에만 대기
                self._log("INFO", "🔄 로봇 리셋 완료 후 새 에피소드 시작 대기 중... (sander_active=1)")
                wait_start = time.perf_counter()
                waiting_message_shown = False  # 대기 메시지 표시 플래그
                while not self.episode_transition_detected:
                    state, sander_active = self.comm.get_latest_state()
                    
                    # ==== ADDED: 대기 중 상태를 1번만 표시 ====
                    if state is not None and not waiting_message_shown and not sander_active:
                        self._log("INFO", "⏳ 로봇 z축 이동 중... sander_active=0 대기 중")
                        waiting_message_shown = True
                    
                    if sander_active and self.waiting_for_episode_start:
                        wait_duration = time.perf_counter() - wait_start
                        self._log("INFO", f"🎬 새 에피소드 시작 감지! (sander_active: 0→1) - {wait_duration:.1f}초 대기")
                        self.waiting_for_episode_start = False
                        self.episode_transition_detected = True
                        self.episode_end_confirmed = False
                        break
                    
                    time.sleep(0.1)
            
            episode_start_state, episode_start_sander_active = self.comm.get_latest_state()
            if not episode_start_sander_active:
                self._log("WARNING", f"⚠️ 경고: 에피소드 {ep+1} 시작 시 RL 플래그 False")
                self._log("INFO", "🔄 RL 활성화 대기 중...")
                wait_start = time.perf_counter()
                while not episode_start_sander_active:
                    episode_start_state, episode_start_sander_active = self.comm.get_latest_state()
                    if time.perf_counter() - wait_start > 60:
                        self._log("WARNING", f"⚠️ 에피소드 {ep+1}에서 RL 활성화 대기 타임아웃")
                        break
                    time.sleep(0.1)
            episode_start_time = time.perf_counter()
            self.reset_episode()
            self._log("INFO", f"📡 에피소드 {ep+1} 시작")
            self.comm.send_residual(0.0, True, False, False)  # episode_done=False, learning_done=False
            prev_state = None
            prev_action = None
            prev_sander_active = False
            episode_packets_received = 0
            episode_packets_sent = 0
            episode_rl_active_steps = 0
            episode_start_perf_time = time.perf_counter()
            tick_count = 0
            
            while True:
                current_time = time.perf_counter()
                next_send_time = episode_start_perf_time + (tick_count + 1) * self.cfg["TICK_SEC"]
                if current_time >= next_send_time:
                    tick_count += 1
                    timing_accurate = abs(current_time - next_send_time) <= self.cfg["TICK_TOL"]
                    res = self.comm.get_latest_state()
                    if res[0] is None:
                        if self.last_valid_state is not None:
                            state = self.last_valid_state.copy()
                            sander_active = self.last_sander_active
                            self._log("DEBUG", f"데이터 없음 - 이전 상태 사용 (step {self.episode_step})")
                        else:
                            state = np.array([0.0, Constants.DEFAULT_FORCE_VALUE, Constants.DEFAULT_FORCE_VALUE, 0.0, 0.0, 0.0], dtype=np.float32)
                            sander_active = False
                            self._log("DEBUG", f"데이터 없음 - 기본값 사용 (step {self.episode_step})")
                        
                    else:
                        state, sander_active = res
                        self.last_valid_state = state.copy()
                        
                        self.last_sander_active = sander_active
                        episode_packets_received += 1
                        self.previous_target_force = state[1]
                else:
                    time.sleep(0.001)
                    continue

                self.episode_step += 1
                done, success = self.is_done(state)
                
                # ==== ADDED: 실시간 제어 지표 데이터 수집 ====
                if state is not None:
                    current_time = time.perf_counter() - episode_start_time
                    self.cplogger.add_data_point(
                        time=current_time,
                        force=state[0],
                        target=state[1],
                        control_effort=abs(rl_residual) if 'rl_residual' in locals() else 0.0,
                        pi_output=state[5]
                    )
                
                # ==== CHANGED: 보상 계산 + 로깅 + HER 저장 ====
                if prev_state is not None and prev_sander_active:
                    # (A) 원래 목표로 보상/저장
                    reward, comp = self.compute_reward_her(
                        prev_state, prev_action, state, prev_state[1],
                        terminal_success=success, return_components=True
                    )
                    self.agent.store_transition(prev_state, prev_action, reward, state, done)
                    self.current_episode_reward += reward

                    # 스텝 로깅 (HER 아님)
                    self.rlogger.log_step(
                        episode=ep+1,
                        step=self.episode_step,
                        prog=comp["prog"],
                        in_band_now=comp["in_band_now"],
                        edot_abs=comp["edot_abs"],
                        du_abs=comp["du_abs"],
                        reward=reward,
                        is_her=0
                    )

                    # HER: 실패한 경우에만 relabel transition 추가
                    if abs(prev_state[0] - prev_state[1]) > self.band_tol_N:
                        # 1. Config에서 HER 샘플 개수 가져오기
                        num_her_samples = self.cfg["HER_SAMPLES"]
                        
                        # 2. Trajectory 기반 future sampling (정석적 HER)
                        # 현재 달성된 힘을 기준으로 다양한 미래 목표 생성
                        current_force = state[0]
                        target_force = prev_state[1]
                        
                        # 3. 다양한 alternative goal 생성
                        future_goals = []
                        for i in range(num_her_samples):
                            if i == 0:
                                # 현재 달성된 힘
                                future_goals.append(current_force)
                            elif i == 1:
                                # 원래 목표와 현재 힘의 중간값
                                future_goals.append((current_force + target_force) / 2)
                            elif i == 2:
                                # 원래 목표 방향으로 약간 이동
                                direction = 1 if target_force > current_force else -1
                                future_goals.append(current_force + direction * np.random.uniform(0.5, 2.0))
                            else:
                                # 랜덤 노이즈 추가
                                future_goals.append(current_force + np.random.normal(0, 1.0))
                        
                        # 4. 각 alternative goal에 대해 HER transition 생성
                        for i in range(num_her_samples):
                            achieved_goal = future_goals[i]
                            
                            # HER 전용 독립적 보상함수 사용
                            her_reward, her_comp = self.compute_reward_her_independent(
                                prev_state, prev_action, state, achieved_goal, 
                                terminal_success=False, return_components=True
                            )
                            self.agent.store_transition(prev_state, prev_action, her_reward, state, done)
                            
                            # HER residual 업데이트는 루프 외부에서 수행

                            # 5. 스텝 로깅 (HER) - 첫 번째 샘플만 로깅
                            if i == 0:
                                self.rlogger.log_step(
                                    episode=ep+1,
                                    step=self.episode_step,
                                    prog=her_comp["prog"],
                                    in_band_now=her_comp["in_band_now"],
                                    edot_abs=her_comp["edot_abs"],
                                    du_abs=her_comp["du_abs"],
                                    reward=her_reward,
                                    is_her=1
                                )
                
                # HER 루프 완료 후 |Δu| 계산용 변수 업데이트
                if prev_action is not None:
                    self.prev_action_for_du = prev_action
                    self.prev_action_for_du_her = prev_action
                
                if (len(self.agent.replay) > self.cfg["REPLAY_WARMUP"] and 
                    self.should_update_now()):
                    self.agent.update_parameters(self.cfg["BATCH_SIZE"])
                
                rl_status = self.check_rl_status(sander_active)
                if rl_status == "terminate":
                    self._log("INFO", "RL 비활성 지속으로 에피소드 종료")
                    break
                elif rl_status == "episode_end":
                    self._log("INFO", "🏁 sander_active=0으로 에피소드 종료 감지 - 로봇 리셋 대기 중...")
                    # 에피소드 종료 처리
                    episode_done = True
                    rl_residual = 0.0
                    break
                
                if done or self.episode_step >= self.max_episode_steps:
                    episode_done = True
                    rl_residual = 0.0
                    if done:
                        if success:
                            self._log("INFO", f"🎯 에피소드 {ep+1} 성공 종료 (밴드 유지 성공, 단계 {self.episode_step}) - episode_done=True 전송")
                        else:
                            self._log("INFO", f"🎯 에피소드 {ep+1} 종료 (조건 만족, 단계 {self.episode_step}) - episode_done=True 전송")
                    else:
                        self._log("INFO", f"🎯 에피소드 {ep+1} 종료 (최대 스텝 도달, 단계 {self.episode_step}) - episode_done=True 전송")
                else:
                    episode_done = False
                    if sander_active:
                        raw_res = self.agent.select_action(state, evaluate=False)
                        rl_residual = self.limit_residual(raw_res)
                        episode_rl_active_steps += 1
                    else:
                        rl_residual = 0.0
                
                if not episode_done:
                    retry_count = 0
                    max_retries = 2
                    ok = False
                    while retry_count <= max_retries and not ok:
                        ok = self.comm.send_residual(rl_residual, timing_accurate, episode_done, False)  # learning_done=False
                        if not ok:
                            retry_count += 1
                            if retry_count <= max_retries:
                                self._log("DEBUG", f"송신 재시도 {retry_count}/{max_retries}")
                                time.sleep(self.cfg["COMM_RETRY_DELAY"])
                    if not ok:
                        self.fail_count += 1
                        self._log("WARNING", f"⚠️ 송신 실패 ({self.fail_count}/{self.FAIL_MAX})")
                        if self.fail_count >= self.FAIL_MAX:
                            self.comm.send_residual(0.0, False, True, False)  # episode_done=True, learning_done=False
                            self._log("WARNING", "통신 상태 악화 → 로봇PC에 PI 전용 대체 권고; 에피소드 종료.")
                            break
                    else:
                        self.fail_count = 0
                        episode_packets_sent += 1

                if episode_done:
                    success_flag = self.end_episode_fast_with_reliable_flag(ep, self.previous_target_force)
                    if success_flag:
                        break

                if (self.last_log_time is None or 
                    current_time - self.last_log_time >= 5.0):
                    mode = "RESIDUAL" if sander_active else "PI-ONLY"
                    force_achieved = " 🎯 TARGET ACHIEVED!" if abs(state[0] - state[1]) < 0.5 else ""
                    timing_status = "EXACT" if timing_accurate else "LATE"
                    self._log("INFO", f"[에피 {ep+1}] 단계 {self.episode_step} | {mode} | "
                          f"F {state[0]:.1f}/{state[1]:.1f}N | "
                          f"PI {state[5]:.3f}MPa | RL {rl_residual:.3f}MPa | {timing_status} | "
                          f"Time: {current_time - episode_start_time:.1f}s | "
                          f"RL_Flag: {sander_active}{force_achieved}")
                    self.last_log_time = current_time

                prev_state = state.copy()
                prev_action = rl_residual
                prev_sander_active = sander_active

            # ---- episode end ----
            episode_duration = time.perf_counter() - episode_start_time
            episode_stat = {
                "episode": ep + 1,
                "duration": episode_duration,
                "steps": self.episode_step,
                "reward": self.current_episode_reward,
                "packets_received": episode_packets_received,
                "packets_sent": episode_packets_sent,
                "rl_active_steps": episode_rl_active_steps,
                "rl_active_ratio": episode_rl_active_steps / self.episode_step if self.episode_step > 0 else 0
            }
            episode_stats.append(episode_stat)
            self.agent.episode_rewards.append(self.current_episode_reward)
            
            # ==== ADDED: 에피소드별 제어 지표 저장 ====
            self.cplogger.save_episode_metrics(ep + 1)
            self.cplogger.reset_episode_data()  # 다음 에피소드를 위해 데이터 초기화
            
            # ==== ADDED: 에피소드 종료 후 로봇 리셋 대기 ====
            # 모든 에피소드 완료 시 다음 에피소드 시작 대기 플래그 설정
            self._log("INFO", "🔄 로봇이 z축으로 이동하여 공압 툴 환경을 리셋하는 중...")
            self._log("INFO", "⏳ 다음 에피소드 시작을 위해 sander_active=1 대기 중...")
            # 다음 에피소드에서 sander_active=1을 감지할 수 있도록 플래그 설정
            self.waiting_for_episode_start = True
            
            if len(self.agent.episode_rewards) > self.agent.max_rewards_history:
                self.agent.episode_rewards = self.agent.episode_rewards[-self.agent.max_rewards_history:]
                self._log("DEBUG", f"보상 기록 정리: {self.agent.max_rewards_history}개로 제한")
            if self.current_episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.current_episode_reward
                self.best_agent_episode = ep
                self.agent.save_model(f"{self.cfg['MODEL_SAVE_DIR']}/test_best_agent_episode_{ep+1}_reward_{self.best_episode_reward:.2f}.pth")

            if (ep + 1) % 10 == 0:
                self._log("INFO", f"\n🎯 === 에피소드 {ep+1}/10 완료 ===")
                self._log("INFO", f"⏱️  지속 시간: {episode_duration:.1f}s")
                self._log("INFO", f"📊 단계: {self.episode_step:,}/{self.max_episode_steps:,} ({self.episode_step/self.max_episode_steps*100:.1f}%)")
                self._log("INFO", f"🏆 보상: {self.current_episode_reward:.2f}")
                self._log("INFO", f"📥 수신 패킷: {episode_packets_received}")
                self._log("INFO", f"📤 송신 패킷: {episode_packets_sent}")
                self._log("INFO", f"🤖 RL 활성 단계: {episode_rl_active_steps} ({episode_stat['rl_active_ratio']*100:.1f}%)")
                self._log("INFO", f"📈 지금까지 최고: {self.best_episode_reward:.2f}")
                if self.episode_step >= self.max_episode_steps:
                    self._log("INFO", f"✅ 완료: 최대 에피소드 단계 도달 ({self.max_episode_steps:,})")
                else:
                    self._log("WARNING", "⚠️  완료: 에피소드 조기 종료 (안전 또는 오류)")
                self._log("INFO", "=" * 40)
            else:
                self._log("INFO", f"🎯 에피소드 {ep+1} 완료 - 보상: {self.current_episode_reward:.2f}, 최고: {self.best_episode_reward:.2f}")

            if (ep + 1) % 10 == 0:
                self.comm.print_communication_stats()

            # ==== REMOVED: 50 에피소드마다 자동 저장 제거 ====
            # self.rlogger.flush_if_needed(ep + 1)  # Ctrl+C나 완료 시에만 저장

        # ==== ADDED: 모든 데이터 최종 저장 ====
        DataSaver.save_all_data(self, episodes, force=True)
        

        self._log("INFO", "\n🎯 최적화된 학습 완료!")
        self._log("INFO", f"✅ {episodes}개 에피소드 성공적으로 완료")
        self._log("INFO", f"🏆 최고 에피소드: {self.best_agent_episode+1}, 최고 보상: {self.best_episode_reward:.2f}")
        
        # ==== ADDED: 강화학습 완료 신호 전송 ====
        self._log("INFO", "📡 강화학습 완료 신호 전송 중...")
        success = self.comm.send_residual(0.0, True, False, True)  # learning_done=True
        if success:
            self._log("INFO", "✅ 강화학습 완료 신호 전송 성공")
        else:
            self._log("WARNING", "⚠️ 강화학습 완료 신호 전송 실패")
        
        self._log("INFO", "\n📊 === 최종 학습 요약 ===")
        total_duration = sum(ep["duration"] for ep in episode_stats)
        avg_reward = np.mean([ep["reward"] for ep in episode_stats])
        self._log("INFO", f"⏱️ 총 지속 시간: {total_duration:.1f}s")
        self._log("INFO", f"📊 평균 보상: {avg_reward:.2f}")
        self._log("INFO", f"📈 최고 보상: {self.best_episode_reward:.2f}")
        self._log("INFO", f"🤖 총 RL 활성 단계: {self.total_rl_active_steps}")
        self.comm.print_communication_stats()
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
                success = env.comm.send_residual(0.0, True, False, True)  # learning_done=True
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
            try:
                print("📁 Learning Done 폴더에 파일들 복사 중...")
                env.ldlogger.copy_episode_rewards(env.agent.episode_rewards, env.rlogger.log_dir)
                env.ldlogger.copy_reward_breakdown(env.rlogger.log_dir)
                print("✅ Learning Done 폴더 복사 완료!")
            except Exception as e:
                print(f"⚠️ Learning Done 폴더 복사 실패: {e}")
            
            
            Logger.log("INFO", "✅ 데이터 저장 완료!")
        except Exception as e:
            Logger.log("ERROR", f"❌ 데이터 저장 실패: {e}")
    sys.exit(0)

# =========================
# Main
# =========================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    SEND_FREQUENCY_HZ = 100
    RECV_FREQUENCY_HZ = 1000
    config = create_config(SEND_FREQUENCY_HZ, RECV_FREQUENCY_HZ)
    print("🚀 TEST VERSION 9: JY_Pneumatic_SAC_Pre_only_test_9_main.py")
    print(f"⚡ 송신 주파수: {SEND_FREQUENCY_HZ}Hz (간격: {config['TICK_SEC']:.3f}초)")
    print(f"📡 수신 주파수: {RECV_FREQUENCY_HZ}Hz (간격: {config['RECV_INTERVAL_SEC']:.3f}초)")
    print("=" * 60)
    np.random.seed(42) 
    torch.manual_seed(42)
    random.seed(42)
    env = PneumaticPolishingEnvironment(config)
    try:
        print(f"🚀 Starting optimized training for {SEND_FREQUENCY_HZ}Hz stable performance...")
        env.run_training(config["EPISODES"])
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
        Logger.log("WARNING", "Interrupted by user.")
        # ==== ADDED: KeyboardInterrupt 시 learning_done=True 전송 ====
        try:
            print("📡 강화학습 중단 신호 전송 중...")
            success = env.comm.send_residual(0.0, True, False, True)  # learning_done=True
            if success:
                print("✅ 강화학습 중단 신호 전송 성공")
            else:
                print("⚠️ 강화학습 중단 신호 전송 실패")
        except Exception as e:
            print(f"⚠️ 강화학습 중단 신호 전송 오류: {e}")
        
        # ==== ADDED: 모든 데이터 저장 ====
        DataSaver.save_all_data(env)
        
        env.comm.close()
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        # ==== ADDED: 예외 발생 시 learning_done=True 전송 ====
        try:
            print("📡 강화학습 오류 종료 신호 전송 중...")
            success = env.comm.send_residual(0.0, True, False, True)  # learning_done=True
            if success:
                print("✅ 강화학습 오류 종료 신호 전송 성공")
            else:
                print("⚠️ 강화학습 오류 종료 신호 전송 실패")
        except Exception as e2:
            print(f"⚠️ 강화학습 오류 종료 신호 전송 오류: {e2}")
        
        # ==== ADDED: 모든 데이터 저장 ====
        DataSaver.save_all_data(env)
        
        env.comm.close()
    finally:
        Logger.log("INFO", "🔚 Training program terminated.")
