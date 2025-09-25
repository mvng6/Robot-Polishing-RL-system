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
# =========================
# CONFIG
# =========================
# 기본 설정
_BASE_CONFIG = {
    # Neural Network
    "STATE_DIM": 6,
    "ACTION_DIM": 1,
    "HIDDEN": 256,
    "LR": 1e-3,
    "GAMMA": 0.99,
    "TAU": 0.005,
    "AUTO_ENTROPY": True,
    # Residual limits (MPa)
    "R_MIN": -0.2,
    "R_MAX":  0.2,
    "R_SLEW_PER_40MS": 0.048,
    # Scheduling - 송신/수신 주파수 설정
    "SEND_FREQ_HZ": 100,  # 송신 주파수 (Hz)
    "RECV_FREQ_HZ": 1000,  # 수신 주파수 (Hz) - 로봇 제어 PC에서 받는 주파수
    "TICK_TOL": 0.02,  # 타이밍 허용 오차 (초)
    # Training
    "BATCH_SIZE": 128,
    "REPLAY_WARMUP": 20,
    # Networking
    "HOST": "0.0.0.0",
    "PORT": 8888,
    "RECV_TIMEOUT_SEC": 0.5,
    "RECV_LOOP_TIMEOUT_SEC": 0.05,  # 수신 루프용 50ms 타임아웃
    "COMM_FAIL_MAX": 3,
    "COMM_RETRY_DELAY": 0.1,  # 통신 재시도 지연 (초)
    "MAX_STALE_DATA_TIME": 2.0,  # 최대 데이터 지연 허용 시간 (2초)
    # Episode
    "EPISODES": 100,
    "MAX_EPISODE_STEPS": 3000,
    # Safety / Reward shaping
    "MAX_FORCE_ERR": 15.0,
    "MAX_PRESS_DELTA": 0.05,
    # Model saving
    "MODEL_SAVE_DIR": "/home/katech/Robot-Polishing-RL-system/py_rl/saved_agents",
    # Logging paths
    "LOG_DIR": "/home/katech/Robot-Polishing-RL-system/py_rl/experiment_logs",
    # Memory management
    "MAX_EPISODE_REWARDS_HISTORY": 1000,  # 최대 보상 기록 수
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
    
    # 논리적 검증: 수신 주파수가 송신 주파수보다 높거나 같아야 함
    if config["RECV_FREQ_HZ"] < config["SEND_FREQ_HZ"]:
        raise ValueError(f"수신 주파수({config['RECV_FREQ_HZ']}Hz)는 송신 주파수({config['SEND_FREQ_HZ']}Hz)보다 높거나 같아야 합니다")
    
    # TICK_SEC 자동 계산 (송신 주파수 기준)
    config["TICK_SEC"] = 1.0 / config["SEND_FREQ_HZ"]
    
    # 수신 간격 자동 계산
    config["RECV_INTERVAL_SEC"] = 1.0 / config["RECV_FREQ_HZ"]
    
    return config

# 기본 CONFIG 생성
CONFIG = create_config()
# =========================
# Utils: seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
    def __init__(self, capacity=100000): 
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
# =========================
# Residual SAC Agent
# =========================   
class ResidualSACAgent:
    def __init__(self, cfg=None):
        if cfg is None:
            raise ValueError("cfg 파라미터가 필요합니다")
        self.cfg = cfg
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 차원 설정
        s_dim, a_dim, hidden = cfg["STATE_DIM"], cfg["ACTION_DIM"], cfg["HIDDEN"]
        # 하이퍼파라미터 설정
        self.gamma, self.tau = cfg["GAMMA"], cfg["TAU"]
        # 엔트로피 설정
        self.alpha = 0.05 # 엔트로피 계수: 탐험-활용 균형 조절
        self.auto_entropy_tuning = cfg["AUTO_ENTROPY"]
        # 신경망 생성
        self.actor = Actor(s_dim, a_dim, hidden).to(self.device)
        self.critic = Critic(s_dim, a_dim, hidden).to(self.device)
        self.critic_target = Critic(s_dim, a_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # 옵티마이저 설정
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg["LR"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg["LR"])
        # 자동 엔트로피 튜닝
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.tensor([a_dim], device=self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg["LR"])
        # 기타 설정 초기화
        self.replay = ReplayBuffer()
        self.total_steps = 0
        self.episode_rewards = []
        self.max_rewards_history = cfg.get("MAX_EPISODE_REWARDS_HISTORY", 1000)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device) # 1차원 배열 -> 2차원 배열
        with torch.no_grad(): # 평가 모드일 때, 아닐 때 각각 출력 다르게
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, log_prob = self.actor.sample(state)
        action = action.cpu().numpy().flatten()
        # Residual 압력 범위 MPa [-0.2, 0.2]
        return float(action[0] * (self.cfg["R_MAX"]))
    
    def store_transition(self, state, action, reward, next_state, done):
        norm_action = action / self.cfg["R_MAX"] # 행동 정규화 [-0.2 ~ 0.2] -> [-1 ~ 1]
        self.replay.push(state, norm_action, reward, next_state, done) # 경험 리플레이에 저장

    def update_parameters(self, batch_size=None):
        bs = batch_size or self.cfg["BATCH_SIZE"]
        if len(self.replay) < bs: return

        s, a, r, ns, d = self.replay.sample(bs) # 리플레이 버퍼에서 랜덤하게 배치 샘플링
        s = torch.FloatTensor(s).to(self.device) # 텐서 변환
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device) # 보상 2차원으로 변환
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device) # 종료 2차원으로 변환

        with torch.no_grad(): # 타겟 계산 시 그래디언트 계산 X
            na, nlogp = self.actor.sample(ns) # 다음 상태에서 행동과 로그 확률
            q1n, q2n = self.critic_target(ns, na) # 두 개의 타겟 크리틱 네트워크 출력
            min_qn = torch.min(q1n, q2n) - self.alpha * nlogp # 더 작은 Q값 선택 + 엔트로피 보너스
            y = r + (1 - d) * self.gamma * min_qn # 타겟 Q값 (벨만 방정식)
        # 크리틱 업데이트    
        q1, q2 = self.critic(s, a) # q1, q2 현재 크리틱 네트워크 Q값
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y) # MSE 손실 (예측값과 타겟값 차이)
        self.critic_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) # 그래디언트 클리핑
        self.critic_opt.step() # 옵티마이저로 가중치 업데이트
        # 액터 업데이트
        pi, logp = self.actor.sample(s) # 현재 상태에서 행동과 로그 확률 샘플링
        q1_pi, q2_pi = self.critic(s, pi) # q1_pi, q2_pi 현재 정책에 따른 Q값
        min_q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = ((self.alpha * logp) - min_q_pi).mean() # 정책 손실 (엔트로피 - Q값)
        self.actor_opt.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        if self.auto_entropy_tuning:
            logp_entropy = logp.squeeze(1)
            a_loss = -(self.log_alpha * (logp_entropy + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); a_loss.backward(); self.alpha_opt.step()
            self.alpha = self.log_alpha.exp()

        # 소프트 업데이트 - 타겟 네트워크를 천천히 업데이트
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
class ResidualRLCommunicator: # TCP 소켓 통신 클래스
    def __init__(self, host, port, recv_timeout, recv_loop_timeout=0.05, cfg=None):
        self.host, self.port = host, port 
        self.recv_timeout = recv_timeout # 수신 타임아웃 설정
        self.recv_loop_timeout = recv_loop_timeout # 수신 루프 타임아웃 설정
        self.cfg = cfg
        self.socket = None
        self.conn = None
        self.connected = False
        # 패킷 포맷 정의
        # c++ -> python: unsigned short + 6 floats + 1bool + unsigned short = 29바이트
        self.CPP_TO_PY_PACKET_FORMAT = ">HffffffBH"
        self.CPP_TO_PY_PACKET_SIZE = 29
        self.CPP_TO_PY_SOF = 0xAAAA
        # python -> c++: unsigned short + float + bool + bool + unsigned short = 10바이트
        self.PY_TO_CPP_PACKET_FORMAT = ">HfBBH"
        self.PY_TO_CPP_PACKET_SIZE = 10
        self.PY_TO_CPP_SOF = 0xBBBB
        # 상태 및 통계 변수
        self.latest_state = None # 최신 상태 데이터
        self.latest_sander_active = False # 샌더 활성화 상태
        self.receive_thread = None # 수신 전용 스레드
        self.is_receiving = False # 수신 중인지 여부
        self.state_lock = threading.Lock()  # 스레드 안전성
        self.stats_lock = threading.Lock()  # 통계 변수 보호용
        self.packets_received = 0
        self.packets_sent = 0
        self.connection_start_time = None
        self.last_packet_time = None  # 마지막 수신 시간 추적
        self.consecutive_failures = 0  # 연속 실패 카운트

    def _log(self, level, message):
        """통합된 로깅 함수"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3] # 밀리초까지
        level_icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍"
        }
        icon = level_icons.get(level, "ℹ️")
        print(f"[{timestamp}] {icon} {message}")

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP 소켓 생성
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 포트 재사용 허용
            self.socket.bind((self.host, self.port)) # 호스트와 포트 바인딩
            self.socket.listen(1) # 최대 1개의 연결 대기
            self.socket.settimeout(1.0) # 1초 타임아웃 설정
            self._log("INFO", f"로봇제어PC 연결 대기 중 {self.host}:{self.port} ...")
            
            while True:
                try:
                    conn, addr = self.socket.accept() # 연결 대기
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
        """동적 주파수로 계속 수신하는 별도 쓰레드 시작"""
        self.is_receiving = True # 수신 상태 플래그
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True) # 별도 스레드에서 수신처리, 메인 프로그램 종료 시 함께 종료
        self.receive_thread.start()
        self._log("INFO", f"{self.cfg['RECV_FREQ_HZ']}Hz 수신 스레드 시작")

    def _receive_loop(self):
        """동적 주파수로 계속 수신하는 메인 루프 - 타이밍 기반 수신"""
        next_receive_time = time.perf_counter()
        recv_interval = self.cfg["RECV_INTERVAL_SEC"]  # 동적 수신 간격
        
        # 메인 수신 루프
        while self.is_receiving: # 수신 중인지 확인
            current_time = time.perf_counter() # 현재 시간 측정
            
            # 동적 주파수로 수신 시도
            if current_time >= next_receive_time:
                next_receive_time += recv_interval  # 동적 간격
                
                try:
                    # 타임아웃을 50ms로 설정 (안정적인 수신)
                    self.conn.settimeout(self.recv_loop_timeout) # 50ms 타임아웃
                    data = self._recv_exact(self.CPP_TO_PY_PACKET_SIZE) # 정확한 크기의 패킷 수신
                    if data:
                        state, sander_active = self._process_packet(data)
                        if state is not None:
                            with self.state_lock:
                                self.latest_state = state
                                self.latest_sander_active = sander_active
                                self.last_packet_time = time.perf_counter()
                            # 성공 시 연속 실패 카운트 리셋
                            self.consecutive_failures = 0
                except socket.timeout:
                    # 50ms 내에 데이터가 없으면 다음 주기까지 대기
                    pass
                except Exception as e:
                    self.consecutive_failures += 1
                    self._log("WARNING", f"수신 루프 오류 ({self.consecutive_failures}회): {e}")
                    if self.consecutive_failures >= 5:  # 5회 연속 실패 시 중단
                        self._log("ERROR", "연속 수신 실패로 수신 루프 중단")
                        break
                    time.sleep(self.cfg["RECV_INTERVAL_SEC"])  # 잠시 대기 후 재시도
            else:
                # 다음 수신 시간까지 대기 (CPU 효율성 개선)
                time.sleep(0.001)  # 1ms 대기
                
        self._log("INFO", "수신 루프 종료")

    def _recv_exact(self, nbytes):
        """정확히 n바이트를 받을 때까지 반복 수신"""
        data = b''
        while len(data) < nbytes:
            chunk = self.conn.recv(nbytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def _process_packet(self, data):
        """수신된 패킷을 처리하여 상태와 플래그 반환"""
        try:
            # 1. 길이 검증 - 29바이트가 맞는지
            if len(data) != self.CPP_TO_PY_PACKET_SIZE:
                self._log("WARNING", f"예상 {self.CPP_TO_PY_PACKET_SIZE}B, 수신 {len(data)}B")
                return None, False
            # 2. 패킷 언패킹 
            try:
                (sof, current_force, target_force, force_error, force_error_dot, 
                 force_error_int, pi_output, sander_active, 
                 received_checksum) = struct.unpack(">HffffffBH", data) # 바이트 데이터를 파이썬 변수로 변환
            except struct.error as e:
                self._log("ERROR", f"패킷 언팩 실패: {e}")
                return None, False
            # 3. SOF 검증
            if sof != self.CPP_TO_PY_SOF:
                self._log("WARNING", f"SOF 불일치: {hex(sof)} (예상: {hex(self.CPP_TO_PY_SOF)})")
                return None, False
            # 4. 체크섬 검증 (CRC-16)
            calculated_checksum = self.calculate_crc16(data[:-2]) # checksum 제외한 부분으로 계산
            if received_checksum != calculated_checksum:
                self._log("ERROR", f"체크섬 오류: 수신:{received_checksum} 계산:{calculated_checksum}")
                return None, False
            # 5. 상태 배열 구성 - 6개 변수로 수정
            state = np.array([
                current_force,      # 0: RL_currentForceZ
                target_force,       # 1: RL_targetForceZ  
                force_error,        # 2: RL_forceZError
                force_error_dot,    # 3: RL_forceZErrordot
                force_error_int,    # 4: RL_forceZErrorintegral
                pi_output,          # 5: RL_pidFlag (float)
            ], dtype=np.float32)
            sander_active = bool(sander_active)
            # 6. 수신 성공 통계 업데이트 (스레드 안전)
            with self.stats_lock:
                self.packets_received += 1
            return state, sander_active
        except Exception as e:
            self._log("ERROR", f"패킷 처리 오류: {e}")
            return None, False
        
    def calculate_crc16(self, data: bytes) -> int:
        """CRC-16/MODBUS 체크섬 계산 (C++와 동일)"""
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
        """메인 루프에서 호출하여 최신 상태 반환 (non-blocking)"""
        with self.state_lock:
            if self.latest_state is not None:
                # 데이터 유효성 검증 (너무 오래된 데이터 체크)
                current_time = time.perf_counter()
                if (self.last_packet_time and 
                    current_time - self.last_packet_time > 2.0):  # 2초 이상 오래된 데이터
                    self._log("WARNING", f"오래된 데이터 감지: {current_time - self.last_packet_time:.2f}초 전")
                
                # sander_active 상태 디버깅을 위한 로깅 추가
                if hasattr(self, 'last_logged_sander_active') and self.last_logged_sander_active != self.latest_sander_active:
                    self._log("DEBUG", f"RL 플래그 변경: {self.last_logged_sander_active} -> {self.latest_sander_active}")
                    self.last_logged_sander_active = self.latest_sander_active
                elif not hasattr(self, 'last_logged_sander_active'):
                    self.last_logged_sander_active = self.latest_sander_active
                    self._log("DEBUG", f"초기 RL 플래그: {self.latest_sander_active}")
                return self.latest_state.copy(), self.latest_sander_active
        return None, False
    
    def send_residual(self, rl_residual, timing_accurate, episode_done):
        """Residual 압력 전송"""
        try:
            # 1. 10바이트 패킷 구성
            # SOF(2) + rl_residual(4) + timing_accurate(1) + episode_done(1) + checksum(2)
            # 2. 체크섬 계산용 데이터 (checksum 제외 부분)
            data_part = struct.pack(">HfBB", 
                                  self.PY_TO_CPP_SOF, 
                                  float(rl_residual), 
                                  bool(timing_accurate), 
                                  bool(episode_done))
            checksum = self.calculate_crc16(data_part)
            # 3. 최종 패킷 (SOF, float, unsigned char, unsigned char, checksum[uint16])
            final_packet = struct.pack(self.PY_TO_CPP_PACKET_FORMAT, 
                                     self.PY_TO_CPP_SOF, 
                                     float(rl_residual), 
                                     bool(timing_accurate), 
                                     bool(episode_done), 
                                     checksum)
            
            # 4. 송신
            self.conn.sendall(final_packet)
            
            # 5. 송신 성공 통계 업데이트 (스레드 안전)
            with self.stats_lock:
                self.packets_sent += 1
            return True
        except Exception as e:
            self._log("ERROR", f"residual 전송 오류: {e}")
            return False
        
    def send_reset(self):
        """에피소드 리셋 명령 전송 (binary protocol)"""
        try:
            # 🎯 binary protocol로 reset 신호 전송
            # SOF(2) + reset_flag(1) + padding(3) + checksum(2) = 8 bytes
            reset_data = struct.pack(">HBxxxH", 0xCCCC, 1, 0)  # 0xCCCC = reset SOF
            checksum = self.calculate_crc16(reset_data[:-2])
            reset_packet = struct.pack(">HBxxxH", 0xCCCC, 1, checksum)
            self.conn.sendall(reset_packet)
            return True
        except Exception as e:
            self._log("ERROR", f"리셋 전송 오류: {e}")
            return False
        
    def get_communication_stats(self):
        """통신 통계 반환 - 간소화된 버전 (스레드 안전)"""
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
        """통신 통계 출력 - 간소화된 버전"""
        stats = self.get_communication_stats()
        self._log("INFO", "\n📊 === 통신 통계 ===")
        self._log("INFO", f"⏱️  가동 시간: {stats['uptime_seconds']:.1f}s")
        self._log("INFO", f"📥 수신된 패킷: {stats['packets_received']}")
        self._log("INFO", f"📤 송신된 패킷: {stats['packets_sent']}")
        self._log("INFO", f"📥 수신률: {stats['receive_rate_hz']:.1f} Hz")
        self._log("INFO", f"📤 송신률: {stats['send_rate_hz']:.1f} Hz")
        self._log("INFO", "=" * 40)

    def close(self):
        """연결 종료 및 리소스 정리"""
        try:
            # 수신 쓰레드 종료
            self.is_receiving = False
            if self.receive_thread and self.receive_thread.is_alive():
                self.receive_thread.join(timeout=1.0)
            # 소켓 종료
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
        # scheduler
        self.last_tick = None
        # residual limiter
        self.prev_residual = 0.0
        # episode stats
        self.episode_step = 0
        self.max_episode_steps = cfg["MAX_EPISODE_STEPS"]
        self.current_episode_reward = 0.0
        self.best_episode_reward = -float("inf")
        self.best_agent_episode = -1
        # RL activity monitor
        self.rl_inactive_count = 0
        self.max_rl_inactive_steps = 250
        self.rl_active_in_episode = False
        self.total_rl_active_steps = 0
        # comm fail
        self.fail_count = 0
        self.FAIL_MAX = cfg["COMM_FAIL_MAX"]
        self.last_log_time = None
        self.previous_target_force = 0.0
        # 마지막 유효한 상태 저장 (송신 연속성 보장)
        self.last_valid_state = None
        self.last_sander_active = False
        # Target achieved 추적 변수
        self.target_achieved_start_time = None
        self.target_achieved_duration = 0.0
        self.target_achieved_threshold = 0.5  # ±0.5N 이내
        self.target_achieved_required_duration = 10.0  # 10초간 유지
        
    def _log(self, level, message):
        """통합된 로깅 함수 - PneumaticPolishingEnvironment 클래스용"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍"
        }
        icon = level_icons.get(level, "ℹ️")
        print(f"[{timestamp}] {icon} {message}")

    def generate_episode_reward_graph(self):
        """에피소드별 보상 그래프 생성 (Ctrl+C 시 호출)"""
        if not hasattr(self, 'agent') or not self.agent.episode_rewards:
            self._log("WARNING", "생성할 보상 데이터가 없습니다")
            return
            
        try:
            import matplotlib.pyplot as plt
            
            # 에피소드별 보상 (누적 보상이 아닌 각 에피소드의 보상)
            episode_rewards = self.agent.episode_rewards
            episodes = list(range(1, len(episode_rewards) + 1))
            
            # 그래프 생성
            plt.figure(figsize=(12, 6))
            plt.plot(episodes, episode_rewards, 'b-', linewidth=2, marker='o', markersize=4)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Episode Reward', fontsize=12)
            plt.title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # 평균선 추가
            if len(episode_rewards) > 1:
                avg_reward = np.mean(episode_rewards)
                plt.axhline(y=avg_reward, color='r', linestyle='--', alpha=0.7, 
                           label=f'Average: {avg_reward:.2f}')
                plt.legend()
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.cfg['LOG_DIR']}/episode_rewards_{timestamp}.png"
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 그래프 저장
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log("INFO", f"📈 에피소드별 보상 그래프 저장: {filename}")
            
        except Exception as e:
            self._log("ERROR", f"에피소드별 보상 그래프 생성 오류: {e}")
    
    # ---- scheduler ----
    def should_send_now(self):
        """정확한 타이밍 제어"""
        now = time.perf_counter()
        if self.last_tick is None:
            self.last_tick = now
            return True, True
        dt = now - self.last_tick
        if dt >= self.cfg["TICK_SEC"]:
            is_exact = abs(dt - self.cfg["TICK_SEC"]) <= self.cfg["TICK_TOL"]
            self.last_tick = now
            return True, is_exact
        return False, False
    # ---- residual limiter ----
    def limit_residual(self, r):
        r = float(np.clip(r, self.cfg["R_MIN"], self.cfg["R_MAX"]))
        delta = np.clip(r - self.prev_residual, -self.cfg["R_SLEW_PER_40MS"], self.cfg["R_SLEW_PER_40MS"])
        r_limited = self.prev_residual + delta
        self.prev_residual = r_limited
        return r_limited
    
    ############################
    ## 지수적 보상함수 (Exponential Reward)
    ############################
    def calculate_reward_exponential(self, state, action_residual, sander_active):
        """
        Course 2 - Stage 1: 지수적 보상 + 스케일 정규화 + 기본 안전성 개선
        
        개선사항:
        1. 스케일 정규화: 각 구성요소를 0~1 범위로 정규화
        2. 가중치 균형: 구성요소 간 균형 조정  
        3. 부드러운 안전성: 선형 페널티로 개선
        4. 연속적 커리큘럼: 급작스러운 보상 변화 제거
        5. 학습 안정화: 보상 클리핑 및 균형잡힌 가중치
        """
        current_force, target_force = state[0], state[1]
        force_error, force_error_dot = state[2], state[3]
        
        force_err = abs(force_error)
        residual_change = abs(action_residual - self.prev_residual)
        
        # === 개선된 지수적 보상 구조 (0~1 범위로 정규화) ===
        
        # 1) 추적 보상 (0~1 범위로 정규화) - 연속적 목표 유도
        tracking_reward = np.exp(-force_err * 1.5)  # 0~1 범위
        
        # 2) 안정성 보상 (0~1 범위로 정규화) - 진동 억제
        stability_reward = np.exp(-abs(force_error_dot) * 2.0)  # 0~1 범위
        
        # 3) 부드러움 보상 (0~1 범위로 정규화) - 급격한 변화 방지
        smoothness_reward = np.exp(-residual_change * 50.0)  # 0~1 범위
        
        # 4) 근접 보너스 (연속적으로 개선) - 급작스러운 변화 제거
        if force_err <= 0.5:
            proximity_bonus = 1.5 * (1 - force_err / 0.5)  # 0~1.5 범위
        else:
            proximity_bonus = 0.0
        
        # === 균형 잡힌 가중치 적용 ===
        # 추적(3.0) > 안정성(1.5) > 근접(1.5) > 부드러움(0.5)
        base_reward = (3.0 * tracking_reward + 
                      1.5 * stability_reward + 
                      0.5 * smoothness_reward + 
                      proximity_bonus)
        
        # === 개선된 안전성 페널티 (부드러운 선형) ===
        if current_force > 80.0:
            safety_penalty = -2.0 * (current_force - 80.0) / 20.0  # 선형 페널티 (-2.0~0)
        else:
            safety_penalty = 0.0
        
        # === 정확도 페널티 (Target_force ±10N 오차 초과 시 강한 페널티) ===
        if force_err > 10.0:
            # 10N 초과 시 더 강한 페널티: -10.0 * (force_err - 10.0) / 10.0
            accuracy_penalty = -10.0 * (force_err - 10.0) / 10.0  # 매우 강한 선형 페널티 (-10.0~0)
        else:
            accuracy_penalty = 0.0
        
        # === 효율성 페널티 (감소) ===
        efficiency_penalty = -0.02 * abs(action_residual)  # 기존 -0.05 → -0.02로 완화
        
        total_reward = base_reward + safety_penalty + accuracy_penalty + efficiency_penalty
        
        # === 보상 클리핑 (학습 안정화) - 강화된 정확도 페널티 고려하여 범위 확장 ===
        total_reward = np.clip(total_reward, -25.0, 8.0)
        
        return float(total_reward)

    def is_done(self, state):
        # 🎯 최대 스텝에 도달했을 때 True
        if self.episode_step >= self.max_episode_steps: 
            return True
        # 🚨 안전장치: 접촉력이 과도하게 높을 때만 종료
        if state[0] > 100.0: 
            self._log("WARNING", f"안전: 힘이 너무 높음 ({state[0]:.1f}N > 100N) - 에피소드 종료")
            return True
        # 🎯 Target_force ±20N 오차 초과 시 강한 페널티와 함께 에피소드 종료
        current_force, target_force = state[0], state[1]
        force_error = abs(current_force - target_force)
        if force_error > 20.0:
            self._log("WARNING", f"정확도: 힘 오차가 너무 큼 ({force_error:.1f}N > 20N) - 에피소드 종료")
            return True
        # 🎯 Target achieved 조건: ±0.5N 이내에서 10초간 유지되면 에피소드 종료
        if force_error <= self.target_achieved_threshold:
            current_time = time.perf_counter()
            if self.target_achieved_start_time is None:
                self.target_achieved_start_time = current_time
            else:
                self.target_achieved_duration = current_time - self.target_achieved_start_time
                if self.target_achieved_duration >= self.target_achieved_required_duration:
                    self._log("SUCCESS", f"🎯 Target achieved 완료! ({self.target_achieved_duration:.1f}s ≥ {self.target_achieved_required_duration}s) - 에피소드 종료")
                    return True
        else:
            # Target achieved 조건이 깨지면 리셋
            if self.target_achieved_start_time is not None:
                self.target_achieved_start_time = None
                self.target_achieved_duration = 0.0
        # ✅ 다른 모든 경우: 계속 진행
        return False

    # ---- RL activity monitor ----
    def check_rl_status(self, sander_active):
        if sander_active:
            self.rl_inactive_count = 0
            self.rl_active_in_episode = True
            self.total_rl_active_steps += 1
            return "active"
        else:
            self.rl_inactive_count += 1
            if self.rl_inactive_count >= self.max_rl_inactive_steps:
                self._log("WARNING", "RL 비활성 너무 오래 → 에피소드 종료")
                return "terminate"
            return "inactive"
    # ---- episode helpers ----
    def end_episode_fast_with_reliable_flag(self, ep, target_force):
        """한 번만 episode_done 신호 전송으로 에피소드 종료"""
        self._log("INFO", f"🎯 에피소드 {ep+1} 완료 (단계 {self.episode_step})")
        # self._log("INFO", f"🚀 episode_done 신호 한 번 전송")
        
        # 🚀 핵심 수정: episode_done을 딱 한 번만 전송
        success = self.comm.send_residual(0.0, True, True)
        
        if success:
            self._log("INFO", f"📡 Episode done 신호 전송 성공")
        else:
            self._log("WARNING", f"⚠️ Episode done 신호 전송 실패")
        
        # 🎯 확인 차 최대 500ms만 대기
        confirmation_start = time.perf_counter()
        max_confirmation_time = 0.5  # 500ms만 대기
        
        while (time.perf_counter() - confirmation_start) < max_confirmation_time:
            state, _ = self.comm.get_latest_state()
            if state is not None and abs(state[1] - target_force) > 1.0:
                elapsed = time.perf_counter() - confirmation_start
                self._log("INFO", f"✅ 에피소드 종료 확인! {target_force:.1f}N → {state[1]:.1f}N ({elapsed:.3f}s 소요)")
                break
            time.sleep(self.cfg["RECV_INTERVAL_SEC"])  # 동적 간격으로 확인
        else:
            # 500ms 후에도 확인 안되면 강제 성공
            self._log("INFO", f"⚡ Episode done 신호 전송 후 에피소드 종료 (500ms 타임아웃) - 다음 에피소드로 진행")
        
        # 🎯 마지막으로 episode_done=False 신호 한 번 전송
        success = self.comm.send_residual(0.0, True, False)
        
        return True

    def reset_episode(self):
        self.prev_residual = 0.0
        self.episode_step = 0
        self.current_episode_reward = 0.0
        self.rl_inactive_count = 0
        self.rl_active_in_episode = False
        self.last_tick = None
        self.last_log_time = None
        # 마지막 유효한 상태 초기화
        self.last_valid_state = None
        self.last_sander_active = False
        # Target achieved 추적 변수 초기화
        self.target_achieved_start_time = None
        self.target_achieved_duration = 0.0
        state, _ = self.comm.get_latest_state()
        if state is not None:
            self.previous_target_force = state[1]  # target_force
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
        # saved_agents 폴더 생성
        model_save_dir = self.cfg["MODEL_SAVE_DIR"]
        os.makedirs(model_save_dir, exist_ok=True)
        self._log("INFO", f"📁 모델 저장 디렉토리: {model_save_dir}")
        self._log("INFO", "🚀 최적화된 Residual RL 학습 시작 - 버전 7")
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
                print(f"⏳ Waiting... Current Force: {current_force:.1f}N, Target: {target_force:.1f}N", end='\r')
            time.sleep(0.04)
            if time.perf_counter() - wait_start_time > 300:
                self._log("WARNING", "\n⚠️ RL 활성화 타임아웃 (5분)")
                return
        episode_stats = []
        for ep in range(episodes):
            self._log("INFO", f"\n🎬 === 에피소드 {ep+1}/{episodes} 시작 ===")
            # 🎯 에피소드 시작 시 RL Flag 상태 확인
            episode_start_state, episode_start_sander_active = self.comm.get_latest_state()
            if not episode_start_sander_active:
                self._log("WARNING", f"⚠️ 경고: 에피소드 {ep+1} 시작 시 RL 플래그 False")
                self._log("INFO", "🔄 RL 활성화 대기 중...")
                wait_start = time.perf_counter()
                while not episode_start_sander_active:
                    episode_start_state, episode_start_sander_active = self.comm.get_latest_state()
                    if time.perf_counter() - wait_start > 60:  # 1분 타임아웃
                        self._log("WARNING", f"⚠️ 에피소드 {ep+1}에서 RL 활성화 대기 타임아웃")
                        break
                    time.sleep(0.1)
            episode_start_time = time.perf_counter()
            self.reset_episode()
            # 새 에피소드 시작 신호 전송
            self._log("INFO", f"📡 에피소드 {ep+1} 시작")
            self.comm.send_residual(0.0, True, False)
            prev_state = None
            prev_action = None
            prev_sander_active = False
            # 에피소드 통계
            episode_packets_received = 0
            episode_packets_sent = 0
            episode_rl_active_steps = 0
            # 타이밍 제어를 위한 변수 초기화
            episode_start_perf_time = time.perf_counter()
            tick_count = 0
            
            while True:
                current_time = time.perf_counter()
                
                # 절대 시간 기준으로 다음 전송 시간 계산 (누적 지연 방지)
                next_send_time = episode_start_perf_time + (tick_count + 1) * self.cfg["TICK_SEC"]
                
                # 타이밍이 되었을 때만 처리
                if current_time >= next_send_time:
                    tick_count += 1
                    timing_accurate = abs(current_time - next_send_time) <= self.cfg["TICK_TOL"]
                    
                    # 최신 상태 가져오기
                    res = self.comm.get_latest_state()
                    if res[0] is None:
                        # 데이터가 없어도 송신 계속 - 이전 유효한 상태 사용
                        if self.last_valid_state is not None:
                            state = self.last_valid_state.copy()
                            sander_active = self.last_sander_active
                            self._log("DEBUG", f"데이터 없음 - 이전 상태 사용 (step {self.episode_step})")
                        else:
                            # 기본값으로 초기화
                            state = np.array([0.0, -30.0, -30.0, 0.0, 0.0, 0.0], dtype=np.float32)
                            sander_active = False
                            self._log("DEBUG", f"데이터 없음 - 기본값 사용 (step {self.episode_step})")
                    else:
                        # 새로운 데이터가 있으면 저장
                        state, sander_active = res
                        self.last_valid_state = state.copy()
                        self.last_sander_active = sander_active
                        episode_packets_received += 1
                        self.previous_target_force = state[1]
                else:
                    # 🚀 최적화: 짧은 sleep으로 CPU 효율성 개선
                    time.sleep(0.001)  # 1ms sleep으로 CPU 부하 감소
                    continue
                # 기준으로 스텝 증가
                self.episode_step += 1
                
                # 현재 상태에 대한 종료 조건 확인 (항상 실행)
                done = self.is_done(state)
                
                # 이전 transition 처리 및 학습
                if prev_state is not None and prev_sander_active:
                    reward = self.calculate_reward_exponential(prev_state, prev_action, prev_sander_active)
                    self.agent.store_transition(prev_state, prev_action, reward, state, done)
                    self.current_episode_reward += reward
                    if len(self.agent.replay) > self.cfg["REPLAY_WARMUP"]:
                        self.agent.update_parameters(self.cfg["BATCH_SIZE"])
                
                # RL 활성 상태 확인
                rl_status = self.check_rl_status(sander_active)
                if rl_status == "terminate":
                    self._log("INFO", "RL 비활성 지속으로 에피소드 종료")
                    break
                
                # 에피소드 종료 확인 (is_done 결과 또는 최대 스텝 도달)
                if done or self.episode_step >= self.max_episode_steps:
                    episode_done = True
                    rl_residual = 0.0
                    if done:
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
                # residual 전송 (재시도 로직 포함)
                retry_count = 0
                max_retries = 2
                ok = False
                
                while retry_count <= max_retries and not ok:
                    ok = self.comm.send_residual(rl_residual, timing_accurate, episode_done)
                    if not ok:
                        retry_count += 1
                        if retry_count <= max_retries:
                            self._log("DEBUG", f"송신 재시도 {retry_count}/{max_retries}")
                            time.sleep(self.cfg["COMM_RETRY_DELAY"])
                
                if not ok:
                    self.fail_count += 1
                    self._log("WARNING", f"⚠️ 송신 실패 ({self.fail_count}/{self.FAIL_MAX})")
                    if self.fail_count >= self.FAIL_MAX:
                        # advise PI-only fallback by ending episode with residual=0
                        self.comm.send_residual(0.0, False, True)
                        self._log("WARNING", "통신 상태 악화 → 로봇PC에 PI 전용 대체 권고; 에피소드 종료.")
                        break
                else:
                    self.fail_count = 0
                    episode_packets_sent += 1
                # 🚀 최적화된 에피소드 종료 - 대폭 시간 단축
                if episode_done:
                    success = self.end_episode_fast_with_reliable_flag(ep, self.previous_target_force)
                    if success:
                        break
                # 2.5초마다 간결한 로깅
                if (self.last_log_time is None or 
                    current_time - self.last_log_time >= 2.5):
                    mode = "RESIDUAL" if sander_active else "PI-ONLY"
                    force_achieved = " 🎯 TARGET ACHIEVED!" if abs(state[0] - state[1]) < 0.5 else ""
                    timing_status = "EXACT" if timing_accurate else "LATE"
                    self._log("INFO", f"[에피 {ep+1}] 단계 {self.episode_step} | {mode} | "
                          f"F {state[0]:.1f}/{state[1]:.1f}N | "
                          f"PI {state[5]:.3f}MPa | RL {rl_residual:.3f}MPa | {timing_status} | "
                          f"Time: {current_time - episode_start_time:.1f}s | "
                          f"RL_Flag: {sander_active}{force_achieved}")
                    self.last_log_time = current_time
                # 다음 transition을 위해 저장
                prev_state = state.copy()
                prev_action = rl_residual
                prev_sander_active = sander_active
            # ---- episode end ----
            episode_duration = time.perf_counter() - episode_start_time
            # 에피소드 통계 저장
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
            
            # 메모리 관리: 보상 기록이 최대 한도를 초과하면 오래된 것 제거
            if len(self.agent.episode_rewards) > self.agent.max_rewards_history:
                self.agent.episode_rewards = self.agent.episode_rewards[-self.agent.max_rewards_history:]
                self._log("DEBUG", f"보상 기록 정리: {self.agent.max_rewards_history}개로 제한")
            if self.current_episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.current_episode_reward
                self.best_agent_episode = ep
                self.agent.save_model(f"{self.cfg['MODEL_SAVE_DIR']}/test_best_agent_episode_{ep+1}_reward_{self.best_episode_reward:.2f}.pth")
            # dynamic threshold saving (10 에피소드마다 최고 성능만 저장)
            if (ep + 1) % 10 == 0:
                # 최근 10개 에피소드 중 최고 성능 찾기
                recent_10 = self.agent.episode_rewards[-10:]
                best_recent_reward = max(recent_10)
                best_recent_episode = recent_10.index(best_recent_reward) + (ep + 1) - 9  # 실제 에피소드 번호
                
                # 최고 성능 모델 저장
                self.agent.save_model(f"{self.cfg['MODEL_SAVE_DIR']}/test_best_10ep_ep{best_recent_episode}_reward_{best_recent_reward:.2f}.pth")
                self._log("INFO", f"📈 최근 10 에피소드 최고 성능 저장: 에피소드 {best_recent_episode}, 보상 {best_recent_reward:.2f}")
            # 에피소드 완료 요약 (10 에피소드마다 상세 출력)
            if (ep + 1) % 10 == 0:
                self._log("INFO", f"\n🎯 === 에피소드 {ep+1}/10 완료 ===")
                self._log("INFO", f"⏱️  지속 시간: {episode_duration:.1f}s")
                self._log("INFO", f"📊 단계: {self.episode_step:,}/{self.max_episode_steps:,} ({self.episode_step/self.max_episode_steps*100:.1f}%)")
                self._log("INFO", f"🏆 보상: {self.current_episode_reward:.2f}")
                self._log("INFO", f"📥 수신 패킷: {episode_packets_received}")
                self._log("INFO", f"📤 송신 패킷: {episode_packets_sent}")
                self._log("INFO", f"🤖 RL 활성 단계: {episode_rl_active_steps} ({episode_stat['rl_active_ratio']*100:.1f}%)")
                self._log("INFO", f"📈 지금까지 최고: {self.best_episode_reward:.2f}")
                # 🎯 에피소드 완료 이유 표시
                if self.episode_step >= self.max_episode_steps:
                    self._log("INFO", f"✅ 완료: 최대 에피소드 단계 도달 ({self.max_episode_steps:,})")
                else:
                    self._log("WARNING", "⚠️  완료: 에피소드 조기 종료 (안전 또는 오류)")
                self._log("INFO", "=" * 40)
            else:
                # 간단한 에피소드 완료 로그
                self._log("INFO", f"🎯 에피소드 {ep+1} 완료 - 보상: {self.current_episode_reward:.2f}, 최고: {self.best_episode_reward:.2f}")
            # 통신 상태 모니터링 (10 에피소드마다)
            if (ep + 1) % 10 == 0:
                self.comm.print_communication_stats()
        # ---- 전체 학습 완료 ----
        self._log("INFO", "\n🎯 최적화된 학습 완료!")
        self._log("INFO", f"✅ {episodes}개 에피소드 성공적으로 완료")
        self._log("INFO", f"🏆 최고 에피소드: {self.best_agent_episode+1}, 최고 보상: {self.best_episode_reward:.2f}")
        # 최종 통계 요약
        self._log("INFO", "\n📊 === 최종 학습 요약 ===")
        total_duration = sum(ep["duration"] for ep in episode_stats)
        avg_reward = np.mean([ep["reward"] for ep in episode_stats])
        self._log("INFO", f"⏱️ 총 지속 시간: {total_duration:.1f}s")
        self._log("INFO", f"📊 평균 보상: {avg_reward:.2f}")
        self._log("INFO", f"📈 최고 보상: {self.best_episode_reward:.2f}")
        self._log("INFO", f"🤖 총 RL 활성 단계: {self.total_rl_active_steps}")
        
        # 최종 통신 통계
        self.comm.print_communication_stats()
        self.comm.close()
# =========================
# Signal Handler for Safe Exit
# =========================
def signal_handler(signum, frame):
    print(f"\n⚠️ Received signal {signum}. Shutting down gracefully...")
    
    # 에피소드별 보상 그래프 생성
    if 'env' in globals():
        try:
            print("📈 에피소드별 보상 그래프 생성 중...")
            env.generate_episode_reward_graph()
            print("✅ 에피소드별 보상 그래프 저장 완료!")
        except Exception as e:
            print(f"❌ 그래프 생성 실패: {e}")
    
    sys.exit(0)

# =========================
# Main
# =========================
if __name__ == "__main__":
    # 시그널 핸들러 설정
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 프로세스 종료
    
    # =========================
    # 송신/수신 주파수 설정
    # =========================
    
    # 송신 주파수 (로봇 제어 PC로 보내는 주파수)
    SEND_FREQUENCY_HZ = 100  # 원하는 송신 주파수 (Hz) 
    
    # 수신 주파수 (로봇 제어 PC에서 받는 주파수)
    RECV_FREQUENCY_HZ = 1000  # 원하는 수신 주파수 (Hz) 
    
    # 설정된 주파수로 CONFIG 생성
    config = create_config(SEND_FREQUENCY_HZ, RECV_FREQUENCY_HZ)
    
    print("🚀 TEST VERSION 7: JY_Pneumatic_SAC_Pre_only_test_7.py")
    print(f"⚡ 송신 주파수: {SEND_FREQUENCY_HZ}Hz (간격: {config['TICK_SEC']:.3f}초)")
    print(f"📡 수신 주파수: {RECV_FREQUENCY_HZ}Hz (간격: {config['RECV_INTERVAL_SEC']:.3f}초)")
    print("=" * 60)
    
    # 랜덤 시드 설정
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    env = PneumaticPolishingEnvironment(config)
    try:
        print(f"🚀 Starting optimized training for {SEND_FREQUENCY_HZ}Hz stable performance...")
        env.run_training(config["EPISODES"])
        print("✅ Training completed successfully!")
        
        # 정상 완료 시에도 에피소드별 보상 그래프 생성
        try:
            print("📈 에피소드별 보상 그래프 생성 중...")
            env.generate_episode_reward_graph()
            print("✅ 에피소드별 보상 그래프 저장 완료!")
        except Exception as e:
            print(f"❌ 그래프 생성 실패: {e}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        env.comm.close()
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        env.comm.close()
    finally:
        print("🔚 Training program terminated.")
