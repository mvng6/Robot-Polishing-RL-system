# Residual SAC Agent for Pneumatic Polishing System - Test Version 6 - OPTIMIZED for 10Hz

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
import pandas as pd
# =========================
# CONFIG
# =========================
CONFIG = {
    # Neural Network
    "STATE_DIM": 6,
    "ACTION_DIM": 1,
    "HIDDEN": 256,
    "LR": 3e-4,
    "GAMMA": 0.99,
    "TAU": 0.005,
    "AUTO_ENTROPY": True,
    # Residual limits (MPa)
    "R_MIN": -0.2,
    "R_MAX":  0.2,
    "R_SLEW_PER_40MS": 0.048,
    # Scheduling - 10Hz
    "TICK_SEC": 0.10,
    "TICK_TOL": 0.02,
    # Training
    "BATCH_SIZE": 64,
    "REPLAY_WARMUP": 25,
    # Networking
    "HOST": "0.0.0.0",
    "PORT": 8888,
    "RECV_TIMEOUT_SEC": 0.5,
    "COMM_FAIL_MAX": 3,
    # Episode
    "EPISODES": 10,
    "MAX_EPISODE_STEPS": 300,
    # Safety / Reward shaping
    "MAX_FORCE_ERR": 15.0,
    "MAX_PRESS_DELTA": 0.05,
    # Logging
    "LOG_EVERY_CTRL": 25,
    "SAVE_THRESH_FREQ": 5,
    "SAVE_THRESH_PCT": 80,
    # Model saving
    "MODEL_SAVE_DIR": "/home/katech/Robot-Polishing-RL-system/py_rl/saved_agents",
    # Data logging
    "LOG_DATA_DIR": "/home/katech/Robot-Polishing-RL-system/py_rl/experiment_logs",
    "LOG_SEND_DATA": True,
}
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
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s_dim, a_dim, hidden = cfg["STATE_DIM"], cfg["ACTION_DIM"], cfg["HIDDEN"]
        self.gamma, self.tau = cfg["GAMMA"], cfg["TAU"]
        self.alpha = 0.2
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

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, log_prob = self.actor.sample(state)
        action = action.cpu().numpy().flatten()
        # scale to residual MPa range [-0.2, 0.2]
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
        # soft update
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
# TCP Communicator (residual only)
# =========================
class ResidualRLCommunicator:
    def __init__(self, host, port, recv_timeout):
        self.host, self.port = host, port 
        self.recv_timeout = recv_timeout
        self.socket = None
        self.conn = None
        self.connected = False
        # Packet format: 6 floats + 1 bool
        self.CPP_TO_PY_PACKET_FORMAT = ">HffffffBH"
        self.CPP_TO_PY_PACKET_SIZE = 29
        self.CPP_TO_PY_SOF = 0xAAAA
        self.PY_TO_CPP_PACKET_FORMAT = ">HfBBH"
        self.PY_TO_CPP_PACKET_SIZE = 10
        self.PY_TO_CPP_SOF = 0xBBBB
        self.latest_state = None
        self.latest_sander_active = False
        self.receive_thread = None
        self.is_receiving = False
        self.state_lock = threading.Lock()  # 스레드 안전성
        self.packets_received = 0
        self.packets_sent = 0
        self.checksum_errors = 0
        self.sof_errors = 0
        self.packet_size_errors = 0
        self.last_packet_time = None
        self.connection_start_time = None
        self.expected_packet_interval = 0.001 # 로봇제어PC에서 100Hz로 전송
        self.packet_receive_times = []
        self.packet_sequence_numbers = []
        self.missed_packets = 0
        self.late_packets = 0
        self.max_packet_history = 100

    def _log(self, level, message):
        """통합된 로깅 함수"""
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
        """100Hz로 계속 수신하는 별도 쓰레드 시작"""
        self.is_receiving = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        self._log("INFO", "100Hz 수신 스레드 시작")

    def _receive_loop(self):
        """100Hz로 계속 수신하는 메인 루프 - 타이밍 기반 수신"""
        next_receive_time = time.perf_counter()
        
        while self.is_receiving:
            current_time = time.perf_counter()
            
            # 100Hz (10ms 간격)로 수신 시도
            if current_time >= next_receive_time:
                next_receive_time += 0.001  # 10ms 간격
                
                try:
                    # 타임아웃을 10ms로 설정 (100Hz 주기에 맞춤)
                    self.conn.settimeout(0.001)
                    data = self._recv_exact(self.CPP_TO_PY_PACKET_SIZE)
                    if data:
                        state, sander_active = self._process_packet(data)
                        if state is not None:
                            with self.state_lock:
                                self.latest_state = state
                                self.latest_sander_active = sander_active
                                self.last_packet_time = time.perf_counter()
                except socket.timeout:
                    # 10ms 내에 데이터가 없으면 다음 주기까지 대기
                    pass
                except Exception as e:
                    self._log("WARNING", f"수신 루프 오류: {e}")
                    break
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
            # 1. 길이 검증
            if len(data) != self.CPP_TO_PY_PACKET_SIZE:
                self.packet_size_errors += 1
                self._log("WARNING", f"예상 {self.CPP_TO_PY_PACKET_SIZE}B, 수신 {len(data)}B (총 {self.packet_size_errors}회)")
                return None, False
            # 2. 언패킹 - 포맷 수정
            try:
                (sof, current_force, target_force, force_error, force_error_dot, 
                 force_error_int, pi_output, sander_active, 
                 received_checksum) = struct.unpack(">HffffffBH", data)
            except struct.error as e:
                self._log("ERROR", f"패킷 언팩 실패: {e}")
                return None, False
            # 3. SOF 검증
            if sof != self.CPP_TO_PY_SOF:
                self.sof_errors += 1
                self._log("WARNING", f"SOF 불일치: {hex(sof)} (예상: {hex(self.CPP_TO_PY_SOF)}) (총 {self.sof_errors}회)")
                return None, False
            # 4. 체크섬 검증 (CRC-16)
            calculated_checksum = self.calculate_crc16(data[:-2])
            if received_checksum != calculated_checksum:
                self.checksum_errors += 1
                self._log("ERROR", f"체크섬 오류: 수신:{received_checksum} 계산:{calculated_checksum} (총 {self.checksum_errors}회)")
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
            # 6. 수신 성공 통계 업데이트
            self.packets_received += 1
            # 🎯 29바이트 데이터 수신 품질 모니터링
            current_time = time.perf_counter()
            # 패킷 수신 시간 기록
            self.packet_receive_times.append(current_time)
            if len(self.packet_receive_times) > self.max_packet_history:
                self.packet_receive_times.pop(0)
            # 패킷 순서 번호 기록
            self.packet_sequence_numbers.append(self.packets_received)
            if len(self.packet_sequence_numbers) > self.max_packet_history:
                self.packet_sequence_numbers.pop(0)
            # 패킷 간격 및 누락 검사 (최소 2개 패킷이 있을 때)
            # 가상 손실률 계산은 제거했지만 변수는 유지 (오류 방지용)
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
                # sander_active 상태 디버깅을 위한 로깅 추가
                if hasattr(self, 'last_logged_sander_active') and self.last_logged_sander_active != self.latest_sander_active:
                    self._log("DEBUG", f"RL 플래그 변경: {self.last_logged_sander_active} -> {self.latest_sander_active}")
                    self.last_logged_sander_active = self.latest_sander_active
                elif not hasattr(self, 'last_logged_sander_active'):
                    self.last_logged_sander_active = self.latest_sander_active
                    self._log("DEBUG", f"초기 RL 플래그: {self.latest_sander_active}")
                return self.latest_state.copy(), self.latest_sander_active
        return None, False
    
    def send_residual(self, rl_residual, timing_accurate, episode_done, packet_monitor=None):
        """10Hz로 residual 전송 (패킷 모니터링 포함)"""
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
            
            # 📊 패킷 모니터링: 전송 예정 패킷 데이터 기록
            if packet_monitor is not None:
                packet_monitor['intended_packet'] = {
                    'sof_hex': f"0x{self.PY_TO_CPP_SOF:04X}",
                    'residual': float(rl_residual),
                    'timing_accurate': bool(timing_accurate),
                    'episode_done': bool(episode_done),
                    'checksum_hex': f"0x{checksum:04X}",
                    'packet_bytes': final_packet.hex().upper(),
                    'packet_size': len(final_packet),
                    'timestamp': time.perf_counter()
                }
            
            # 4. 송신
            self.conn.sendall(final_packet)
            
            # 📊 패킷 모니터링: 실제 전송된 패킷 데이터 기록
            if packet_monitor is not None:
                packet_monitor['sent_packet'] = {
                    'sof_hex': f"0x{self.PY_TO_CPP_SOF:04X}",
                    'residual': float(rl_residual),
                    'timing_accurate': bool(timing_accurate),
                    'episode_done': bool(episode_done),
                    'checksum_hex': f"0x{checksum:04X}",
                    'packet_bytes': final_packet.hex().upper(),
                    'packet_size': len(final_packet),
                    'send_success': True,
                    'timestamp': time.perf_counter()
                }
            
            # 5. 송신 성공 통계 업데이트
            self.packets_sent += 1
            return True
        except Exception as e:
            self._log("ERROR", f"residual 전송 오류: {e}")
            # 📊 패킷 모니터링: 전송 실패 기록
            if packet_monitor is not None:
                packet_monitor['sent_packet'] = {
                    'send_success': False,
                    'error': str(e),
                    'timestamp': time.perf_counter()
                }
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
        """통신 통계 반환"""
        uptime = time.perf_counter() - self.connection_start_time if self.connection_start_time else 0
        # 🎯 29바이트 데이터 수신 품질 통계 계산
        packet_quality_stats = {}
        if len(self.packet_receive_times) >= 2:
            # 평균 패킷 간격 계산
            intervals = [self.packet_receive_times[i] - self.packet_receive_times[i-1] 
                        for i in range(1, len(self.packet_receive_times))]
            avg_interval = sum(intervals) / len(intervals)
            packet_quality_stats["avg_packet_interval_ms"] = avg_interval * 1000
            # 패킷 간격 표준편차 계산
            interval_variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            packet_quality_stats["packet_interval_std_ms"] = (interval_variance ** 0.5) * 1000
            # 누락된 패킷 비율 계산
            expected_packets = uptime / self.expected_packet_interval
            packet_quality_stats["packet_loss_rate_percent"] = (self.missed_packets / expected_packets * 100) if expected_packets > 0 else 0
        return {
            "uptime_seconds": uptime,
            "packets_received": self.packets_received,
            "packets_sent": self.packets_sent,
            "checksum_errors": self.checksum_errors,
            "sof_errors": self.sof_errors,
            "packet_size_errors": self.packet_size_errors,
            "last_packet_time": self.last_packet_time,
            "receive_rate_hz": self.packets_received / uptime if uptime > 0 else 0,
            "send_rate_hz": self.packets_sent / uptime if uptime > 0 else 0,
            # 가상 손실률 통계는 제거했지만 변수는 유지 (오류 방지용)
            "missed_packets": self.missed_packets,
            "late_packets": self.late_packets,
            **packet_quality_stats
        }
    def print_communication_stats(self):
        """통신 통계 출력 - Robot PC 100Hz 전송 기준으로 평가"""
        stats = self.get_communication_stats()
        self._log("INFO", "\n📊 === 통신 통계 ===")
        self._log("INFO", f"⏱️  가동 시간: {stats['uptime_seconds']:.1f}s")
        self._log("INFO", f"📥 수신된 패킷: {stats['packets_received']}")
        self._log("INFO", f"📤 송신된 패킷: {stats['packets_sent']}")
        self._log("INFO", f"📥 수신률: {stats['receive_rate_hz']:.1f} Hz")
        self._log("INFO", f"📤 송신률: {stats['send_rate_hz']:.1f} Hz")
        self._log("INFO", f"❌ 체크섬 오류: {stats['checksum_errors']}")
        self._log("INFO", f"⚠️  SOF 오류: {stats['sof_errors']}")
        self._log("INFO", f"📏 패킷 크기 오류: {stats['packet_size_errors']}")
        
        # 🎯 Robot PC 100Hz 전송 기준으로 패킷 손실률 재계산
        robot_packets_sent = int(stats['uptime_seconds'] * 100)  # 100Hz
        successfully_received = stats['packets_received']
        actual_missed = robot_packets_sent - successfully_received
        actual_loss_rate = (actual_missed / robot_packets_sent) * 100 if robot_packets_sent > 0 else 0
        
        self._log("INFO", "\n🤖 === 로봇PC 기반 패킷 손실 ===")
        self._log("INFO", f"🤖 로봇PC 송신: {robot_packets_sent:,} 패킷 (100Hz)")
        self._log("INFO", f"📥 성공 수신: {successfully_received:,}")
        self._log("INFO", f"❌ 실제 누락: {actual_missed:,}")
        self._log("INFO", f"📊 실제 손실률: {actual_loss_rate:.3f}%")
        
        # Robot PC 기준 품질 평가
        if actual_loss_rate < 0.1:
            robot_quality = "🟢 EXCELLENT"
        elif actual_loss_rate < 1.0:
            robot_quality = "🟡 GOOD"
        elif actual_loss_rate < 5.0:
            robot_quality = "🟠 FAIR"
        else:
            robot_quality = "🔴 POOR"
        self._log("INFO", f"🤖 로봇PC 품질: {robot_quality}")
        
        # 가상 손실률 관련 정보 제거 - 불필요한 모니터링
        
        if stats['last_packet_time']:
            time_since_last = time.perf_counter() - stats['last_packet_time']
            self._log("INFO", f"\n🕐 마지막 패킷: {time_since_last:.3f}s 전")
        
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
# Environment (Method 1) - TEST VERSION
# =========================
class PneumaticPolishingEnvironment:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.agent = ResidualSACAgent(cfg)
        self.comm = ResidualRLCommunicator(cfg["HOST"], cfg["PORT"], cfg["RECV_TIMEOUT_SEC"])
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
        self.send_data_logger = None
        self.episode_send_data = []
        # 송신 데이터 추적 및 딜레이 모니터링
        self.send_data_history = []
        self.training_start_time = None
        self.episode_timing_stats = {}  # 에피소드별 타이밍 통계
        self.current_episode_delays = []  # 현재 에피소드 딜레이
        # 마지막 유효한 상태 저장 (송신 연속성 보장)
        self.last_valid_state = None
        self.last_sander_active = False
        # episode_done 전송 모니터링
        self.episode_done_monitor = []
        self.base_timestamp = None
        # 패킷 모니터링 (전송 예정 vs 실제 전송)
        self.packet_intended_monitor = []
        self.packet_sent_monitor = []
        
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
    def generate_send_data_analysis(self):
        """송신 데이터 시간별 분석 및 Excel/그래프 생성"""
        if not self.cfg["LOG_SEND_DATA"] or not self.send_data_history:
            self._log("WARNING", "분석할 송신 데이터 없음")
            return
            
        try:
            self._log("INFO", "📊 송신 데이터 분석 생성 중...")
            
            # DataFrame 생성
            df = pd.DataFrame(self.send_data_history)
            
            # Excel 파일 생성 (2개 시트) - openpyxl 가용성 확인
            try:
                with pd.ExcelWriter(self.excel_filename, engine='openpyxl') as writer:
                    # 원본 데이터
                    df.to_excel(writer, sheet_name='Raw_Data', index=False)
                    
                    # 에피소드별 요약 (딜레이 통계 포함)
                    df_episode = self._create_episode_summary(df)
                    df_episode.to_excel(writer, sheet_name='Episode_Summary', index=False)
                    
                    # 딜레이 상세 데이터 (딜레이가 발생한 시점들만)
                    df_delays = df[df['timing_accurate'] == 0].copy() if len(df[df['timing_accurate'] == 0]) > 0 else pd.DataFrame()
                    if not df_delays.empty:
                        df_delays.to_excel(writer, sheet_name='Delays_Detail', index=False)
                    
                self._log("INFO", f"📈 Excel 파일 생성: {self.excel_filename}")
            except ImportError:
                self._log("WARNING", "openpyxl 라이브러리가 없어 Excel 파일을 생성할 수 없습니다. CSV 파일만 사용됩니다.")
            except Exception as excel_error:
                self._log("ERROR", f"Excel 파일 생성 오류: {excel_error}")
            
            # 그래프 생성
            self._create_send_data_graphs(df)
            self._log("INFO", f"🎨 그래프 저장: {self.graph_filename}")
            
        except Exception as e:
            self._log("ERROR", f"송신 데이터 분석 오류: {e}")
    
    def _create_episode_summary(self, df):
        """에피소드별 요약 데이터 생성 (딜레이 모니터링 포함)"""
        summary = df.groupby('episode').agg({
            'residual_mpa': ['mean', 'std', 'min', 'max'],
            'timing_accurate': ['mean', 'sum', 'count'],  # 정확도, 정확 횟수, 전체 횟수
            'episode_done': 'sum',
            'time_s': ['min', 'max']
        }).reset_index()
        
        # 컬럼명 정리
        summary.columns = [
            'episode', 'residual_mean', 'residual_std', 'residual_min', 'residual_max',
            'timing_accuracy_rate', 'accurate_count', 'total_packets', 'episode_done_count', 'start_time_s', 'end_time_s'
        ]
        
        # 추가 계산 컬럼
        summary['duration_s'] = summary['end_time_s'] - summary['start_time_s']
        summary['delay_count'] = summary['total_packets'] - summary['accurate_count']  # 딜레이 횟수
        summary['delay_rate_percent'] = (summary['delay_count'] / summary['total_packets'] * 100).round(2)  # 딜레이 비율
        summary['target_packets'] = (summary['duration_s'] / 0.10).round(0).astype(int)  # 10Hz 기준 목표 패킷 수
        summary['send_rate_hz'] = (summary['total_packets'] / summary['duration_s']).round(2)  # 실제 전송속도
        
        return summary
    
    def _create_send_data_graphs(self, df):
        """송신 데이터 시각화 그래프 생성"""
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            fig.suptitle('RL System Send Data Analysis - Time Series', fontsize=16, fontweight='bold')
            
            # pandas Series를 numpy array로 변환하여 matplotlib 호환성 확보
            time_data = df['time_s'].values
            residual_data = df['residual_mpa'].values
            timing_data = df['timing_accurate'].values
            episode_done_data = df['episode_done'].values
            
            # 1. Residual Pressure over Time
            ax1 = axes[0]
            ax1.plot(time_data, residual_data, 'b-', linewidth=0.8, alpha=0.7, label='Residual Pressure')
            ax1.set_ylabel('Residual Pressure (MPa)', fontsize=12)
            ax1.set_title('1. Residual Pressure Sent to Robot PC', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Episode 구분선 추가
            episode_changes = df[df['step'] == 1]['time_s'].values
            for ep_time in episode_changes[1:]:  # 첫 번째 제외
                ax1.axvline(x=ep_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # 2. Timing Accuracy over Time (0/1 format)
            ax2 = axes[1]
            timing_smooth = df['timing_accurate'].rolling(window=10, center=True).mean().values  # numpy array로 변환
            ax2.plot(time_data, timing_smooth, 'g-', linewidth=1.2, label='Timing Accuracy (1s avg)')
            
            # Late packet 시점만 추출
            late_mask = timing_data == 0
            late_times = time_data[late_mask]
            late_values = np.zeros(len(late_times))
            ax2.scatter(late_times, late_values, color='red', alpha=0.6, s=10, label='Late Packets (0)')
            
            ax2.set_ylabel('Timing Accuracy (0=Late, 1=OnTime)', fontsize=12)
            ax2.set_title('2. Timing Accuracy (10Hz Target)', fontsize=14, fontweight='bold')
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Episode 구분선 추가
            for ep_time in episode_changes[1:]:
                ax2.axvline(x=ep_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # 3. Episode Done Flags over Time (0/1 format)
            ax3 = axes[2]
            episode_done_mask = episode_done_data == 1
            episode_done_times = time_data[episode_done_mask]
            episode_done_episodes = df[df['episode_done'] == 1]['episode'].values
            
            # Episode Done 신호를 시간축에 따라 표시 (Y축은 해당 시간의 에피소드 번호)
            episode_at_done_times = df.loc[episode_done_mask, 'episode'].values
            
            ax3.scatter(episode_done_times, episode_at_done_times, 
                       color='orange', s=50, alpha=0.8, label='Episode Done Signals (1)')
            ax3.set_xlabel('Time (seconds)', fontsize=12)
            ax3.set_ylabel('Episode', fontsize=12)
            ax3.set_title('3. Episode Done Flags Sent to Robot PC (0=Continue, 1=Done)', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Episode 구분선 추가 (다른 그래프와 동일한 방식)
            for ep_time in episode_changes[1:]:
                ax3.axvline(x=ep_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            plt.tight_layout()
            plt.savefig(self.graph_filename, dpi=300, bbox_inches='tight')
            plt.close()
            self._log("INFO", f"✅ 그래프 파일 생성 성공: {self.graph_filename}")
            
        except Exception as e:
            self._log("ERROR", f"❌ 그래프 생성 오류: {e}")
            self._log("WARNING", "⚠️ 그래프 생성 실패했지만 분석은 계속 진행됩니다")
    
    def generate_communication_graphs(self):
        """통합 통신 모니터링 - 통계 + 송신 데이터 분석"""
        # 기존 통신 통계
        if hasattr(self, 'comm') and self.comm:
            self.comm.print_communication_stats()
        else:
            self._log("WARNING", "통계를 위한 통신 객체 없음")
        
        # 새로운 송신 데이터 분석
        self.generate_send_data_analysis()
    def _init_data_logging(self):
        """향상된 데이터 로깅 시스템 - CSV + Excel + 자동 그래프 생성"""
        if self.cfg["LOG_SEND_DATA"]:
            log_dir = self.cfg["LOG_DATA_DIR"]
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # CSV 파일 초기화 (단순화된 헤더)
            csv_filename = f"{log_dir}/send_data_log_{timestamp}.csv"
            with open(csv_filename, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow([
                    'time_s', 'episode', 'step', 'residual_mpa', 
                    'timing_accurate', 'episode_done', 'sander_active', 'current_force', 
                    'target_force', 'force_error', 'force_error_dot', 'force_error_int', 'pi_output'
                ])
            
            self.send_data_logger = csv_filename
            self.training_start_time = time.perf_counter()
            self.episode_send_data = []
            self.send_data_history = []
            self.episode_timing_stats = {}
            self.current_episode_delays = []
            
            # Excel 및 그래프 파일명 설정
            self.excel_filename = f"{log_dir}/RL_Send_Analysis_{timestamp}.xlsx"
            self.graph_filename = f"{log_dir}/RL_Send_Graphs_{timestamp}.png"
            
            self._log("INFO", f"📊 향상된 데이터 로깅 초기화:")
            self._log("INFO", f"  📄 CSV: {csv_filename}")
            self._log("INFO", f"  📈 Excel: {self.excel_filename}")
            self._log("INFO", f"  🎨 그래프: {self.graph_filename}")
            
    def _log_send_data(self, episode, step, residual_mpa, timing_accurate, episode_done, sander_active, state):
        """향상된 송신 데이터 로깅 - 시간대별 추적 포함"""
        if not self.cfg["LOG_SEND_DATA"] or not self.send_data_logger:
            return
            
        # 상대 시간 기록 (학습 시작 이후 경과 시간)
        relative_time_s = time.perf_counter() - self.training_start_time
        
        data_row = [
            relative_time_s, episode, step, residual_mpa, 
            1 if timing_accurate else 0,  # 플래그를 0/1로 저장
            1 if episode_done else 0,     # 플래그를 0/1로 저장
            1 if sander_active else 0,    # 플래그를 0/1로 저장
            state[0] if state is not None else 0.0,  # current_force
            state[1] if state is not None else 0.0,  # target_force
            state[2] if state is not None else 0.0,  # force_error
            state[3] if state is not None else 0.0,  # force_error_dot
            state[4] if state is not None else 0.0,  # force_error_int
            state[5] if state is not None else 0.0   # pi_output
        ]
        
        try:
            with open(self.send_data_logger, 'a', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(data_row)
        except Exception as e:
            self._log("ERROR", f"CSV 로깅 오류: {e}")
        
        # 딜레이 모니터링 (타이밍이 정확하지 않은 경우)
        if not timing_accurate:
            self.current_episode_delays.append({
                'episode': episode,
                'step': step,
                'time_s': relative_time_s,
                'residual_mpa': residual_mpa
            })
        
        # 분석용 데이터 저장
        send_data_entry = {
            'time_s': relative_time_s,
            'episode': episode,
            'step': step,
            'residual_mpa': residual_mpa,
            'timing_accurate': 1 if timing_accurate else 0,  # 0=딜레이, 1=정확
            'episode_done': 1 if episode_done else 0,        # 0=False, 1=True
            'sander_active': 1 if sander_active else 0,      # 0=False, 1=True
            'current_force': state[0] if state is not None else 0.0,
            'target_force': state[1] if state is not None else 0.0,
            'force_error': state[2] if state is not None else 0.0,
            'force_error_dot': state[3] if state is not None else 0.0,
            'force_error_int': state[4] if state is not None else 0.0,
            'pi_output': state[5] if state is not None else 0.0
        }
        
        self.episode_send_data.append(send_data_entry)
        self.send_data_history.append(send_data_entry)
    
    def _log_episode_done_monitoring(self, episode, step, success, residual, timing_accurate, episode_done):
        """episode_done 전송 모니터링 데이터 로깅 - 매 단계마다 호출"""
        if not self.cfg["LOG_SEND_DATA"]:
            return
            
        # 상대 시간 기록 (학습 시작 이후 경과 시간)
        relative_time_s = time.perf_counter() - self.training_start_time if self.training_start_time else 0
        
        monitor_entry = {
            'time_s': relative_time_s,
            'episode': episode,
            'step': step,
            'success': 1 if success else 0,
            'residual_mpa': residual,
            'timing_accurate': 1 if timing_accurate else 0,
            'episode_done': 1 if episode_done else 0,
            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
        }
        
        self.episode_done_monitor.append(monitor_entry)
    
    def _save_episode_done_monitor(self, episode):
        """각 에피소드별 episode_done 모니터링 데이터를 CSV로 저장"""
        if not self.cfg["LOG_SEND_DATA"] or not self.episode_done_monitor:
            return
            
        try:
            log_dir = self.cfg["LOG_DATA_DIR"]
            os.makedirs(log_dir, exist_ok=True)
            
            # 파일명: episode_done_monitor_날짜시간_episode번호.csv
            if not self.base_timestamp:
                self.base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"{log_dir}/episode_done_monitor_{self.base_timestamp}_ep{episode:02d}.csv"
            
            with open(filename, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                # 헤더 작성
                writer.writerow([
                    'time_s', 'episode', 'step', 'success', 'residual_mpa', 
                    'timing_accurate', 'episode_done', 'timestamp'
                ])
                
                # 데이터 작성 (현재 에피소드 관련 데이터만)
                episode_data = [entry for entry in self.episode_done_monitor if entry['episode'] == episode]
                for entry in episode_data:
                    writer.writerow([
                        entry['time_s'], entry['episode'], entry['step'], 
                        entry['success'], entry['residual_mpa'], entry['timing_accurate'],
                        entry['episode_done'], entry['timestamp']
                    ])
            
            self._log("INFO", f"📁 Episode {episode} done 모니터링 저장: {filename}")
            self._log("INFO", f"📊 총 {len(episode_data)}개 단계 기록됨")
            
        except Exception as e:
            self._log("ERROR", f"Episode done 모니터링 저장 오류: {e}")
    
    def _log_packet_monitoring(self, episode, step, packet_data):
        """패킷 모니터링 데이터 로깅 (전송 예정 vs 실제 전송) - 매 단계마다 호출"""
        if not self.cfg["LOG_SEND_DATA"]:
            return
            
        if packet_data is None:
            self._log("DEBUG", f"패킷 모니터링 건너뜀 - packet_data is None (에피소드 {episode}, 단계 {step})")
            return
            
        # 상대 시간 기록
        relative_time_s = time.perf_counter() - self.training_start_time if self.training_start_time else 0
        
        # 전송 예정 패킷 데이터
        if 'intended_packet' in packet_data:
            intended = packet_data['intended_packet']
            intended_entry = {
                'time_s': relative_time_s,
                'episode': episode,
                'step': step,
                'sof_hex': intended['sof_hex'],
                'residual': intended['residual'],
                'timing_accurate': intended['timing_accurate'],
                'episode_done': intended['episode_done'],
                'checksum_hex': intended['checksum_hex'],
                'packet_bytes': intended['packet_bytes'],
                'packet_size': intended['packet_size'],
                'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            self.packet_intended_monitor.append(intended_entry)
        
        # 실제 전송된 패킷 데이터
        if 'sent_packet' in packet_data:
            sent = packet_data['sent_packet']
            sent_entry = {
                'time_s': relative_time_s,
                'episode': episode,
                'step': step,
                'send_success': sent.get('send_success', False),
                'error': sent.get('error', ''),
                'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            
            # 전송 성공 시에만 패킷 데이터 추가
            if sent.get('send_success', False):
                sent_entry.update({
                    'sof_hex': sent['sof_hex'],
                    'residual': sent['residual'],
                    'timing_accurate': sent['timing_accurate'],
                    'episode_done': sent['episode_done'],
                    'checksum_hex': sent['checksum_hex'],
                    'packet_bytes': sent['packet_bytes'],
                    'packet_size': sent['packet_size']
                })
            
            self.packet_sent_monitor.append(sent_entry)
    
    def _save_packet_monitoring(self, episode):
        """각 에피소드별 패킷 모니터링 데이터를 CSV로 저장"""
        if not self.cfg["LOG_SEND_DATA"]:
            self._log("WARNING", "LOG_SEND_DATA가 False로 설정되어 패킷 모니터링 저장하지 않음")
            return
            
        try:
            log_dir = self.cfg["LOG_DATA_DIR"]
            os.makedirs(log_dir, exist_ok=True)
            
            if not self.base_timestamp:
                self.base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 전송 예정 패킷 CSV 파일
            intended_filename = f"{log_dir}/packet_intended_{self.base_timestamp}_ep{episode:02d}.csv"
            episode_intended_data = [entry for entry in self.packet_intended_monitor if entry['episode'] == episode]
            

            
            if episode_intended_data:
                with open(intended_filename, 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow([
                        'time_s', 'episode', 'step', 'sof_hex', 'residual', 
                        'timing_accurate', 'episode_done', 'checksum_hex', 
                        'packet_bytes', 'packet_size', 'timestamp'
                    ])
                    
                    for entry in episode_intended_data:
                        writer.writerow([
                            entry['time_s'], entry['episode'], entry['step'],
                            entry['sof_hex'], entry['residual'], entry['timing_accurate'],
                            entry['episode_done'], entry['checksum_hex'], 
                            entry['packet_bytes'], entry['packet_size'], entry['timestamp']
                        ])
                
                self._log("INFO", f"📁 Episode {episode} 전송예정 패킷 저장: {intended_filename}")
            
            # 실제 전송된 패킷 CSV 파일
            sent_filename = f"{log_dir}/packet_sent_{self.base_timestamp}_ep{episode:02d}.csv"
            episode_sent_data = [entry for entry in self.packet_sent_monitor if entry['episode'] == episode]
            
            if episode_sent_data:
                with open(sent_filename, 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow([
                        'time_s', 'episode', 'step', 'send_success', 'error',
                        'sof_hex', 'residual', 'timing_accurate', 'episode_done', 
                        'checksum_hex', 'packet_bytes', 'packet_size', 'timestamp'
                    ])
                    
                    for entry in episode_sent_data:
                        writer.writerow([
                            entry['time_s'], entry['episode'], entry['step'],
                            entry['send_success'], entry.get('error', ''),
                            entry.get('sof_hex', ''), entry.get('residual', ''),
                            entry.get('timing_accurate', ''), entry.get('episode_done', ''),
                            entry.get('checksum_hex', ''), entry.get('packet_bytes', ''),
                            entry.get('packet_size', ''), entry['timestamp']
                        ])
                
                self._log("INFO", f"📁 Episode {episode} 실제전송 패킷 저장: {sent_filename}")
            
            self._log("INFO", f"📊 패킷 모니터링 완료 - 예정:{len(episode_intended_data)}, 전송:{len(episode_sent_data)}")
            
        except Exception as e:
            self._log("ERROR", f"패킷 모니터링 저장 오류: {e}")
    # ---- scheduler ----
    def should_send_now(self):
        """정확한 10Hz 타이밍 제어 - 최적화된 버전"""
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
    # ---- reward / done ----
    def calculate_reward(self, state, action_residual, sander_active):
        current_force, target_force = state[0], state[1]
        force_err = abs(current_force - target_force)
        residual_change = abs(action_residual - self.prev_residual)
        # 1) tracking
        reward = -(force_err / self.cfg["MAX_FORCE_ERR"])
        if force_err < 1.0: reward += 0.5
        # 2) smoothness (sander_active에 따라 가중치 조정)
        smooth_w = 0.2 if sander_active else 0.3
        reward += -smooth_w * (residual_change / self.cfg["MAX_PRESS_DELTA"])
        # 3) safety
        if current_force > 80.0: reward += -5.0
        # 4) residual magnitude penalty
        reward += -0.1 * abs(action_residual)
        return float(reward)

    def is_done(self, state):
        # 🎯 오직 최대 스텝에 도달했을 때만 True (에피소드 끝까지 진행)
        if self.episode_step >= self.max_episode_steps: 
            return True
        # 🚨 안전장치: 접촉력이 과도하게 높을 때만 종료
        if state[0] > 100.0: 
            self._log("WARNING", f"안전: 힘이 너무 높음 ({state[0]:.1f}N > 100N) - 에피소드 종료")
            return True
        # ✅ 다른 모든 경우: 계속 진행 (목표 접촉력 달성해도 계속)
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
    # ---- episode helpers - OPTIMIZED FOR 10Hz ----
    def end_episode_fast_with_reliable_flag(self, ep, target_force):
        """한 번만 episode_done 신호 전송으로 에피소드 종료"""
        self._log("INFO", f"🎯 에피소드 {ep+1} 완료 (단계 {self.episode_step})")
        # self._log("INFO", f"🚀 episode_done 신호 한 번 전송")
        
        # 🚀 핵심 수정: episode_done을 딱 한 번만 전송
        packet_monitor = {}
        success = self.comm.send_residual(0.0, True, True, packet_monitor)
        
        # 📊 모니터링: episode_done 전송 기록
        self._log_episode_done_monitoring(ep+1, 1000, success, 0.0, True, True)
        # 📊 모니터링: 패킷 데이터 기록
        self._log_packet_monitoring(ep+1, 1000, packet_monitor)
        
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
            time.sleep(0.01)  # 10ms 간격으로 확인
        else:
            # 500ms 후에도 확인 안되면 강제 성공
            self._log("INFO", f"⚡ Episode done 신호 전송 후 에피소드 종료 (500ms 타임아웃) - 다음 에피소드로 진행")
        
        # 🎯 마지막으로 episode_done=False 신호 한 번 전송
        packet_monitor = {}
        success = self.comm.send_residual(0.0, True, False, packet_monitor)
        
        # 📊 모니터링: episode_done=False 전송도 기록
        self._log_episode_done_monitoring(ep+1, 2000, success, 0.0, True, False)
        # 📊 모니터링: 패킷 데이터 기록
        self._log_packet_monitoring(ep+1, 2000, packet_monitor)
        
        # 📊 에피소드별 모니터링 데이터 저장
        self._save_episode_done_monitor(ep+1)
        self._save_packet_monitoring(ep+1)
        
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
        self._log("INFO", "🚀 최적화된 Residual RL 학습 시작 - 버전 6")
        self._log("INFO", "📡 목표: 10Hz residual 출력 (100ms 간격)")
        self._log("INFO", "⚡ 최적화: 10Hz 안정적 송신을 위한 빠른 에피소드 전환")
        self._log("INFO", "⏱️  에피소드: 300 단계 (10Hz에서 30초)")
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
        self._init_data_logging()
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
            # 10Hz 타이밍 제어를 위한 변수 초기화
            next_send_time = time.perf_counter() + self.cfg["TICK_SEC"]
            
            while True:
                current_time = time.perf_counter()
                
                # 10Hz 타이밍이 되었을 때만 처리
                if current_time >= next_send_time:
                    # 다음 전송 시간 설정 (정확한 100ms 간격)
                    next_send_time += self.cfg["TICK_SEC"]
                    timing_accurate = abs(current_time - (next_send_time - self.cfg["TICK_SEC"])) <= self.cfg["TICK_TOL"]
                    
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
                # 10Hz 기준으로 스텝 증가
                self.episode_step += 1
                
                # 이전 transition 처리 및 학습
                if prev_state is not None and prev_sander_active:
                    reward = self.calculate_reward(prev_state, prev_action, prev_sander_active)
                    done = self.is_done(state)
                    self.agent.store_transition(prev_state, prev_action, reward, state, done)
                    self.current_episode_reward += reward
                    if len(self.agent.replay) > self.cfg["REPLAY_WARMUP"]:
                        self.agent.update_parameters(self.cfg["BATCH_SIZE"])
                
                # RL 활성 상태 확인
                rl_status = self.check_rl_status(sander_active)
                if rl_status == "terminate":
                    self._log("INFO", "RL 비활성 지속으로 에피소드 종료")
                    break
                
                # 에피소드 종료 확인
                if self.episode_step >= self.max_episode_steps:
                    episode_done = True
                    rl_residual = 0.0
                    self._log("INFO", f"🎯 에피소드 {ep+1} 종료 (단계 {self.episode_step}) - episode_done=True 전송")
                else:
                    episode_done = False
                    if sander_active:
                        raw_res = self.agent.select_action(state, evaluate=False)
                        rl_residual = self.limit_residual(raw_res)
                        episode_rl_active_steps += 1
                    else:
                        rl_residual = 0.0
                # residual 전송
                # 📊 패킷 모니터링 데이터 준비
                packet_monitor = {}
                ok = self.comm.send_residual(rl_residual, timing_accurate, episode_done, packet_monitor)
                
                # 📊 모니터링: 매 단계마다 episode_done 상태 기록
                self._log_episode_done_monitoring(ep+1, self.episode_step, ok, rl_residual, timing_accurate, episode_done)
                
                # 📊 모니터링: 매 단계마다 패킷 데이터 기록
                self._log_packet_monitoring(ep+1, self.episode_step, packet_monitor)
                
                # CSV 데이터 로깅
                self._log_send_data(ep+1, self.episode_step, rl_residual, timing_accurate, episode_done, sander_active, state)
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
                    force_achieved = " 🎯" if abs(state[0] - state[1]) < 0.5 else ""
                    timing_status = "EXACT" if timing_accurate else "LATE"
                    force_achieved = " 🎯 TARGET ACHIEVED!" if abs(state[0] - state[1]) < 0.5 else ""
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
            if self.current_episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.current_episode_reward
                self.best_agent_episode = ep
                self.agent.save_model(f"{self.cfg['MODEL_SAVE_DIR']}/test_best_agent_episode_{ep+1}_reward_{self.best_episode_reward:.2f}.pth")
            # dynamic threshold saving (테스트용으로 더 자주)
            if ep % self.cfg["SAVE_THRESH_FREQ"] == 0 and ep > 0:
                recent = self.agent.episode_rewards[-self.cfg["SAVE_THRESH_FREQ"]:]
                th = np.percentile(recent, self.cfg["SAVE_THRESH_PCT"])
                if self.current_episode_reward >= th:
                    self.agent.save_model(f"{self.cfg['MODEL_SAVE_DIR']}/test_high_perf_ep_{ep+1}_reward_{self.current_episode_reward:.2f}.pth")
            # 에피소드 완료 요약
            self._log("INFO", f"\n🎯 === 에피소드 {ep+1}/10 완료 ===")
            self._log("INFO", f"⏱️  지속 시간: {episode_duration:.1f}s")
            self._log("INFO", f"📊 단계: {self.episode_step:,}/{self.max_episode_steps:,} ({self.episode_step/self.max_episode_steps*100:.1f}%)")
            self._log("INFO", f"🏆 보상: {self.current_episode_reward:.2f}")
            self._log("INFO", f"📥 수신 패킷: {episode_packets_received}")
            self._log("INFO", f"📤 송신 패킷: {episode_packets_sent}")
            self._log("INFO", f"🤖 RL 활성 단계: {episode_rl_active_steps} ({episode_stat['rl_active_ratio']*100:.1f}%)")
            self._log("INFO", f"📈 지금까지 최고: {self.best_episode_reward:.2f}")
            # 🎯 데이터 로깅 요약
            if self.cfg["LOG_SEND_DATA"] and self.send_data_logger:
                self._log("INFO", f"📊 데이터 로깅: {len(self.episode_send_data)}개 레코드를 {self.send_data_logger}에")
            # 🎯 에피소드 완료 이유 표시
            if self.episode_step >= self.max_episode_steps:
                self._log("INFO", f"✅ 완료: 최대 에피소드 단계 도달 ({self.max_episode_steps:,})")
            else:
                self._log("WARNING", "⚠️  완료: 에피소드 조기 종료 (안전 또는 오류)")
            self._log("INFO", "=" * 40)
            # 통신 상태 모니터링 (매 에피소드마다)
            self.comm.print_communication_stats()
        # ---- 전체 학습 완료 ----
        self._log("INFO", "\n🎯 최적화된 학습 완료!")
        self._log("INFO", f"✅ {episodes}개 에피소드 성공적으로 완료")
        self._log("INFO", f"🏆 최고 에피소드: {self.best_agent_episode+1}, 최고 보상: {self.best_episode_reward:.2f}")
        # 🎯 통신 모니터링 그래프 자동 생성
        self._log("INFO", "\n📊 통신 모니터링 그래프 생성 중...")
        self.generate_communication_graphs()
        self._log("INFO", "✅ 통신 그래프 성공적으로 생성!")
        self._log("INFO", f"📈 총 RL 활성 단계: {self.total_rl_active_steps}")
        # 전체 통계 요약
        self._log("INFO", "\n📊 === 최종 학습 요약 ===")
        total_duration = sum(ep["duration"] for ep in episode_stats)
        total_packets_received = sum(ep["packets_received"] for ep in episode_stats)
        total_packets_sent = sum(ep["packets_sent"] for ep in episode_stats)
        avg_reward = np.mean([ep["reward"] for ep in episode_stats])
        self._log("INFO", f"⏱️  총 지속 시간: {total_duration:.1f}s")
        self._log("INFO", f"📥 총 수신 패킷: {total_packets_received}")
        self._log("INFO", f"📤 총 송신 패킷: {total_packets_sent}")
        self._log("INFO", f"📊 평균 보상: {avg_reward:.2f}")
        self._log("INFO", f"📈 최고 보상: {self.best_episode_reward:.2f}")
        self._log("INFO", f"🤖 총 RL 활성 단계: {self.total_rl_active_steps}")
        # 🚀 성능 개선 분석
        actual_send_rate = total_packets_sent / total_duration if total_duration > 0 else 0
        target_send_rate = 10.0
        performance_improvement = (actual_send_rate / target_send_rate) * 100 if target_send_rate > 0 else 0
        
        self._log("INFO", "\n⚡ === 성능 분석 ===")
        self._log("INFO", f"🎯 목표 송신률: {target_send_rate:.1f} Hz")
        self._log("INFO", f"📊 실제 송신률: {actual_send_rate:.1f} Hz")
        self._log("INFO", f"📈 성능: 목표의 {performance_improvement:.1f}%")
        
        if performance_improvement >= 95:
            self._log("INFO", f"🟢 우수: 10Hz 목표에 매우 근접!")
        elif performance_improvement >= 85:
            self._log("INFO", f"🟡 양호: 10Hz 목표에 근접")
        elif performance_improvement >= 70:
            self._log("INFO", f"🟠 보통: 합리적인 성능")
        else:
            self._log("INFO", f"🔴 개선 필요: 목표 성능 이하")
        
        # 에피소드별 상세 통계
        self._log("INFO", "\n📋 === 에피소드 상세 ===")
        for ep_stat in episode_stats:
            self._log("INFO", f"에피 {ep_stat['episode']:2d}: "
                  f"Reward {ep_stat['reward']:6.2f}, "
                  f"Steps {ep_stat['steps']:4d}, "
                  f"Duration {ep_stat['duration']:5.1f}s, "
                  f"RL Active {ep_stat['rl_active_steps']:4d} "
                  f"({ep_stat['rl_active_ratio']*100:4.1f}%)")
        # 최종 통신 통계
        self.comm.print_communication_stats()
        # 🎯 통신 모니터링 그래프 생성
        self._log("INFO", "\n📊 === 통신 모니터링 그래프 생성 ===")
        self.generate_communication_graphs()
        self.comm.close()
# =========================
# Signal Handler for Safe Exit
# =========================
def signal_handler(signum, frame):
    print(f"\n⚠️ Received signal {signum}. Shutting down gracefully...")
    sys.exit(0)

# =========================
# Main - CLEANED VERSION
# =========================
if __name__ == "__main__":
    # 시그널 핸들러 설정
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 프로세스 종료
    
    print("🚀 TEST VERSION 6: JY_Pneumatic_SAC_Pre_only_test_6.py")
    print("⚡ OPTIMIZATION: Fast episode transitions for 10Hz stable performance")
    print("=" * 60)
    # 랜덤 시드 설정
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    env = PneumaticPolishingEnvironment(CONFIG)
    try:
        print("🚀 Starting optimized training for 10Hz stable performance...")
        env.run_training(CONFIG["EPISODES"])
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        env.comm.close()
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\n📊 Generating communication graphs before exit...")
        try:
            env.generate_communication_graphs()
            print("✅ Communication graphs generated successfully!")
        except Exception as graph_error:
            print(f"⚠️ Failed to generate graphs: {graph_error}")
        env.comm.close()
    finally:
        print("🔚 Training program terminated.")
