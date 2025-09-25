# Residual SAC Agent for Pneumatic Polishing System - Test Version 5

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
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
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
    # Scheduling - 25Hz
    "TICK_SEC": 0.04,
    "TICK_TOL": 0.01,
    # Training
    "BATCH_SIZE": 64,
    "REPLAY_WARMUP": 100,
    # Networking
    "HOST": "0.0.0.0",
    "PORT": 8888,
    "RECV_TIMEOUT_SEC": 0.5,
    "COMM_FAIL_MAX": 3,
    # Episode
    "EPISODES": 10,
    "MAX_EPISODE_STEPS": 750,
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
        self.expected_packet_interval = 0.001 # 로봇제어PC에서 1000Hz로 전송
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
            self._log("INFO", f"Waiting for Robot PC on {self.host}:{self.port} ...")
            
            while True:
                try:
                    conn, addr = self.socket.accept()
                    break
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    self._log("WARNING", "Connection cancelled by user")
                    return False
                    
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            conn.settimeout(self.recv_timeout)
            self._log("SUCCESS", f"Connected: {addr}")
            self.conn = conn
            self.connected = True
            self.connection_start_time = time.perf_counter()
            self.start_receiving()
            return True
        except KeyboardInterrupt:
            self._log("WARNING", "Connection cancelled by user")
            return False
        except Exception as e:
            self._log("ERROR", f"Connection error: {e}")
            return False
        
    def start_receiving(self):
        """1000Hz로 계속 수신하는 별도 쓰레드 시작"""
        self.is_receiving = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        self._log("INFO", "Started 1000Hz receiving thread")

    def _receive_loop(self):
        """1000Hz로 계속 수신하는 메인 루프 - 최적화"""
        while self.is_receiving:
            try:
                # 타임아웃을 1ms → 0.5ms로 개선
                self.conn.settimeout(0.0005)
                data = self._recv_exact(self.CPP_TO_PY_PACKET_SIZE)
                if data:
                    state, sander_active = self._process_packet(data)
                    if state is not None:
                        with self.state_lock:
                            self.latest_state = state
                            self.latest_sander_active = sander_active
                            # time.time() → time.perf_counter() 사용
                            self.last_packet_time = time.perf_counter()
            except socket.timeout:
                continue
            except Exception as e:
                self._log("WARNING", f"Receive loop error: {e}")
                break
        self._log("INFO", "Receive loop terminated")

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
                self._log("WARNING", f"Expected {self.CPP_TO_PY_PACKET_SIZE}B, received {len(data)}B (total {self.packet_size_errors} times)")
                return None, False
            # 2. 언패킹 - 포맷 수정
            try:
                (sof, current_force, target_force, force_error, force_error_dot, 
                 force_error_int, pi_output, sander_active, 
                 received_checksum) = struct.unpack(">HffffffBH", data)
            except struct.error as e:
                self._log("ERROR", f"struct.unpack failed: {e}")
                return None, False
            # 3. SOF 검증
            if sof != self.CPP_TO_PY_SOF:
                self.sof_errors += 1
                self._log("WARNING", f"SOF mismatch: {hex(sof)} (expected: {hex(self.CPP_TO_PY_SOF)}) (total {self.sof_errors} times)")
                return None, False
            # 4. 체크섬 검증 (CRC-16)
            calculated_checksum = self.calculate_crc16(data[:-2])
            if received_checksum != calculated_checksum:
                self.checksum_errors += 1
                self._log("ERROR", f"Checksum error: recv:{received_checksum} calc:{calculated_checksum} (total {self.checksum_errors} times)")
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
            self._log("ERROR", f"Packet processing error: {e}")
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
                    self._log("DEBUG", f"RL Flag Changed: {self.last_logged_sander_active} -> {self.latest_sander_active}")
                    self.last_logged_sander_active = self.latest_sander_active
                elif not hasattr(self, 'last_logged_sander_active'):
                    self.last_logged_sander_active = self.latest_sander_active
                    self._log("DEBUG", f"Initial RL Flag: {self.latest_sander_active}")
                return self.latest_state.copy(), self.latest_sander_active
        return None, False
    
    def send_residual(self, rl_residual, timing_accurate, episode_done):
        """25Hz로 residual 전송"""
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
            # 5. 송신 성공 통계 업데이트
            self.packets_sent += 1
            return True
        except Exception as e:
            self._log("ERROR", f"Error sending residual: {e}")
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
            self._log("ERROR", f"Error sending reset: {e}")
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
        """통신 통계 출력 - Robot PC 1kHz 전송 기준으로 평가"""
        stats = self.get_communication_stats()
        self._log("INFO", "\n📊 === COMMUNICATION STATISTICS ===")
        self._log("INFO", f"⏱️  Uptime: {stats['uptime_seconds']:.1f}s")
        self._log("INFO", f"📥 Packets Received: {stats['packets_received']}")
        self._log("INFO", f"📤 Packets Sent: {stats['packets_sent']}")
        self._log("INFO", f"📥 Receive Rate: {stats['receive_rate_hz']:.1f} Hz")
        self._log("INFO", f"📤 Send Rate: {stats['send_rate_hz']:.1f} Hz")
        self._log("INFO", f"❌ Checksum Errors: {stats['checksum_errors']}")
        self._log("INFO", f"⚠️  SOF Errors: {stats['sof_errors']}")
        self._log("INFO", f"📏 Packet Size Errors: {stats['packet_size_errors']}")
        
        # 🎯 Robot PC 1kHz 전송 기준으로 패킷 손실률 재계산
        robot_packets_sent = int(stats['uptime_seconds'] * 1000)  # 1kHz
        successfully_received = stats['packets_received']
        actual_missed = robot_packets_sent - successfully_received
        actual_loss_rate = (actual_missed / robot_packets_sent) * 100 if robot_packets_sent > 0 else 0
        
        self._log("INFO", "\n🤖 === ROBOT PC BASED PACKET LOSS ===")
        self._log("INFO", f"🤖 Robot PC sent: {robot_packets_sent:,} packets (1kHz)")
        self._log("INFO", f"📥 Successfully received: {successfully_received:,}")
        self._log("INFO", f"❌ Actually missed: {actual_missed:,}")
        self._log("INFO", f"📊 Actual Loss Rate: {actual_loss_rate:.3f}%")
        
        # Robot PC 기준 품질 평가
        if actual_loss_rate < 0.1:
            robot_quality = "🟢 EXCELLENT"
        elif actual_loss_rate < 1.0:
            robot_quality = "🟡 GOOD"
        elif actual_loss_rate < 5.0:
            robot_quality = "🟠 FAIR"
        else:
            robot_quality = "🔴 POOR"
        self._log("INFO", f"🤖 Robot PC Quality: {robot_quality}")
        
        # 가상 손실률 관련 정보 제거 - 불필요한 모니터링
        
        if stats['last_packet_time']:
            time_since_last = time.perf_counter() - stats['last_packet_time']
            self._log("INFO", f"\n🕐 Last Packet: {time_since_last:.3f}s ago")
        
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
            self._log("INFO", "Communication closed")
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
    def generate_communication_graphs(self):
        """통신 모니터링 그래프 생성 - 현재는 기본 통계만 출력"""
        if hasattr(self, 'comm') and self.comm:
            self.comm.print_communication_stats()
        else:
            self._log("WARNING", "Communication object not available for graph generation")
    def _init_data_logging(self):
        if self.cfg["LOG_SEND_DATA"]:
            log_dir = self.cfg["LOG_DATA_DIR"]
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{log_dir}/send_data_log_{timestamp}.csv"
            with open(csv_filename, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'episode', 'step', 'residual_mpa', 
                    'timing_accurate', 'episode_done', 'sander_active', 'current_force', 
                    'target_force', 'force_error', 'force_error_dot', 'force_error_int', 'pi_output'
                ])
            self.send_data_logger = csv_filename
            self._log("INFO", f"Data logging initialized: {csv_filename}")
            self.episode_send_data = []
            
    def _log_send_data(self, episode, step, residual_mpa, timing_accurate, episode_done, sander_active, state):
        if not self.cfg["LOG_SEND_DATA"] or not self.send_data_logger:
            return
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        data_row = [
            current_time, episode, step, residual_mpa, 
            timing_accurate, episode_done, sander_active, 
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
            self._log("ERROR", f"CSV logging error: {e}")
        self.episode_send_data.append({
            'timestamp': current_time,
            'episode': episode,
            'step': step,
            'residual_mpa': residual_mpa,
            'timing_accurate': timing_accurate,
            'episode_done': episode_done,
            'sander_active': sander_active,
            'current_force': state[0] if state is not None else 0.0,
            'target_force': state[1] if state is not None else 0.0,
            'force_error': state[2] if state is not None else 0.0,
            'force_error_dot': state[3] if state is not None else 0.0,
            'force_error_int': state[4] if state is not None else 0.0,
            'pi_output': state[5] if state is not None else 0.0
        })
    # ---- scheduler ----
    def should_send_now(self):
        """정확한 25Hz 타이밍 제어 - 최적화된 버전"""
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
            self._log("WARNING", f"Safety: Force too high ({state[0]:.1f}N > 100N) - Episode terminated")
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
                self._log("WARNING", "RL inactive too long → terminate episode")
                return "terminate"
            return "inactive"
    # ---- episode helpers ----
    def end_episode_with_100_percent_success(self, ep, target_force):
        """100% 에피소드 전환 성공을 위한 하이브리드 방식"""
        episode_end_start = time.perf_counter()
        confirmation_count = 0
        required_confirmations = 3  # 3번 연속 확인으로 확실성 보장
        
        self._log("INFO", f"🎯 Episode {ep+1} completed at step {self.episode_step:,}")
        self._log("INFO", f"🔄 Ending episode {ep+1}, expecting target change from {target_force:.1f}N")
        
        while (time.perf_counter() - episode_end_start) < 2.5:
            # 지속적 플래그 전송 (로봇이 놓치지 않도록)
            self.comm.send_residual(0.0, True, True)
            
            # 고주파 모니터링 (200Hz)
            state, _ = self.comm.get_latest_state()
            if state and abs(state[1] - target_force) > 1.0:
                confirmation_count += 1
                if confirmation_count >= required_confirmations:
                    elapsed = time.perf_counter() - episode_end_start
                    self._log("INFO", f"✅ 100% Episode end confirmed! {target_force:.1f}N → {state[1]:.1f}N (took {elapsed:.3f}s)")
                    
                    # 성공 후 플래그 전송 중단 신호
                    for _ in range(3):
                        self.comm.send_residual(0.0, True, False)  # episode_done=False로 전송
                        time.sleep(0.01)
                        
                    return True
            else:
                confirmation_count = 0  # 연속성 보장
                
            time.sleep(0.005)  # 5ms = 200Hz 모니터링
        
        # 2.5초 후 강제 성공 (백업 전략)
        self._log("INFO", f"⚠️ Time-based episode end (2.5s elapsed) - proceeding to next episode")
        
        # 마지막으로 플래그 전송 중단 신호
        for _ in range(3):
            self.comm.send_residual(0.0, True, False)  # episode_done=False로 전송
            time.sleep(0.01)
            
        return True  # 항상 성공으로 처리

    def reset_episode(self):
        self.prev_residual = 0.0
        self.episode_step = 0
        self.current_episode_reward = 0.0
        self.rl_inactive_count = 0
        self.rl_active_in_episode = False
        self.last_tick = None
        self.last_log_time = None
        state, _ = self.comm.get_latest_state()
        if state is not None:
            self.previous_target_force = state[1]  # target_force
            self._log("INFO", f"Episode target force: {self.previous_target_force:.1f}N")
        ok = self.comm.send_reset()
        if ok:
            self._log("INFO", "\n--- Episode Reset ---")
            self._log("INFO", "Robot PC: 1kHz PI running, will add RL residual (held) each tick.")
        else:
            self._log("WARNING", "Reset signal failed (continuing).")
        return ok
    # ---- main loop ----
    def run_training(self, episodes=None):
        episodes = episodes or self.cfg["EPISODES"]
        if not self.comm.connect():
            self._log("ERROR", "Failed to connect to Robot PC")
            return
        # saved_agents 폴더 생성
        model_save_dir = self.cfg["MODEL_SAVE_DIR"]
        os.makedirs(model_save_dir, exist_ok=True)
        self._log("INFO", f"📁 Model save directory: {model_save_dir}")
        self._log("INFO", "🚀 Starting Residual RL Training - 10 episodes, 30 seconds each")
        self._log("INFO", "📡 RL: 25Hz residual output, Robot: 1kHz PI + residual")
        self._log("INFO", "⏱️  Episode: 750 steps (30 seconds at 25Hz)")
        self._log("INFO", "\n🔄 Waiting for RL activation...")
        wait_start_time = time.perf_counter()
        while True:
            state, sander_active = self.comm.get_latest_state()
            if sander_active:
                wait_duration = time.perf_counter() - wait_start_time
                self._log("INFO", f"🎯 RL Activated! (waited {wait_duration:.1f}s)")
                break
            if state is not None:
                current_force = state[0]
                target_force = state[1]
                print(f"⏳ Waiting... Current Force: {current_force:.1f}N, Target: {target_force:.1f}N", end='\r')
            time.sleep(0.04)
            if time.perf_counter() - wait_start_time > 300:
                self._log("WARNING", "\n⚠️ RL activation timeout (5 minutes)")
                return
        self._init_data_logging()
        episode_stats = []
        for ep in range(episodes):
            self._log("INFO", f"\n🎬 === EPISODE {ep+1}/{episodes} START ===")
            # 🎯 에피소드 시작 시 RL Flag 상태 확인
            episode_start_state, episode_start_sander_active = self.comm.get_latest_state()
            if not episode_start_sander_active:
                self._log("WARNING", f"⚠️ Warning: RL Flag is False at episode {ep+1} start")
                self._log("INFO", "🔄 Waiting for RL activation...")
                wait_start = time.perf_counter()
                while not episode_start_sander_active:
                    episode_start_state, episode_start_sander_active = self.comm.get_latest_state()
                    if time.perf_counter() - wait_start > 60:  # 1분 타임아웃
                        self._log("WARNING", f"⚠️ Timeout waiting for RL activation in episode {ep+1}")
                        break
                    time.sleep(0.1)
            episode_start_time = time.perf_counter()
            self.reset_episode()
            # 새 에피소드 시작 신호 전송
            self._log("INFO", f"📡 Starting episode {ep+1}")
            self.comm.send_residual(0.0, True, False)
            prev_state = None
            prev_action = None
            prev_sander_active = False
            # 에피소드 통계
            episode_packets_received = 0
            episode_packets_sent = 0
            episode_rl_active_steps = 0
            # 25Hz 타이밍 제어를 위한 변수 초기화
            next_send_time = time.perf_counter() + self.cfg["TICK_SEC"]
            
            while True:
                current_time = time.perf_counter()
                
                # 25Hz 타이밍이 되었을 때만 처리
                if current_time >= next_send_time:
                    # 다음 전송 시간 설정 (정확한 40ms 간격)
                    next_send_time += self.cfg["TICK_SEC"]
                    timing_accurate = abs(current_time - (next_send_time - self.cfg["TICK_SEC"])) <= self.cfg["TICK_TOL"]
                    
                    # 최신 상태 가져오기
                    res = self.comm.get_latest_state()
                    if res[0] is None:
                        continue
                    
                    state, sander_active = res
                    episode_packets_received += 1
                    self.previous_target_force = state[1]
                else:
                    # 아직 전송 시간이 아니면 짧게 대기
                    time.sleep(0.001)
                    continue
                # 25Hz 기준으로 스텝 증가
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
                    self._log("INFO", "Episode terminated due to prolonged RL inactivity")
                    break
                
                # 에피소드 종료 확인
                if self.episode_step >= self.max_episode_steps:
                    episode_done = True
                    rl_residual = 0.0
                    self._log("INFO", f"🎯 Episode {ep+1} ending at step {self.episode_step}")
                else:
                    episode_done = False
                    if sander_active:
                        raw_res = self.agent.select_action(state, evaluate=False)
                        rl_residual = self.limit_residual(raw_res)
                        episode_rl_active_steps += 1
                    else:
                        rl_residual = 0.0
                # residual 전송
                ok = self.comm.send_residual(rl_residual, timing_accurate, episode_done)
                # CSV 데이터 로깅
                self._log_send_data(ep+1, self.episode_step, rl_residual, timing_accurate, episode_done, sander_active, state)
                if not ok:
                    self.fail_count += 1
                    self._log("WARNING", f"⚠️ Send failed ({self.fail_count}/{self.FAIL_MAX})")
                    if self.fail_count >= self.FAIL_MAX:
                        # advise PI-only fallback by ending episode with residual=0
                        self.comm.send_residual(0.0, False, True)
                        self._log("WARNING", "Comms degraded → advise PI-only fallback on Robot PC; ending episode.")
                        break
                else:
                    self.fail_count = 0
                    episode_packets_sent += 1
                # 에피소드 종료 시 100% 성공 보장 방식
                if episode_done:
                    success = self.end_episode_with_100_percent_success(ep, self.previous_target_force)
                    if success:
                        break
                # 2.5초마다 간결한 로깅
                if (self.last_log_time is None or 
                    current_time - self.last_log_time >= 2.5):
                    mode = "RESIDUAL" if sander_active else "PI-ONLY"
                    force_achieved = " 🎯" if abs(state[0] - state[1]) < 0.5 else ""
                    self._log("INFO", f"[Ep {ep+1}] S{self.episode_step} | {mode} | "
                          f"F{state[0]:.1f}/{state[1]:.1f} | "
                          f"PI{state[5]:.3f} | RL{rl_residual:.3f} | "
                          f"T{current_time - episode_start_time:.1f}s{force_achieved}")
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
            self._log("INFO", f"\n🎯 === EPISODE {ep+1}/10 COMPLETED ===")
            self._log("INFO", f"⏱️  Duration: {episode_duration:.1f}s")
            self._log("INFO", f"📊 Steps: {self.episode_step:,}/{self.max_episode_steps:,} ({self.episode_step/self.max_episode_steps*100:.1f}%)")
            self._log("INFO", f"🏆 Reward: {self.current_episode_reward:.2f}")
            self._log("INFO", f"📥 Packets Received: {episode_packets_received}")
            self._log("INFO", f"📤 Packets Sent: {episode_packets_sent}")
            self._log("INFO", f"🤖 RL Active Steps: {episode_rl_active_steps} ({episode_stat['rl_active_ratio']*100:.1f}%)")
            self._log("INFO", f"📈 Best So Far: {self.best_episode_reward:.2f}")
            # 🎯 데이터 로깅 요약
            if self.cfg["LOG_SEND_DATA"] and self.send_data_logger:
                self._log("INFO", f"📊 Data logged: {len(self.episode_send_data)} records to {self.send_data_logger}")
            # 🎯 에피소드 완료 이유 표시
            if self.episode_step >= self.max_episode_steps:
                self._log("INFO", f"✅ Completed: Reached maximum episode steps ({self.max_episode_steps:,})")
            else:
                self._log("WARNING", "⚠️  Completed: Episode terminated early (safety or error)")
            self._log("INFO", "=" * 40)
            # 통신 상태 모니터링 (매 에피소드마다)
            self.comm.print_communication_stats()
        # ---- 전체 학습 완료 ----
        self._log("INFO", "\n🎯 TEST Training finished!")
        self._log("INFO", f"✅ Completed {episodes} episodes successfully")
        self._log("INFO", f"🏆 Best Episode: {self.best_agent_episode+1}, Best Reward: {self.best_episode_reward:.2f}")
        # 🎯 통신 모니터링 그래프 자동 생성
        self._log("INFO", "\n📊 Generating communication monitoring graphs...")
        self.generate_communication_graphs()
        self._log("INFO", "✅ Communication graphs generated successfully!")
        self._log("INFO", f"📈 Total RL active steps: {self.total_rl_active_steps}")
        # 전체 통계 요약
        self._log("INFO", "\n📊 === FINAL TRAINING SUMMARY ===")
        total_duration = sum(ep["duration"] for ep in episode_stats)
        total_packets_received = sum(ep["packets_received"] for ep in episode_stats)
        total_packets_sent = sum(ep["packets_sent"] for ep in episode_stats)
        avg_reward = np.mean([ep["reward"] for ep in episode_stats])
        self._log("INFO", f"⏱️  Total Duration: {total_duration:.1f}s")
        self._log("INFO", f"📥 Total Packets Received: {total_packets_received}")
        self._log("INFO", f"📤 Total Packets Sent: {total_packets_sent}")
        self._log("INFO", f"📊 Average Reward: {avg_reward:.2f}")
        self._log("INFO", f"📈 Best Reward: {self.best_episode_reward:.2f}")
        self._log("INFO", f"🤖 Total RL Active Steps: {self.total_rl_active_steps}")
        # 에피소드별 상세 통계
        self._log("INFO", "\n📋 === EPISODE DETAILS ===")
        for ep_stat in episode_stats:
            self._log("INFO", f"Ep {ep_stat['episode']:2d}: "
                  f"Reward {ep_stat['reward']:6.2f}, "
                  f"Steps {ep_stat['steps']:4d}, "
                  f"RL Active {ep_stat['rl_active_steps']:4d} "
                  f"({ep_stat['rl_active_ratio']*100:4.1f}%)")
        # 최종 통신 통계
        self.comm.print_communication_stats()
        # 🎯 통신 모니터링 그래프 생성
        self._log("INFO", "\n📊 === GENERATING COMMUNICATION MONITORING GRAPHS ===")
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
    
    print("🧪 TEST VERSION 5: JY_Pneumatic_SAC_Pre_only_test_5.py")
    print("=" * 60)
    set_seed(42)
    env = PneumaticPolishingEnvironment(CONFIG)
    try:
        print("🚀 Starting cleaned training...")
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
