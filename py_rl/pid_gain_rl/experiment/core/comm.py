"""
로봇/센서 통신 래퍼
"""
import socket
import struct
import threading
import time
import numpy as np
from collections import deque
from ..utils.loggers.base_logger import AppLogger

class PIDGainCommunicator:
    """
    로봇 제어 PC와의 TCP 통신 관리
    - PID gain 전송 (에피소드당 한 번)
    - 실시간 상태 데이터 수신 (1kHz)
    - 연결 상태 모니터링
    """

    def __init__(
        self, host, port, recv_timeout, recv_loop_timeout=0.05, cfg=None
    ):
        self.host, self.port = host, port
        self.recv_timeout = recv_timeout
        self.recv_loop_timeout = recv_loop_timeout
        self.cfg = cfg
        self.socket = None
        self.conn = None
        self.connected = False

        # SOF, current_force, target_force, error, edot, eint, pi_output, sander_active,
        # precharge_applied, J3, prep_flag, checksum
        self.CPP_TO_PY_PACKET_FORMAT = ">HffffffBffBH"
        self.CPP_TO_PY_PACKET_SIZE = struct.calcsize(
            self.CPP_TO_PY_PACKET_FORMAT
        )
        self.CPP_TO_PY_SOF = 0xAAAA
        # self.PY_TO_CPP_PACKET_FORMAT = ">HfBBBH"  # SOF, rl_residual,
        # timing_accurate, episode_done, learning_done, checksum (미사용)
        # self.PY_TO_CPP_PACKET_SIZE = 11  # SOF(2) + rl_residual(4) +
        # timing_accurate(1) + episode_done(1) + learning_done(1) +
        # checksum(2) = 11 bytes (미사용)
        # self.PY_TO_CPP_SOF = 0xBBBB  # (미사용)

        # PID gain 전송용 패킷 포맷
        # SOF, precharge, Kp, Ki, Kd, timing_accurate, episode_done, learning_done, checksum
        self.PID_PACKET_FORMAT = ">HffffBBBH"
        # SOF(2) + precharge(4) + Kp(4) + Ki(4) + Kd(4) + timing(1) + ep_done(1) +
        # learn_done(1) + checksum(2) = 23 bytes
        self.PID_PACKET_SIZE = 23
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
        # 준비 구간 힘 평균 계산용
        self.prep_force_deque = deque(
            maxlen=int(3.0 / self.cfg["RECV_INTERVAL_SEC"])
            if self.cfg and "RECV_INTERVAL_SEC" in self.cfg and self.cfg["RECV_INTERVAL_SEC"] > 0
            else 3000
        )
        self.last_prep_force_avg = 0.0
        self.prev_sander_active = False
        self.prev_prep_flag = False

    def _log(self, level, message):
        AppLogger.log(level, message)

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.settimeout(1.0)
            self._log(
                "INFO", f"로봇제어PC 연결 대기 중 {self.host}:{self.port} ..."
            )
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
        self.receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True
        )
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
                    if data is None:
                        self.consecutive_failures += 1
                        self._log(
                            "WARNING",
                            f"수신 데이터 없음/연결 종료 감지 "
                            f"({self.consecutive_failures}회)",
                        )
                        if self.consecutive_failures >= 5:
                            self._log("ERROR", "연속 수신 실패로 수신 루프 중단")
                            break
                        time.sleep(self.cfg["RECV_INTERVAL_SEC"])
                        continue
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
                    self._log(
                        "WARNING",
                        f"수신 루프 오류 ({self.consecutive_failures}회): {e}",
                    )
                    if self.consecutive_failures >= 5:
                        self._log("ERROR", "연속 수신 실패로 수신 루프 중단")
                        break
                    time.sleep(self.cfg["RECV_INTERVAL_SEC"])
            else:
                time.sleep(0.001)
        self._log("INFO", "수신 루프 종료")

    def _recv_exact(self, nbytes):
        data = b""
        while len(data) < nbytes:
            chunk = self.conn.recv(nbytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _process_packet(self, data):
        try:
            if len(data) != self.CPP_TO_PY_PACKET_SIZE:
                self._log(
                    "WARNING",
                    f"예상 {self.CPP_TO_PY_PACKET_SIZE}B, 수신 {len(data)}B",
                )
                return None, False
            try:
                (
                    sof,
                    current_force,
                    target_force,
                    force_error,
                    force_error_dot,
                    force_error_int,
                    pi_output,
                    sander_active,
                    precharge_applied,
                    j3_prep,
                    prep_flag,
                    received_checksum,
                ) = struct.unpack(self.CPP_TO_PY_PACKET_FORMAT, data)
            except struct.error as e:
                self._log("ERROR", f"패킷 언팩 실패: {e}")
                return None, False
            if sof != self.CPP_TO_PY_SOF:
                self._log(
                    "WARNING",
                    f"SOF 불일치: {hex(sof)} (예상: {hex(self.CPP_TO_PY_SOF)})",
                )
                return None, False
            calculated_checksum = self.calculate_crc16(data[:-2])
            if received_checksum != calculated_checksum:
                self._log(
                    "ERROR",
                    f"체크섬 오류: 수신:{received_checksum} 계산:{calculated_checksum}",
                )
                return None, False
            state = np.array(
                [
                    current_force,  # 0
                    target_force,  # 1
                    force_error,  # 2
                    force_error_dot,  # 3
                    force_error_int,  # 4
                    pi_output,  # 5
                    precharge_applied,  # 6
                    j3_prep,  # 7
                    0.0,  # prep_force_avg placeholder (8)
                    prep_flag,  # 9 (uint8 -> float)
                ],
                dtype=np.float32,
            )
            sander_active = bool(sander_active)
            prep_flag_bool = bool(prep_flag)

            # 준비 구간 힘 평균 계산 (prep_flag ON 동안 누적, OFF 또는 상승 시 고정)
            if prep_flag_bool and not sander_active:
                self.prep_force_deque.append(current_force)
                if len(self.prep_force_deque) > 0:
                    self.last_prep_force_avg = float(
                        sum(self.prep_force_deque) / len(self.prep_force_deque)
                    )
                state[8] = self.last_prep_force_avg
            else:
                if self.prev_prep_flag and not prep_flag_bool:
                    # 준비 플래그가 꺼지면 평균 고정
                    state[8] = self.last_prep_force_avg
                elif not self.prev_sander_active and sander_active:
                    # sander_active 상승 에지에서 평균 고정
                    state[8] = self.last_prep_force_avg
                else:
                    # 이후에는 고정값 유지
                    state[8] = self.last_prep_force_avg

            # 준비 플래그 상태를 상태 벡터에 반영 (0/1)
            state[9] = 1.0 if prep_flag_bool else 0.0

            self.prev_sander_active = sander_active
            self.prev_prep_flag = prep_flag_bool

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
                if (
                    self.last_packet_time
                    and current_time - self.last_packet_time > 2.0
                ):
                    # 오래된 데이터 경고는 한 번만 출력
                    if not self.old_data_warning_logged:
                        self._log(
                            "WARNING",
                            f"오래된 데이터 감지: {current_time - self.last_packet_time:.2f}초 전",
                        )
                        self.old_data_warning_logged = True
                else:
                    # 데이터가 정상이면 경고 플래그 리셋
                    self.old_data_warning_logged = False

                if (
                    hasattr(self, "last_logged_sander_active")
                    and self.last_logged_sander_active
                    != self.latest_sander_active
                ):
                    self._log(
                        "DEBUG",
                        f"RL 플래그 변경: {self.last_logged_sander_active} -> "
                        f"{self.latest_sander_active}",
                    )
                    self.last_logged_sander_active = self.latest_sander_active
                elif not hasattr(self, "last_logged_sander_active"):
                    self.last_logged_sander_active = self.latest_sander_active
                    self._log(
                        "DEBUG", f"초기 RL 플래그: {self.latest_sander_active}"
                    )
                return self.latest_state.copy(), self.latest_sander_active
        return None, False

    def send_pid_once(
        self,
        kp,
        ki,
        kd,
        precharge=0.0,
        timing_accurate=True,
        episode_done=False,
        learning_done=False,
    ):
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
            # 프리차지 값은 소수점 3째자리에서 고정 (통신/로그 일관성)
            precharge = round(float(precharge), 3)
            payload = struct.pack(
                ">HffffBBB",
                self.PID_SOF,
                precharge,
                float(kp),
                float(ki),
                float(kd),
                bool(timing_accurate),
                bool(episode_done),
                bool(learning_done),
            )
            checksum = self.calculate_crc16(payload)
            final_packet = struct.pack(
                self.PID_PACKET_FORMAT,
                self.PID_SOF,
                float(precharge),
                float(kp),
                float(ki),
                float(kd),
                bool(timing_accurate),
                bool(episode_done),
                bool(learning_done),
                checksum,
            )
            self.conn.sendall(final_packet)
            with self.stats_lock:
                self.packets_sent += 1
            self._log(
                "INFO",
                f"📡 PID/Precharge 전송: precharge={precharge:.3f}MPa, "
                f"Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}",
            )
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
        uptime = (
            time.perf_counter() - self.connection_start_time
            if self.connection_start_time
            else 0
        )
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
