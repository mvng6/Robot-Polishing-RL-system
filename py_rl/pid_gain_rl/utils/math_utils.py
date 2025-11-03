"""
수학 유틸리티 - PID 액션 스케일링 및 상태 생성
"""
import numpy as np
from typing import List, Optional, Dict, Any

from ..constants import Constants


def scale_action_to_pid(action, pid_range):
    """
    액션을 PID 게인으로 스케일링 (벡터화)
    - 내부 액션 a ∈ [-1, 1]^3 → 실제 PID
    - Kp, Ki: 선형 매핑(소수점 2자리 반올림)
    - Kd: 로그 스케일 매핑(소수점 6자리)로 극소 범위 해상도 확보 (예: [1e-6, 1e-3])
    """
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    # 선형 매핑: Kp, Ki
    kp_lo, kp_hi = pid_range["Kp"]
    ki_lo, ki_hi = pid_range["Ki"]
    kp = kp_lo + (a[0] + 1.0) * 0.5 * (kp_hi - kp_lo)
    ki = ki_lo + (a[1] + 1.0) * 0.5 * (ki_hi - ki_lo)
    # 로그 매핑: Kd
    kd_lo, kd_hi = pid_range["Kd"]
    kd_lo_safe = max(kd_lo, 1e-12)
    kd_hi_safe = max(kd_hi, kd_lo_safe * 10.0)
    loL, hiL = np.log10(kd_lo_safe), np.log10(kd_hi_safe)
    kdL = loL + (a[2] + 1.0) * 0.5 * (hiL - loL)
    kd = float(10 ** kdL)
    pid_gains = np.array([kp, ki, kd], dtype=np.float32)
    
    # P, I: 소수점 2자리 반올림
    pid_gains[0] = round(pid_gains[0], 2)  # Kp
    pid_gains[1] = round(pid_gains[1], 2)  # Ki
    pid_gains[2] = round(pid_gains[2], 6)  # Kd - 로그 스케일 해상도 유지
    
    return pid_gains


def create_initial_state(
    force_data, 
    target_force, 
    prev_pid_gains=None, 
    episode_history=None, 
    dt_sec=0.001
):
    """
    초기 상태 벡터 생성 (20차원)
    
    기존 12차원 (로봇PC 6 + 강화학습PC 6) + 궤적 요약 8차원
    
    Args:
        force_data: 힘 데이터 리스트
        target_force: 목표 힘
        prev_pid_gains: 이전 PID 게인 [Kp, Ki, Kd]
        episode_history: 에피소드 히스토리 (미사용)
        dt_sec: 샘플링 시간 (미사용)
    
    Returns:
        state: 20차원 numpy array
    """
    import numpy as np
    
    # 1. 기존 12차원
    if not force_data:
        force_data = [target_force]
    
    force_arr = np.array(force_data, dtype=np.float64)
    
    # 기본 6차원 (로봇PC 전송 예정)
    base_state = [
        prev_pid_gains[0] if prev_pid_gains is not None else 40.0,  # Kp
        prev_pid_gains[1] if prev_pid_gains is not None else 50.0,  # Ki
        prev_pid_gains[2] if prev_pid_gains is not None else 0.0,   # Kd
        0.0,  # prev_reward (초기값)
        target_force,
        10.0,  # episode_seconds (기본값)
    ]
    
    # 강화학습PC 계산 6차원
    if len(force_arr) >= 10:
        errors = force_arr - target_force
        base_stats = [
            float(np.mean(force_arr)),
            float(np.std(force_arr)),
            float(np.min(force_arr)),
            float(np.max(force_arr)),
            float(np.mean(errors)),
            float(np.std(errors)),
        ]
    else:
        base_stats = [target_force, 0.0, target_force, target_force, 0.0, 0.0]
    
    base_state.extend(base_stats)  # 12차원 완성
    
    # 2. 🆕 궤적 요약 8차원 (초기값은 모두 0)
    # 에피소드 시작 시점이므로 궤적 정보 없음
    trajectory_features = [
        0.0,  # overshoot
        0.0,  # settling_time
        0.0,  # rmse
        0.0,  # band_ratio
        0.0,  # oscillation_freq
        0.0,  # oscillation_amp
        0.0,  # rise_time
        0.0,  # steady_state_error
    ]
    
    state = np.array(base_state + trajectory_features, dtype=np.float32)
    
    # NaN/Inf 가드
    if np.isnan(state).any() or np.isinf(state).any():
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    
    return state  # 20차원 반환

# 주의: _estimate_state_from_previous_pid() 함수는 삭제되었습니다.
# 이 함수는 experiment 폴더의 레거시 코드에서만 사용되며,
# 현재 모듈화된 코드에서는 create_initial_state()를 사용합니다.

