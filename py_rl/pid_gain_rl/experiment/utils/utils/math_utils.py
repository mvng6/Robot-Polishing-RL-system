"""
수학 유틸리티 - PID 액션 스케일링 및 상태 생성
"""
import numpy as np
from typing import List, Optional, Dict, Any

from ...config.constants import Constants


def scale_action_to_control(action, pid_range, precharge_range):
    """
    액션을 프리차지 공압 + PID 게인으로 스케일링
    - action: [-1, 1]^4 -> [precharge, Kp, Ki, Kd]
    """
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    # 프리차지 (선형 매핑)
    p_lo, p_hi = precharge_range
    precharge = p_lo + (a[0] + 1.0) * 0.5 * (p_hi - p_lo)
    precharge = float(np.round(np.clip(precharge, p_lo, p_hi), 3))  # 소수 3자리까지

    # PID 매핑
    kp_lo, kp_hi = pid_range["Kp"]
    ki_lo, ki_hi = pid_range["Ki"]
    kp = kp_lo + (a[1] + 1.0) * 0.5 * (kp_hi - kp_lo)
    ki = ki_lo + (a[2] + 1.0) * 0.5 * (ki_hi - ki_lo)
    kd_lo, kd_hi = pid_range["Kd"]
    pid_gains = np.array([kp, ki, 0.0], dtype=np.float32)

    use_linear_kd = (kd_lo <= 0.0) or (kd_hi <= 0.03)
    if use_linear_kd:
        kd = (a[3] + 1.0) * 0.5 * (kd_hi - max(kd_lo, 0.0))
    else:
        kd_lo_safe = 1e-6 if kd_lo <= 0.0 else kd_lo
        kd_hi_safe = max(kd_hi, kd_lo_safe * 10.0)
        loL, hiL = np.log10(kd_lo_safe), np.log10(kd_hi_safe)
        kdL = loL + (a[3] + 1.0) * 0.5 * (hiL - loL)
        kd = float(10 ** kdL)
    pid_gains[2] = kd

    def quantize(value, lo, hi, step=0.01, decimals=2):
        value = np.clip(value, lo, hi)
        quantized = np.round(value / step) * step
        quantized = np.clip(quantized, lo, hi)
        return float(np.round(quantized, decimals))

    pid_gains[0] = quantize(pid_gains[0], kp_lo, kp_hi)
    pid_gains[1] = quantize(pid_gains[1], ki_lo, ki_hi)
    kd_step = 0.001 if kd_hi <= 0.03 else 0.01
    pid_gains[2] = quantize(pid_gains[2], max(kd_lo, 0.0), kd_hi, step=kd_step, decimals=3)

    return precharge, pid_gains


def scale_action_to_pid(action, pid_range):
    """
    액션을 PID 게인으로 스케일링 (벡터화)
    - 내부 액션 a ∈ [-1, 1]^3 → 실제 PID
    - Kp, Ki, Kd: 각각 0.01 단위로 양자화하여 소수점 둘째 자리까지 사용
    """
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    # 선형 매핑: Kp, Ki
    kp_lo, kp_hi = pid_range["Kp"]
    ki_lo, ki_hi = pid_range["Ki"]
    kp = kp_lo + (a[0] + 1.0) * 0.5 * (kp_hi - kp_lo)
    ki = ki_lo + (a[1] + 1.0) * 0.5 * (ki_hi - ki_lo)
    # Kd 매핑: 작은 범위(≤0.02)나 하한이 0이면 선형 매핑으로 안정화, 그 외 로그 매핑
    kd_lo, kd_hi = pid_range["Kd"]
    pid_gains = np.array([kp, ki, 0.0], dtype=np.float32)

    use_linear_kd = (kd_lo <= 0.0) or (kd_hi <= 0.03)
    if use_linear_kd:
        kd = (a[2] + 1.0) * 0.5 * (kd_hi - max(kd_lo, 0.0))
    else:
        kd_lo_safe = 1e-6 if kd_lo <= 0.0 else kd_lo
        kd_hi_safe = max(kd_hi, kd_lo_safe * 10.0)
        loL, hiL = np.log10(kd_lo_safe), np.log10(kd_hi_safe)
        kdL = loL + (a[2] + 1.0) * 0.5 * (hiL - loL)
        kd = float(10 ** kdL)
    pid_gains[2] = kd

    def quantize(value, lo, hi, step=0.01, decimals=2):
        value = np.clip(value, lo, hi)
        quantized = np.round(value / step) * step
        quantized = np.clip(quantized, lo, hi)
        return float(np.round(quantized, decimals))

    pid_gains[0] = quantize(pid_gains[0], kp_lo, kp_hi)
    pid_gains[1] = quantize(pid_gains[1], ki_lo, ki_hi)
    kd_step = 0.001 if kd_hi <= 0.03 else 0.01
    pid_gains[2] = quantize(pid_gains[2], max(kd_lo, 0.0), kd_hi, step=kd_step, decimals=3)
    
    return pid_gains


def create_initial_state(
    force_data,
    target_force,
    prev_pid_gains=None,
    episode_history=None,
    dt_sec=0.001,
):
    """
    초기 상태 벡터 생성 (STATE_DIM 차원 = 10)

    Args:
        force_data: 힘 데이터 리스트(선택사항)
        target_force: 목표 힘
        prev_pid_gains: 미사용 (호환성 유지용)
        episode_history: 미사용 (호환성 유지용)
        dt_sec: 미사용 (호환성 유지용)

    Returns:
        state: [현재힘, 목표힘, 힘 오차, 오차 미분, 오차 적분, PI 출력]
    """
    import numpy as np

    if force_data:
        current_force = float(force_data[-1])
    else:
        current_force = float(Constants.INITIAL_CONTACT_FORCE)

    target_force_value = float(target_force)
    error = current_force - target_force_value
    error_dot = 0.0
    error_int = 0.0
    pi_output = float(Constants.INITIAL_PI_OUTPUT)

    base_state = [
        current_force,
        target_force_value,
        error,
        error_dot,
        error_int,
        pi_output,
    ]

    # 준비 컨텍스트 자리(프리차지 적용, J3, 준비 힘 평균, 준비 플래그)
    prep_context = [0.0, 0.0, 0.0, 0.0]

    state = np.array(base_state + prep_context, dtype=np.float32)

    if np.isnan(state).any() or np.isinf(state).any():
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    return state

# 주의: _estimate_state_from_previous_pid() 함수는 삭제되었습니다.
# 이 함수는 experiment 폴더의 레거시 코드에서만 사용되며,
# 현재 모듈화된 코드에서는 create_initial_state()를 사용합니다.
