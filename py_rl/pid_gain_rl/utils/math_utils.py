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
    target_force=-40.0,
    previous_pid_gains=None,
    episode_history=None,
    dt_sec=0.001,
):
    """초기 상태 생성 (12차원)"""
    if not force_data:
        if previous_pid_gains is not None:
            return _estimate_state_from_previous_pid(
                previous_pid_gains,
                target_force,
                episode_history,
            )
        else:
            print("🆕 [첫 에피소드] 기본 상태로 시작 (데이터 없음)")
            return np.array(
                [
                    target_force,  # current_force
                    target_force,  # target_force
                    0.0,           # error
                    0.0,           # error_dot
                    0.0,           # error_int
                    0.0,           # pid_norm_summary
                    0.0,           # pid_step_norm
                    0.0,           # recent_error_avg
                    0.0,           # recent_error_std
                    0.0,           # performance_trend
                    0.0,           # avg_recent_performance
                    0.0,           # episode_count_norm
                ],
                dtype=np.float32,
            )

    force_arr = np.asarray(force_data, dtype=np.float32)
    current_force = force_arr[-1]
    error = current_force - target_force
    all_errors = force_arr - target_force

    if all_errors.size > 0:
        recent_error_avg = float(np.mean(np.abs(all_errors)))
        recent_error_std = (
            float(np.std(all_errors)) if all_errors.size > 1 else 0.0
        )
    else:
        recent_error_avg, recent_error_std = 0.0, 0.0

    if force_arr.size >= 2:
        error_dot = float((force_arr[-1] - force_arr[-2]) / dt_sec)
    else:
        error_dot = 0.0

    error_int = float(np.sum(np.abs(all_errors)) * dt_sec)
    performance_trend = avg_recent_performance = 0.0

    # PID 정규화 및 메타 피처 계산 (압축)
    pid_norm_summary = 0.0
    pid_step_norm = 0.0
    episode_count_norm = 0.0
    
    if episode_history is not None and len(episode_history) > 0:
        recent_rewards = [ep["reward"] for ep in episode_history[-5:]]
        recent_pids = [ep["pid_gains"] for ep in episode_history[-5:]]
        
        if len(recent_rewards) >= 2:
            performance_trend = recent_rewards[-1] - recent_rewards[-2]
            avg_recent_performance = np.mean(recent_rewards)
        else:
            performance_trend = 0.0
            avg_recent_performance = recent_rewards[0] if recent_rewards else 0.0
            
        # PID 정규화된 값 계산 및 압축
        if len(recent_pids) >= 1:
            pid_range = Constants.DEFAULT_PID_RANGE
            last_pid = np.array(recent_pids[-1], dtype=np.float32)
            pid_norm = np.array([
                (last_pid[0] - pid_range["Kp"][0]) / (pid_range["Kp"][1] - pid_range["Kp"][0] + 1e-9),
                (last_pid[1] - pid_range["Ki"][0]) / (pid_range["Ki"][1] - pid_range["Ki"][0] + 1e-9),
                (last_pid[2] - pid_range["Kd"][0]) / (pid_range["Kd"][1] - pid_range["Kd"][0] + 1e-9),
            ], dtype=np.float32)
            pid_norm_summary = float(np.linalg.norm(pid_norm) / np.sqrt(3.0))
            
            # 직전 에피소드 대비 PID 스텝 크기
            if len(recent_pids) >= 2:
                prev_pid = np.array(recent_pids[-2], dtype=np.float32)
                prev_pid_norm = np.array([
                    (prev_pid[0] - pid_range["Kp"][0]) / (pid_range["Kp"][1] - pid_range["Kp"][0] + 1e-9),
                    (prev_pid[1] - pid_range["Ki"][0]) / (pid_range["Ki"][1] - pid_range["Ki"][0] + 1e-9),
                    (prev_pid[2] - pid_range["Kd"][0]) / (pid_range["Kd"][1] - pid_range["Kd"][0] + 1e-9),
                ], dtype=np.float32)
                pid_step_norm = float(np.linalg.norm(pid_norm - prev_pid_norm))
                
        episode_count_norm = min(1.0, len(episode_history) / float(Constants.DEFAULT_EPISODES))
    else:
        performance_trend = avg_recent_performance = 0.0

    # 12D 상태 벡터
    state = np.array(
        [
            current_force,
            target_force,
            error,
            error_dot,
            error_int,
            pid_norm_summary,
            pid_step_norm,
            recent_error_avg,
            recent_error_std,
            performance_trend,
            avg_recent_performance,
            episode_count_norm,
        ],
        dtype=np.float32,
    )
    
    # 검증
    if np.isnan(state).any() or np.isinf(state).any():
        print(f"❌ [오류] create_initial_state에서 NaN/Inf 발견!")
        print(f"   state: {state}")
        state = np.array([
            target_force, target_force, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        print(f"   → 기본값으로 대체: {state}")
    
    return state


def _estimate_state_from_previous_pid(
    previous_pid_gains,
    target_force,
    episode_history=None,
):
    """이전 PID를 기반으로 초기 상태 추정"""
    pid_norm_summary = 0.0
    pid_step_norm = 0.0
    episode_count_norm = 0.0
    
    if episode_history is not None and len(episode_history) > 0:
        recent_rewards = [ep["reward"] for ep in episode_history[-5:]]
        recent_pids = [ep["pid_gains"] for ep in episode_history[-5:]]
        
        if len(recent_rewards) >= 2:
            performance_trend = recent_rewards[-1] - recent_rewards[-2]
            avg_recent_performance = np.mean(recent_rewards)
        else:
            performance_trend = 0.0
            avg_recent_performance = recent_rewards[0] if recent_rewards else 0.0
            
        if len(recent_pids) >= 1:
            pid_range = Constants.DEFAULT_PID_RANGE
            last_pid = np.array(recent_pids[-1], dtype=np.float32)
            pid_norm = np.array([
                (last_pid[0] - pid_range["Kp"][0]) / (pid_range["Kp"][1] - pid_range["Kp"][0] + 1e-9),
                (last_pid[1] - pid_range["Ki"][0]) / (pid_range["Ki"][1] - pid_range["Ki"][0] + 1e-9),
                (last_pid[2] - pid_range["Kd"][0]) / (pid_range["Kd"][1] - pid_range["Kd"][0] + 1e-9),
            ], dtype=np.float32)
            pid_norm_summary = float(np.linalg.norm(pid_norm) / np.sqrt(3.0))
            
            if len(recent_pids) >= 2:
                prev_pid = np.array(recent_pids[-2], dtype=np.float32)
                prev_pid_norm = np.array([
                    (prev_pid[0] - pid_range["Kp"][0]) / (pid_range["Kp"][1] - pid_range["Kp"][0] + 1e-9),
                    (prev_pid[1] - pid_range["Ki"][0]) / (pid_range["Ki"][1] - pid_range["Ki"][0] + 1e-9),
                    (prev_pid[2] - pid_range["Kd"][0]) / (pid_range["Kd"][1] - pid_range["Kd"][0] + 1e-9),
                ], dtype=np.float32)
                pid_step_norm = float(np.linalg.norm(pid_norm - prev_pid_norm))
                
        episode_count_norm = min(1.0, len(episode_history) / float(Constants.DEFAULT_EPISODES))
        
        if avg_recent_performance > 0:
            avg_error, error_std = -target_force * 0.05, abs(target_force) * 0.02
        else:
            avg_error, error_std = -target_force * 0.15, abs(target_force) * 0.08
    else:
        avg_error, error_std = -target_force * 0.1, abs(target_force) * 0.05
        performance_trend = avg_recent_performance = 0.0

    estimated_force = target_force + avg_error
    state = np.array(
        [
            estimated_force,
            target_force,
            avg_error,
            0.0,
            0.0,
            pid_norm_summary,
            pid_step_norm,
            abs(avg_error),
            error_std,
            performance_trend,
            avg_recent_performance,
            episode_count_norm,
        ],
        dtype=np.float32,
    )
    
    # 검증
    if np.isnan(state).any() or np.isinf(state).any():
        print(f"❌ [오류] _estimate_state_from_previous_pid에서 NaN/Inf 발견!")
        state = np.array([
            target_force, target_force, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
    
    return state

