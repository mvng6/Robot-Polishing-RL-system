# reward_engineering_course.py
# 폴리싱 로봇 강화학습을 위한 보상 함수 설계 커리큘럼
# Author: Claude Code Analysis
# Date: 2025-08-26

import numpy as np
import torch


# =========================
# 현재 보상 함수 분석
# =========================
"""
현재 구현된 보상 함수의 문제점:
1. 희소성 (Sparse Reward): 대부분 음수 보상
2. 스케일 불균형: 안전 페널티(-5.0)가 너무 강함
3. 이진적 보상: 1N 임계값이 너무 급작스러움
4. 탐험 부족: 긍정적 신호가 부족하여 탐험 저해

현재 보상 구조:
def calculate_reward(self, state, action_residual, sander_active):
    current_force, target_force = state[0], state[1]
    force_err = abs(current_force - target_force)
    residual_change = abs(action_residual - self.prev_residual)
    
    # 1) tracking: -(0~1) 정규화된 오차
    reward = -(force_err / self.cfg["MAX_FORCE_ERR"])
    if force_err < 1.0: reward += 0.5  # 1N 이내 보너스
    
    # 2) smoothness: 급격한 변화 페널티
    smooth_w = 0.2 if sander_active else 0.3
    reward += -smooth_w * (residual_change / self.cfg["MAX_PRESS_DELTA"])
    
    # 3) safety: 과도한 힘 페널티
    if current_force > 80.0: reward += -5.0
    
    # 4) efficiency: residual 크기 페널티
    reward += -0.1 * abs(action_residual)
"""


# =========================
# 보상 함수 개선 방안들
# =========================

class RewardFunctionCourse:
    """폴리싱 로봇 강화학습을 위한 보상 함수 커리큘럼"""
    
    def __init__(self, config):
        self.cfg = config
        self.prev_residual = 0.0
        self.current_episode = 0
        
        # 성능 추적을 위한 변수들
        self.prev_episode_error = None
        self.reward_history = []
        self.reward_running_mean = 0.0
        self.reward_running_std = 1.0
        
        # 상태 탐험 추적
        self.state_visit_count = {}
        self.action_history = []
        
    
    # ========================================
    # 방안 1: 다층 보상 구조 (Multi-Level Reward)
    # ========================================
    def calculate_reward_multilevel(self, state, action_residual, sander_active):
        """
        장점: 명확한 행동 가이드라인, 디버깅 용이
        단점: 임계값 설정 민감, 하드코딩된 구간
        적용: 초기 프로토타입, 도메인 지식 반영 필요시
        """
        current_force, target_force = state[0], state[1]
        force_error, force_error_dot = state[2], state[3]
        
        # === 기본 보상 구조 ===
        reward = 0.0
        
        # 1) 정밀 추적 보상 (Progressive Tracking)
        force_err = abs(force_error)
        if force_err <= 0.5:      # 매우 정확
            reward += 2.0
        elif force_err <= 1.0:    # 정확
            reward += 1.0 
        elif force_err <= 2.0:    # 양호
            reward += 0.5
        elif force_err <= 5.0:    # 허용
            reward += 0.1
        else:                     # 부정확
            reward -= (force_err - 5.0) * 0.1
        
        # 2) 안정성 보상 (Stability)
        error_rate = abs(force_error_dot)
        if error_rate <= 0.5:
            reward += 0.5  # 안정적
        elif error_rate > 2.0:
            reward -= 0.5  # 진동
        
        # 3) 효율성 보상 (Efficiency) - 더 부드럽게
        residual_penalty = -0.05 * abs(action_residual)  # 기존 -0.1 → -0.05
        reward += residual_penalty
        
        # 4) 부드러움 보상 (Smoothness) - 개선
        residual_change = abs(action_residual - self.prev_residual)
        if residual_change <= 0.01:
            reward += 0.2  # 부드러운 제어 보너스
        elif residual_change > 0.05:
            reward -= 0.3  # 급격한 변화 페널티
        
        # 5) 안전성 (Safety) - 단계적 페널티
        if current_force > 100.0:
            reward -= 10.0  # 위험
        elif current_force > 90.0:
            reward -= 2.0   # 경고
        elif current_force > 80.0:
            reward -= 0.5   # 주의
        
        return reward
    
    
    # ========================================
    # 방안 2: 지수적 보상 (Exponential Reward) - 추천!
    # ========================================
    def calculate_reward_exponential(self, state, action_residual, sander_active):
        """
        장점: 연속적, 부드러운 보상 신호, 미세 조정 가능
        단점: 지수 함수 파라미터 튜닝 필요
        적용: 첫 번째 시도 추천, 안정적 학습
        """
        current_force, target_force = state[0], state[1]
        force_error, force_error_dot = state[2], state[3]
        
        # === 지수적 근접 보상 ===
        force_err = abs(force_error)
        
        # 1) 지수적 추적 보상 (0~2.0 범위)
        tracking_reward = 2.0 * np.exp(-force_err * 2.0)  # 오차가 작을수록 지수적 증가
        
        # 2) 안정성 보상 (미분항 활용)
        error_velocity = abs(force_error_dot)
        stability_reward = 1.0 * np.exp(-error_velocity * 3.0)
        
        # 3) 목표 도달 보너스 (기존보다 부드럽게)
        if force_err <= 0.5:
            proximity_bonus = 1.5 * (1 - force_err / 0.5)  # 0.5N 이내에서 선형 보너스
        else:
            proximity_bonus = 0.0
        
        # 4) 부드러운 제어 보상
        residual_change = abs(action_residual - self.prev_residual)
        smoothness_reward = 0.5 * np.exp(-residual_change * 50.0)
        
        # 5) 기본 보상 합성
        reward = tracking_reward + 0.5 * stability_reward + proximity_bonus + smoothness_reward
        
        # 6) 페널티 (기존 유지하되 강도 조절)
        if current_force > 80.0:
            safety_penalty = -2.0 * (current_force - 80.0) / 20.0  # 선형 페널티
            reward += safety_penalty
        
        reward -= 0.02 * abs(action_residual)  # 효율성 페널티 감소
        
        return reward
    
    
    # ========================================
    # 방안 3: 커리큘럼 기반 적응형 보상
    # ========================================
    def calculate_reward_curriculum(self, state, action_residual, sander_active):
        """
        장점: 단계적 학습, 초기 탐험 → 정밀 제어 자연스러운 전환
        단점: 에피소드 추적 필요, 복잡성 증가
        적용: 고급 기법, 장기 학습 프로젝트
        """
        current_force, target_force = state[0], state[1]
        force_error = state[2]
        
        # === 학습 단계에 따른 적응형 보상 ===
        episode = self.current_episode
        
        # 1) 초기 단계 (1-3 에피소드): 관대한 보상
        if episode <= 3:
            tolerance = 3.0  # 3N 허용
            bonus_scale = 2.0
        # 2) 중간 단계 (4-7 에피소드): 점진적 엄격
        elif episode <= 7:
            tolerance = 1.5  # 1.5N 허용
            bonus_scale = 1.5
        # 3) 고급 단계 (8+ 에피소드): 엄격한 기준
        else:
            tolerance = 0.5  # 0.5N 허용
            bonus_scale = 1.0
        
        force_err = abs(force_error)
        
        # 적응형 보상 계산
        if force_err <= tolerance:
            reward = bonus_scale * (1.0 - force_err / tolerance)
        else:
            reward = -0.5 * (force_err - tolerance)
        
        # 진보 보너스 (시간에 따른 향상)
        if self.prev_episode_error is not None:
            improvement = self.prev_episode_error - force_err
            if improvement > 0:
                reward += 0.5 * improvement  # 개선 보너스
        
        # 일관성 보상 (에러율 안정성)
        error_velocity = abs(state[3])  # force_error_dot
        if error_velocity <= 1.0:
            reward += 0.3  # 안정적 제어
        
        return reward
    
    
    # ========================================
    # 방안 4: 내적 동기 기반 보상 (Intrinsic Motivation)
    # ========================================
    def calculate_reward_intrinsic(self, state, action_residual, sander_active):
        """
        장점: 자발적 탐험, 호기심 기반 학습
        단점: 복잡한 구현, 계산 비용 높음
        적용: 연구용, 고급 탐험 필요시
        """
        current_force, target_force = state[0], state[1]
        force_error = state[2]
        
        # === 기본 외적 보상 ===
        force_err = abs(force_error)
        extrinsic_reward = 1.0 * np.exp(-force_err)
        
        # === 내적 동기 보상 (호기심 기반) ===
        # 1) 새로운 상태 탐험 보상
        state_novelty = self._calculate_state_novelty(state)
        curiosity_reward = 0.3 * state_novelty
        
        # 2) 기술 향상 보상 (이전 성능 대비)
        skill_reward = self._calculate_skill_progress(force_err)
        
        # 3) 다양성 보상 (액션 다양성)
        action_diversity = self._calculate_action_diversity(action_residual)
        diversity_reward = 0.2 * action_diversity
        
        # === 종합 보상 ===
        total_reward = extrinsic_reward + curiosity_reward + skill_reward + diversity_reward
        
        # === 자동 스케일링 ===
        total_reward = self._normalize_reward(total_reward)
        
        return total_reward
    
    
    # ========================================
    # 방안 5: 하이브리드 보상 (권장 최종 버전)
    # ========================================
    def calculate_reward_hybrid(self, state, action_residual, sander_active):
        """
        지수적 보상 + 커리큘럼 + 안전성 강화
        실제 로봇 시스템에 최적화된 종합 보상 함수
        """
        current_force, target_force = state[0], state[1]
        force_error, force_error_dot = state[2], state[3]
        force_error_int = state[4]
        
        # === 주요 보상 구성 요소 ===
        force_err = abs(force_error)
        
        # 1) 지수적 추적 보상 (핵심)
        tracking_reward = 3.0 * np.exp(-force_err * 1.5)
        
        # 2) PID 기반 안정성 보상
        # P: 비례 오차
        p_reward = -0.1 * force_err
        # I: 적분 오차 (누적 바이어스 방지)
        i_penalty = -0.05 * abs(force_error_int) if abs(force_error_int) > 5.0 else 0.0
        # D: 미분 오차 (진동 억제)
        d_reward = 0.5 * np.exp(-abs(force_error_dot) * 2.0)
        
        pid_reward = p_reward + i_penalty + d_reward
        
        # 3) 커리큘럼 기반 적응
        curriculum_bonus = self._get_curriculum_bonus(force_err)
        
        # 4) 부드러운 제어
        residual_change = abs(action_residual - self.prev_residual)
        smoothness_reward = 0.3 * np.exp(-residual_change * 100.0)
        
        # 5) 안전성 (단계적)
        safety_reward = self._calculate_safety_reward(current_force)
        
        # === 최종 보상 합성 ===
        base_reward = tracking_reward + pid_reward + smoothness_reward
        bonus_reward = curriculum_bonus
        penalty_reward = safety_reward
        
        total_reward = base_reward + bonus_reward + penalty_reward
        
        # 효율성 페널티 (최소한)
        total_reward -= 0.01 * abs(action_residual)
        
        return total_reward
    
    
    # ========================================
    # 보조 함수들 (Helper Functions)
    # ========================================
    
    def _calculate_state_novelty(self, state):
        """상태 공간 탐험 정도 측정"""
        # 상태를 이산화하여 방문 빈도 추적
        state_key = tuple(np.round(state, 1))  # 소수점 1자리로 양자화
        
        if state_key not in self.state_visit_count:
            self.state_visit_count[state_key] = 0
        
        self.state_visit_count[state_key] += 1
        
        # 방문 빈도가 낮을수록 높은 새로움 점수
        visit_count = self.state_visit_count[state_key]
        novelty = 1.0 / (1.0 + visit_count * 0.1)
        
        return novelty
    
    def _calculate_skill_progress(self, current_error):
        """기술 향상 보상 계산"""
        if len(self.reward_history) < 10:
            return 0.0
        
        # 최근 10스텝 평균 오차와 비교
        recent_avg_error = np.mean([abs(h) for h in self.reward_history[-10:]])
        
        if current_error < recent_avg_error:
            progress = (recent_avg_error - current_error) / recent_avg_error
            return 0.5 * progress
        
        return 0.0
    
    def _calculate_action_diversity(self, action_residual):
        """액션 다양성 측정"""
        self.action_history.append(action_residual)
        
        if len(self.action_history) < 5:
            return 0.0
        
        # 최근 5개 액션의 표준편차를 다양성 지표로 사용
        recent_actions = self.action_history[-5:]
        diversity = np.std(recent_actions)
        
        # 적절한 다양성 범위로 정규화
        normalized_diversity = min(diversity / 0.1, 1.0)
        
        return normalized_diversity
    
    def _normalize_reward(self, reward):
        """보상 정규화 (학습 안정화)"""
        # 지수 이동 평균으로 실행 평균/표준편차 업데이트
        alpha = 0.01
        self.reward_running_mean = (1 - alpha) * self.reward_running_mean + alpha * reward
        
        squared_diff = (reward - self.reward_running_mean) ** 2
        self.reward_running_std = np.sqrt((1 - alpha) * self.reward_running_std ** 2 + alpha * squared_diff)
        
        # 정규화된 보상 반환
        normalized_reward = (reward - self.reward_running_mean) / (self.reward_running_std + 1e-8)
        
        return normalized_reward
    
    def _get_curriculum_bonus(self, force_err):
        """커리큘럼 기반 보너스 계산"""
        episode = self.current_episode
        
        if episode <= 3:
            # 초기: 3N 이내면 보너스
            return 1.0 if force_err <= 3.0 else 0.0
        elif episode <= 7:
            # 중기: 1.5N 이내면 보너스
            return 1.0 if force_err <= 1.5 else 0.0
        else:
            # 후기: 0.5N 이내면 보너스
            return 2.0 if force_err <= 0.5 else 0.0
    
    def _calculate_safety_reward(self, current_force):
        """안전성 보상 계산"""
        if current_force <= 80.0:
            return 0.0  # 안전 범위
        elif current_force <= 90.0:
            return -0.5 * (current_force - 80.0) / 10.0  # 경고 (-0.5~0)
        elif current_force <= 100.0:
            return -2.0 * (current_force - 90.0) / 10.0 - 0.5  # 위험 (-2.5~-0.5)
        else:
            return -10.0  # 매우 위험
    
    def update_episode(self, episode):
        """에피소드 업데이트"""
        self.current_episode = episode
    
    def update_prev_residual(self, residual):
        """이전 residual 업데이트"""
        self.prev_residual = residual
    
    def add_to_history(self, reward):
        """보상 히스토리 추가"""
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:  # 메모리 관리
            self.reward_history = self.reward_history[-500:]


# =========================
# 보상 함수 비교 및 평가 도구
# =========================

class RewardComparator:
    """여러 보상 함수의 성능을 비교하는 도구"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_reward_function(self, reward_func, test_cases, name="default"):
        """보상 함수를 테스트 케이스로 평가"""
        rewards = []
        for state, action, sander_active in test_cases:
            reward = reward_func(state, action, sander_active)
            rewards.append(reward)
        
        self.results[name] = {
            'rewards': rewards,
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'positive_ratio': sum(1 for r in rewards if r > 0) / len(rewards)
        }
    
    def print_comparison(self):
        """보상 함수들의 비교 결과 출력"""
        print("=" * 60)
        print("보상 함수 성능 비교")
        print("=" * 60)
        
        for name, stats in self.results.items():
            print(f"\n{name}:")
            print(f"  평균: {stats['mean']:.3f}")
            print(f"  표준편차: {stats['std']:.3f}")
            print(f"  범위: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  양수 보상 비율: {stats['positive_ratio']:.2%}")


# =========================
# 실전 구현 가이드
# =========================

"""
== 추천 구현 순서 ==

1단계: 지수적 보상 (calculate_reward_exponential) 먼저 시도
- 이유: 구현 간단하면서도 효과적
- 기대효과: 연속적이고 부드러운 보상 신호

2단계: 다층 보상 (calculate_reward_multilevel) 추가 실험  
- 이유: 명확한 행동 가이드라인, 디버깅 용이
- 기대효과: 도메인 지식 반영 가능

3단계: 하이브리드 보상 (calculate_reward_hybrid) 최종 적용
- 이유: 실전 최적화된 종합 보상
- 기대효과: 안정적이고 효율적인 학습

== 하이퍼파라미터 권장값 ==

CONFIG.update({
    "REWARD_SCALE": 1.0,          # 전체 보상 스케일
    "TRACKING_WEIGHT": 2.0,       # 추적 정확도 가중치  
    "STABILITY_WEIGHT": 1.0,      # 안정성 가중치
    "EFFICIENCY_WEIGHT": 0.05,    # 효율성 가중치
    "SAFETY_THRESHOLD": 80.0,     # 안전 임계값 (N)
    "CURRICULUM_EPISODES": [3, 7] # 커리큘럼 전환점
})

== A/B 테스트 예시 ==

# 보상 함수별 성능 비교
reward_course = RewardFunctionCourse(CONFIG)
comparator = RewardComparator()

# 테스트 케이스 생성 
test_cases = [
    # (state, action_residual, sander_active)
    ([current_force, target_force, error, error_dot, error_int, pi], residual, True)
    for current_force, target_force, error, error_dot, error_int, pi, residual
    in test_scenarios
]

# 각 보상 함수 평가
comparator.evaluate_reward_function(
    reward_course.calculate_reward_exponential, test_cases, "exponential"
)
comparator.evaluate_reward_function(
    reward_course.calculate_reward_multilevel, test_cases, "multilevel"  
)

# 결과 비교
comparator.print_comparison()
"""

# =========================
# 사용 예시
# =========================

if __name__ == "__main__":
    # CONFIG 설정 (실제 CONFIG 딕셔너리 사용)
    CONFIG = {
        "MAX_FORCE_ERR": 15.0,
        "MAX_PRESS_DELTA": 0.05,
        "R_MAX": 0.2,
    }
    
    # 보상 함수 커스 초기화
    reward_course = RewardFunctionCourse(CONFIG)
    
    # 예시 상태
    state = np.array([25.0, 30.0, -5.0, -0.5, -10.0, 0.15])  # [current_force, target_force, error, error_dot, error_int, pi_output]
    action_residual = 0.05  # MPa
    sander_active = True
    
    # 각 보상 함수 테스트
    print("보상 함수 비교 테스트")
    print("-" * 40)
    
    reward1 = reward_course.calculate_reward_multilevel(state, action_residual, sander_active)
    print(f"다층 보상: {reward1:.3f}")
    
    reward2 = reward_course.calculate_reward_exponential(state, action_residual, sander_active) 
    print(f"지수 보상: {reward2:.3f}")
    
    reward3 = reward_course.calculate_reward_curriculum(state, action_residual, sander_active)
    print(f"커리큘럼 보상: {reward3:.3f}")
    
    reward4 = reward_course.calculate_reward_hybrid(state, action_residual, sander_active)
    print(f"하이브리드 보상: {reward4:.3f}")