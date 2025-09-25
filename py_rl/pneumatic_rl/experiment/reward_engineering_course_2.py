# reward_engineering_course_2.py
# 폴리싱 로봇 강화학습을 위한 개선된 보상 함수 커리큘럼
# Author: Claude Code Analysis + Advanced Theory Integration
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

개선 방향:
- 이론적 견고함과 실용성을 모두 확보
- 점진적 개선으로 안전한 발전
- 각 단계별 성능 검증 가능
"""


# =========================
# 개선된 보상 함수 커리큘럼
# =========================

class ImprovedRewardFunctionCourse:
    """폴리싱 로봇 강화학습을 위한 개선된 보상 함수 커리큘럼"""
    
    def __init__(self, config):
        self.cfg = config
        self.prev_residual = 0.0
        self.current_episode = 0
        
        # 성능 추적을 위한 변수들
        self.prev_episode_error = None
        self.reward_history = []
        self.reward_running_mean = 0.0
        self.reward_running_std = 1.0
        
        # PBRS를 위한 변수들
        self.prev_potential = None
        self.prev_error = None
        self.gamma = 0.99  # 할인 인자
        
        # 상태 탐험 추적
        self.state_visit_count = {}
        self.action_history = []
        
        # Multi-Objective를 위한 변수들
        self.objective_weights_history = []
        self.performance_history = {'precision': [], 'stability': [], 'safety': []}
        
    
    # ========================================
    # 1단계: 지수적 보상 + 기본 스케일링 개선
    # ========================================
    def calculate_reward_exponential_improved(self, state, action_residual, sander_active):
        """
        기존 지수적 보상 + 스케일 정규화 + 기본 안전성 강화
        
        개선사항:
        1. 스케일 정규화: 각 구성요소를 0~1 범위로 정규화
        2. 가중치 균형: 구성요소 간 균형 조정
        3. 부드러운 안전성: 선형 페널티로 개선
        """
        current_force, target_force = state[0], state[1]
        force_error, force_error_dot = state[2], state[3]
        
        force_err = abs(force_error)
        residual_change = abs(action_residual - self.prev_residual)
        
        # === 개선된 지수적 보상 구조 ===
        
        # 1) 추적 보상 (0~1 범위로 정규화)
        tracking_reward = np.exp(-force_err * 1.5)  # 0~1 범위
        
        # 2) 안정성 보상 (0~1 범위로 정규화)
        stability_reward = np.exp(-abs(force_error_dot) * 2.0)  # 0~1 범위
        
        # 3) 부드러움 보상 (0~1 범위로 정규화)
        smoothness_reward = np.exp(-residual_change * 50.0)  # 0~1 범위
        
        # 4) 근접 보너스 (기존 유지하되 스케일 조정)
        if force_err <= 0.5:
            proximity_bonus = 1.5 * (1 - force_err / 0.5)  # 0~1.5 범위
        else:
            proximity_bonus = 0.0
        
        # === 균형 잡힌 가중치 적용 ===
        base_reward = (3.0 * tracking_reward + 
                      1.5 * stability_reward + 
                      0.5 * smoothness_reward + 
                      proximity_bonus)
        
        # === 개선된 안전성 페널티 (부드러운 선형) ===
        if current_force > 80.0:
            safety_penalty = -2.0 * (current_force - 80.0) / 20.0  # 선형 페널티
        else:
            safety_penalty = 0.0
        
        # === 효율성 페널티 (감소) ===
        efficiency_penalty = -0.02 * abs(action_residual)
        
        total_reward = base_reward + safety_penalty + efficiency_penalty
        
        return total_reward
    
    
    # ========================================
    # 2단계: 다층 보상 + 적응적 가중치
    # ========================================
    def calculate_reward_multilevel_adaptive(self, state, action_residual, sander_active):
        """
        기존 다층 보상 + 상황별 적응적 가중치 + 기본 PBRS 개념
        
        개선사항:
        1. 적응적 가중치: 상황별 목적 우선순위 조정
        2. 기본 PBRS: 간단한 개선 보너스 추가
        3. 기존 구조 유지: 검증된 다층 구조 보존
        """
        current_force, target_force = state[0], state[1]
        force_error, force_error_dot = state[2], state[3]
        
        force_err = abs(force_error)
        residual_change = abs(action_residual - self.prev_residual)
        
        # === 기존 다층 구조 유지 ===
        reward = 0.0
        
        # 1) 정밀 추적 보상 (기존 구조)
        if force_err <= 0.5:
            tracking_reward = 2.0
        elif force_err <= 1.0:
            tracking_reward = 1.0
        elif force_err <= 2.0:
            tracking_reward = 0.5
        elif force_err <= 5.0:
            tracking_reward = 0.1
        else:
            tracking_reward = -(force_err - 5.0) * 0.1
        
        # 2) 안정성 보상 (기존 구조)
        error_rate = abs(force_error_dot)
        if error_rate <= 0.5:
            stability_reward = 0.5
        elif error_rate > 2.0:
            stability_reward = -0.5
        else:
            stability_reward = 0.0
        
        # 3) 부드러움 보상 (기존 구조)
        if residual_change <= 0.01:
            smoothness_reward = 0.2
        elif residual_change > 0.05:
            smoothness_reward = -0.3
        else:
            smoothness_reward = 0.0
        
        # === 개선사항 1: 상황별 적응적 가중치 ===
        if force_err > 5.0:  # 큰 오차 시 정밀도 우선
            weights = {'tracking': 0.7, 'stability': 0.2, 'smoothness': 0.1}
        elif force_err > 2.0:  # 중간 오차 시 균형
            weights = {'tracking': 0.5, 'stability': 0.3, 'smoothness': 0.2}
        else:  # 작은 오차 시 안정성 강조
            weights = {'tracking': 0.3, 'stability': 0.5, 'smoothness': 0.2}
        
        # 가중치 적용
        reward = (weights['tracking'] * tracking_reward + 
                 weights['stability'] * stability_reward + 
                 weights['smoothness'] * smoothness_reward)
        
        # === 개선사항 2: 기본 PBRS 개념 (간단한 형태) ===
        if self.prev_error is not None:
            error_improvement = self.prev_error - force_err
            if error_improvement > 0:
                reward += 0.5 * error_improvement  # 개선 보너스
        self.prev_error = force_err
        
        # === 안전성 페널티 (기존 유지) ===
        if current_force > 100.0:
            reward -= 10.0
        elif current_force > 90.0:
            reward -= 2.0
        elif current_force > 80.0:
            reward -= 0.5
        
        # === 효율성 페널티 ===
        reward -= 0.05 * abs(action_residual)
        
        return reward
    
    
    # ========================================
    # 3단계: 하이브리드 보상 + 고급 이론 통합
    # ========================================
    def calculate_reward_hybrid_advanced(self, state, action_residual, sander_active):
        """
        기존 하이브리드 + PBRS + Multi-Objective + 안전성 제약 강화
        
        개선사항:
        1. PBRS 적용: 정책 불변성 보장
        2. Multi-Objective: 전략적 목적 결합
        3. 안전성 강화: Hard/Soft constraint 분리
        """
        current_force, target_force = state[0], state[1]
        force_error, force_error_dot = state[2], state[3]
        force_error_int = state[4] if len(state) > 4 else 0.0
        
        force_err = abs(force_error)
        residual_change = abs(action_residual - self.prev_residual)
        
        # === 기존 하이브리드 구조 유지 ===
        
        # 1) 지수적 추적 보상 (핵심)
        tracking_reward = 3.0 * np.exp(-force_err * 1.5)
        
        # 2) PID 기반 안정성 보상
        p_reward = -0.1 * force_err
        i_penalty = -0.05 * abs(force_error_int) if abs(force_error_int) > 5.0 else 0.0
        d_reward = 0.5 * np.exp(-abs(force_error_dot) * 2.0)
        pid_reward = p_reward + i_penalty + d_reward
        
        # 3) 부드러운 제어
        smoothness_reward = 0.3 * np.exp(-residual_change * 100.0)
        
        # === 개선사항 1: PBRS 적용 ===
        current_potential = (np.exp(-force_err * 1.5) + 
                           np.exp(-abs(force_error_dot) * 2.0) + 
                           np.exp(-max(0, current_force - 80.0) * 0.1))
        
        if self.prev_potential is not None:
            pbrs_reward = self.gamma * current_potential - self.prev_potential
        else:
            pbrs_reward = 0.0
        self.prev_potential = current_potential
        
        # === 개선사항 2: Multi-Objective 접근법 ===
        objectives = {
            'precision': tracking_reward,
            'stability': pid_reward,
            'safety': self._calculate_safety_reward_advanced(current_force),
            'efficiency': -0.01 * abs(action_residual),
            'smoothness': smoothness_reward
        }
        
        # 상황별 적응적 가중치
        weights = self._get_adaptive_weights(force_err, current_force)
        
        # === 개선사항 3: 안전성 제약 강화 ===
        safety_penalty, should_terminate = self._check_safety_constraints(current_force)
        
        # === 최종 보상 계산 ===
        base_reward = sum(w * obj for w, obj in zip(weights.values(), objectives.values()))
        total_reward = base_reward + pbrs_reward + safety_penalty
        
        # 보상 클리핑 (학습 안정화)
        total_reward = np.clip(total_reward, -50.0, 20.0)
        
        return float(total_reward), should_terminate
    
    
    # ========================================
    # 보조 함수들 (Helper Functions)
    # ========================================
    
    def _get_adaptive_weights(self, force_err, current_force):
        """상황별 적응적 가중치 계산"""
        if force_err > 10.0:  # 큰 오차 시 정밀도 우선
            return {'precision': 0.6, 'stability': 0.2, 'safety': 0.1, 
                   'efficiency': 0.05, 'smoothness': 0.05}
        elif force_err > 5.0:  # 중간 오차 시 균형
            return {'precision': 0.4, 'stability': 0.3, 'safety': 0.15, 
                   'efficiency': 0.075, 'smoothness': 0.075}
        elif current_force > 85.0:  # 안전 위험 시 안전성 우선
            return {'precision': 0.2, 'stability': 0.2, 'safety': 0.5, 
                   'efficiency': 0.05, 'smoothness': 0.05}
        else:  # 작은 오차 시 안정성과 효율성 강조
            return {'precision': 0.3, 'stability': 0.35, 'safety': 0.1, 
                   'efficiency': 0.125, 'smoothness': 0.125}
    
    def _calculate_safety_reward_advanced(self, current_force):
        """고급 안전성 보상 계산"""
        if current_force <= 80.0:
            return 0.0  # 안전 범위
        elif current_force <= 90.0:
            return -0.5 * (current_force - 80.0) / 10.0  # 경고 (-0.5~0)
        elif current_force <= 100.0:
            return -2.0 * (current_force - 90.0) / 10.0 - 0.5  # 위험 (-2.5~-0.5)
        else:
            return -10.0  # 매우 위험
    
    def _check_safety_constraints(self, current_force):
        """안전성 제약 검사 (Hard/Soft constraint 분리)"""
        # Hard constraint: 절대 위반하면 안 되는 한계
        if current_force > 100.0:
            return -1000.0, True  # Episode termination
        
        # Soft constraint with exponential penalty
        if current_force > 80.0:
            safety_violation = current_force - 80.0
            penalty = -10.0 * np.exp(safety_violation * 0.1)
            return penalty, False
        
        return 0.0, False
    
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
# 성능 비교 및 평가 도구
# =========================

class ImprovedRewardComparator:
    """개선된 보상 함수들의 성능을 비교하는 도구"""
    
    def __init__(self):
        self.results = {}
        self.test_cases = []
    
    def generate_test_cases(self, num_cases=100):
        """다양한 테스트 케이스 생성"""
        test_cases = []
        
        for _ in range(num_cases):
            # 다양한 상황 시뮬레이션
            current_force = np.random.uniform(20.0, 100.0)
            target_force = np.random.uniform(25.0, 35.0)
            force_error = current_force - target_force
            force_error_dot = np.random.uniform(-2.0, 2.0)
            force_error_int = np.random.uniform(-10.0, 10.0)
            pi_output = np.random.uniform(0.1, 0.2)
            
            state = [current_force, target_force, force_error, force_error_dot, force_error_int, pi_output]
            action_residual = np.random.uniform(-0.1, 0.1)
            sander_active = np.random.choice([True, False])
            
            test_cases.append((state, action_residual, sander_active))
        
        self.test_cases = test_cases
        return test_cases
    
    def evaluate_reward_function(self, reward_func, test_cases=None, name="default"):
        """보상 함수를 테스트 케이스로 평가"""
        if test_cases is None:
            test_cases = self.test_cases
        
        rewards = []
        termination_flags = []
        
        for state, action, sander_active in test_cases:
            result = reward_func(state, action, sander_active)
            
            # 3단계는 (reward, should_terminate) 튜플 반환
            if isinstance(result, tuple):
                reward, should_terminate = result
                termination_flags.append(should_terminate)
            else:
                reward = result
                termination_flags.append(False)
            
            rewards.append(reward)
        
        self.results[name] = {
            'rewards': rewards,
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'positive_ratio': sum(1 for r in rewards if r > 0) / len(rewards),
            'termination_rate': sum(termination_flags) / len(termination_flags) if termination_flags else 0.0,
            'reward_range': np.max(rewards) - np.min(rewards)
        }
    
    def print_comparison(self):
        """보상 함수들의 비교 결과 출력"""
        print("=" * 80)
        print("개선된 보상 함수 성능 비교")
        print("=" * 80)
        
        for name, stats in self.results.items():
            print(f"\n{name}:")
            print(f"  평균 보상: {stats['mean']:.3f}")
            print(f"  표준편차: {stats['std']:.3f}")
            print(f"  범위: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  양수 보상 비율: {stats['positive_ratio']:.2%}")
            print(f"  보상 범위: {stats['reward_range']:.3f}")
            if 'termination_rate' in stats:
                print(f"  종료율: {stats['termination_rate']:.2%}")
    
    def plot_comparison(self):
        """보상 분포 시각화 (matplotlib 사용)"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('보상 함수 성능 비교', fontsize=16)
            
            # 1) 보상 분포 히스토그램
            ax1 = axes[0, 0]
            for name, stats in self.results.items():
                ax1.hist(stats['rewards'], alpha=0.6, label=name, bins=20)
            ax1.set_title('보상 분포')
            ax1.set_xlabel('보상 값')
            ax1.set_ylabel('빈도')
            ax1.legend()
            
            # 2) 평균 보상 비교
            ax2 = axes[0, 1]
            names = list(self.results.keys())
            means = [self.results[name]['mean'] for name in names]
            ax2.bar(names, means)
            ax2.set_title('평균 보상 비교')
            ax2.set_ylabel('평균 보상')
            
            # 3) 양수 보상 비율
            ax3 = axes[1, 0]
            positive_ratios = [self.results[name]['positive_ratio'] for name in names]
            ax3.bar(names, positive_ratios)
            ax3.set_title('양수 보상 비율')
            ax3.set_ylabel('비율')
            
            # 4) 보상 범위
            ax4 = axes[1, 1]
            ranges = [self.results[name]['reward_range'] for name in names]
            ax4.bar(names, ranges)
            ax4.set_title('보상 범위')
            ax4.set_ylabel('범위')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib가 설치되지 않아 시각화를 건너뜁니다.")


# =========================
# 실전 구현 가이드
# =========================

"""
== 개선된 추천 구현 순서 ==

1단계: 지수적 보상 개선 (calculate_reward_exponential_improved)
- 기존 지수적 보상 + 스케일 정규화 + 기본 안전성 강화
- 기대효과: 연속적이고 균형 잡힌 보상 신호

2단계: 다층 보상 적응형 (calculate_reward_multilevel_adaptive)  
- 기존 다층 보상 + 적응적 가중치 + 기본 PBRS 개념
- 기대효과: 상황별 최적화된 행동 가이드라인

3단계: 하이브리드 고급 (calculate_reward_hybrid_advanced)
- 기존 하이브리드 + PBRS + Multi-Objective + 안전성 제약 강화
- 기대효과: 이론적으로 견고하고 실용적인 최종 보상 함수

== 성능 검증 전략 ==

각 단계에서 다음 지표들을 모니터링:
1. 학습 안정성: 보상 분산, 수렴 속도
2. 최종 성능: 평균 오차, 성공률
3. 안전성: 안전 위반 빈도, 종료율
4. 효율성: 학습 시간, 에피소드 수

== A/B 테스트 예시 ==

# 보상 함수별 성능 비교
reward_course = ImprovedRewardFunctionCourse(CONFIG)
comparator = ImprovedRewardComparator()

# 테스트 케이스 생성
test_cases = comparator.generate_test_cases(1000)

# 각 보상 함수 평가
comparator.evaluate_reward_function(
    reward_course.calculate_reward_exponential_improved, test_cases, "1단계_지수적_개선"
)
comparator.evaluate_reward_function(
    reward_course.calculate_reward_multilevel_adaptive, test_cases, "2단계_다층_적응형"
)
comparator.evaluate_reward_function(
    reward_course.calculate_reward_hybrid_advanced, test_cases, "3단계_하이브리드_고급"
)

# 결과 비교
comparator.print_comparison()
comparator.plot_comparison()
"""


# =========================
# 사용 예시
# =========================

if __name__ == "__main__":
    # CONFIG 설정
    CONFIG = {
        "MAX_FORCE_ERR": 15.0,
        "MAX_PRESS_DELTA": 0.05,
        "R_MAX": 0.2,
    }
    
    # 개선된 보상 함수 커스 초기화
    reward_course = ImprovedRewardFunctionCourse(CONFIG)
    
    # 예시 상태
    state = np.array([25.0, 30.0, -5.0, -0.5, -10.0, 0.15])
    action_residual = 0.05
    sander_active = True
    
    # 각 개선된 보상 함수 테스트
    print("개선된 보상 함수 비교 테스트")
    print("-" * 50)
    
    # 1단계: 지수적 보상 개선
    reward1 = reward_course.calculate_reward_exponential_improved(state, action_residual, sander_active)
    print(f"1단계 - 지수적 보상 개선: {reward1:.3f}")
    
    # 2단계: 다층 보상 적응형
    reward2 = reward_course.calculate_reward_multilevel_adaptive(state, action_residual, sander_active)
    print(f"2단계 - 다층 보상 적응형: {reward2:.3f}")
    
    # 3단계: 하이브리드 고급
    reward3, should_terminate = reward_course.calculate_reward_hybrid_advanced(state, action_residual, sander_active)
    print(f"3단계 - 하이브리드 고급: {reward3:.3f} (종료: {should_terminate})")
    
    # 성능 비교 테스트
    print("\n성능 비교 테스트")
    print("-" * 50)
    
    comparator = ImprovedRewardComparator()
    test_cases = comparator.generate_test_cases(100)
    
    comparator.evaluate_reward_function(
        reward_course.calculate_reward_exponential_improved, test_cases, "1단계_지수적_개선"
    )
    comparator.evaluate_reward_function(
        reward_course.calculate_reward_multilevel_adaptive, test_cases, "2단계_다층_적응형"
    )
    comparator.evaluate_reward_function(
        reward_course.calculate_reward_hybrid_advanced, test_cases, "3단계_하이브리드_고급"
    )
    
    comparator.print_comparison()