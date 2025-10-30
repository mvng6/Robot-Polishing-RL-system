# PID 게인 강화학습 시스템 발전 로드맵
## 멀티 타깃 접촉력 일반화 전략

### 🎯 최종 목표
**30-50N 범위의 랜덤한 목표 접촉력에 대해 적절한 PID 게인을 자동으로 출력하는 강화학습 에이전트**
- 한 에피소드에 목표가 여러 번 변경 (예: -35N → -46N → -40N)
- 각 목표에 대해 최적의 PID 게인 선택

### 📊 현재 상황
- **실행 중인 시스템**: 고정된 -40N 목표 접촉력
- **문제**: 아직 충분한 학습이 이루어지지 않음, 안정성 확보 필요
- **다음 단계**: -40N에서 안정적인 성능 확보 → 점진적 확장

### 📈 성공 기준
1. **고정 목표**(-40N): RMSE < 2N, Settling Time < 2초, Overshoot < 10%
2. **다중 목표**(30-50N): 모든 목표에서 위 기준 만족
3. **동적 변경**: 에피소드 내 목표 변경 시 빠른 적응

---

## 단계별 발전 전략

### Phase 0: 기본 안정화 (현재 단계)
**목표**: 고정된 -40N에서 안정적인 학습 및 성능 확보

#### 0.1 핵심 지표 정의
**성공 기준**:
- RMSE < 2N (목표 접촉력 대비)
- Settling Time < 2초 (목표 범위 ±5% 도달 시간)
- Overshoot < 10%
- Band Time > 70% (에피소드 시간 동안 목표 범위 유지)

**실패 기준**:
- RMSE > 5N 지속
- Overshoot > 25%
- 진동 발생 (오실레이션)
- 목표 도달 실패

#### 0.2 모니터링 체계
현재 코드에서 확인해야 할 부분:
```python
# 파일: JY_PID_Gain_SAC_MDP_monitor_3_reset.py
# 수집해야 할 데이터:
{
    "episode": ep,
    "target_force": -40.0,  # 고정
    "pid_gains": [Kp, Ki, Kd],
    "reward": float,
    "metrics": {
        "rmse": float,           # ← 이게 2N 이하인가?
        "settling_time": float,  # ← 이게 2초 이하인가?
        "overshoot": float,      # ← 이게 10% 이하인가?
        "band_time": float       # ← 이게 70% 이상인가?
    }
}
```

#### 0.3 초기 학습 전략
1. **하이퍼파라미터 튜닝**
   - Learning rate: 3e-4 (현재) → 테스트 필요
   - Batch size: 128 (현재) → 충분한 데이터 수집 후 증가 가능
   - Updates per episode: 8 (현재) → 과적합 방지에 적합

2. **보상 함수 점검**
   - RMSE 가중치 확인
   - Overshoot 패널티 확인
   - 각 지표가 균형잡혀 있는지 확인

3. **안정성 검증**
   - 최소 50 에피소드 동안 안정적 성능
   - 10 에피소드 rolling average로 추세 확인
   - 발산 없이 수렴하는지 확인

#### 0.4 로그 분석 방법
```bash
# 로그 파일 위치
py_rl/pid_gain_rl/experiment_logs/

# 확인할 파일들:
- episode_rewards.csv: 에피소드별 보상
- episode_metrics.csv: RMSE, settling time 등
- reward_breakdown.csv: 보상 구성 요소
```

**예상 소요 기간**: 2-3주 (안정적 성능 확보까지)

#### 0.5 전이학습 전략
**각 Phase 전환 시 모델 저장 및 로드**:
```python
# Phase 0 완료 시 최고 성능 모델 저장
best_model_path = f"{model_save_dir}/phase0_best_model.pth"
agent.save_model(best_model_path)

# Phase 1 시작 시 전이학습
agent.transfer_learning_setup(best_model_path, learning_rate_scale=0.1)
```

**전이학습 장점**:
- 이전 Phase의 학습된 지식을 활용
- 더 빠른 수렴 속도
- 안정적인 학습 진행

---

### Phase 1: 고정 목표에서 안정적인 성능 확보
**목표**: -40N 고정 목표에서 목표 성능 달성

#### 1.1 성능 기준 확립
현재 코드에서 다음 지표를 추적:
- `rmse`: 에피소드 전체의 RMSE
- `settling_time`: 목표 ±5% 범위 도달 시간
- `overshoot`: 최대 오버슈트 비율
- `band_time`: 목표 범위 내 체류 시간

#### 1.2 학습 진행 모니터링
**주요 확인 사항**:
1. 보상이 시간에 따라 증가하는가? (수렴성)
2. RMSE가 감소하는가? (정확도 개선)
3. Settling time이 단축되는가? (응답 속도)
4. Overshoot이 감소하는가? (안정성)

#### 1.3 문제 해결 체크리스트
만약 성능이 나아지지 않으면:
- [ ] Replay buffer 크기 확인 (충분한 데이터?)
- [ ] Network capacity 확인 (충분히 표현력 있는가?)
- [ ] Reward function 균형 확인
- [ ] Exploration vs Exploitation trade-off
- [ ] PID 범위 설정 확인 (너무 넓거나 좁은가?)

#### 1.4 성공 기준 달성 확인
```python
# 에피소드 50-100개를 실행한 후 검증
success_criteria = {
    "rmse_mean": np.mean(rmse_list) < 2.0,
    "settling_mean": np.mean(settling_time_list) < 2.0,
    "overshoot_max": np.max(overshoot_list) < 0.10,
    "band_time_min": np.min(band_time_list) > 0.70,
    "stability": np.std(rmse_list[-10:]) < 0.5  # 마지막 10개 에피소드 안정성
}
```

**예상 소요 기간**: 2-3주

---

### Phase 2: 단일 목표 변동 (30-50N 범위)
**목표**: 에피소드별 랜덤 목표 접촉력에 적응하는 에이전트

#### 2.1 목표 범위 확장
현재: 고정된 -40N
목표: 30-50N 범위의 랜덤 목표

#### 2.2 구현 방법
```python
# 에피소드 시작 시 랜덤 목표 생성
target_min, target_max = 30, 50
current_target = -random.uniform(target_min, target_max)

# 상태 벡터에 목표 힘 전달
state = create_initial_state(
    force_data=[],
    target_force=current_target,  # ← 변경!
    previous_pid_gains=prev_pid,
    episode_history=history
)
```

#### 2.3 점진적 확장 전략
**단계 A**: 좁은 범위부터
- 목표: -38N ~ -42N (±2N)
- 이유: 기존 학습된 -40N 근처
- 검증: 모든 목표에서 성능 유지

**단계 B**: 중간 범위
- 목표: -35N ~ -45N (±5N)
- 검증: 성능 유지 확인

**단계 C**: 최종 범위
- 목표: -30N ~ -50N (전체 범위)
- 검증: 최종 일반화 성능

#### 2.4 검증 지표
각 목표 구간별 성능 추적:
```python
performance_by_target = {
    "30-35N": {"rmse": [], "settling": []},
    "36-40N": {"rmse": [], "settling": []},
    "41-45N": {"rmse": [], "settling": []},
    "46-50N": {"rmse": [], "settling": []}
}
```

**예상 소요 기간**: 2-3주

---

### Phase 3: 에피소드 내 목표 변경 테스트
**목표**: 한 에피소드 내에서 목표가 변경될 때의 대응 능력 학습

#### 3.1 시나리오 설계
```python
# 에피소드 내 목표 변경 스케줄
episode_targets = [
    (-35, 3.0),  # 첫 3초: -35N
    (-46, 6.0),  # 다음 3초: -46N  
    (-40, 9.0),  # 마지막: -40N
]
```

#### 3.2 핵심 문제 분석
**현재 시스템 제약**:
- 에피소드 시작 시 **한 번만** PID 게인 선택
- 에피소드 동안 동일한 PID 게인 유지
- 목표가 변경되어도 PID 게인 변경 불가

**해결 방안**:
한 에피소드를 여러 스테이지로 분할하고, 각 스테이지마다 새로운 PID 게인 선택

#### 3.3 구현 전략
```python
# 스테이지 정의
episode_stages = [
    (target_force=-35.0, duration=3.0),  # 0-3초
    (target_force=-46.0, duration=6.0),  # 3-6초  
    (target_force=-40.0, duration=4.0),  # 6-10초
]

# 각 스테이지마다 실행
for stage_idx, (target, duration) in enumerate(episode_stages):
    # 1. PID 게인 선택
    state = create_initial_state([], target, prev_pid, history)
    pid_gains, _ = agent.select_action(state)
    
    # 2. 스테이지 실행
    stage_reward = run_stage(target, duration, pid_gains)
    
    # 3. 학습 (optional - 스테이지 종료 시)
    if stage_idx > 0:
        agent.store_transition(...)
        agent.update_parameters_one_step()
```

#### 3.4 점진적 확장
**단계 1**: 2단계 에피소드
- 스테이지 1: -40N (5초)
- 스테이지 2: -45N (5초)
- 검증: 각 스테이지에서 적절한 PID 게인 출력

**단계 2**: 3단계 에피소드
- 스테이지 1: -35N (3초)
- 스테이지 2: -46N (4초)
- 스테이지 3: -40N (3초)

**단계 3**: 랜덤 다단계
- 스테이지 개수, 목표, 길이 모두 랜덤

#### 3.5 성능 기준
각 스테이지별로 독립적으로 평가:
- 각 스테이지의 RMSE < 2N
- Settling time < 전체 스테이지 시간의 50%
- 스테이지 전환 시 오버슈트 최소화

**예상 소요 기간**: 3-4주

---

### Phase 4: 알고리즘 개선 및 최적화

**목표**: 더 빠른 적응과 일반화를 위한 알고리즘 개선

#### 4.1 상태 표현 개선
**현재 상태 벡터(12D)**:
```
[0] current_force
[1] target_force  ← 중요!
[2] error
[3] error_dot
[4] error_int
[5] pid_norm_summary
[6] pid_step_norm
[7] recent_error_avg
[8] recent_error_std
[9] performance_trend
[10] avg_recent_performance
[11] episode_count_norm
```

**개선 제안**:
- **목표 히스토리**: 최근 N개의 목표 히스토리 추가
- **상대적 오차**: error / target_force (정규화된 오차)
- **목표 변화율**: 목표 변경 속도 추적

#### 4.2 보상 함수 튜닝
현재 보상이 고정 목표에 최적화되어 있음:
```python
# 현재: 절대값 기반 보상
rmse = sqrt(mean((force - TARGET_FORCE)^2))

# 개선: 정규화된 상대 오차
rmse_normalized = sqrt(mean((force - target_force) / abs(target_force))^2)
```

#### 4.3 액션 정규화
여러 목표에서 작동하려면 액션을 정규화:
```python
# 현재: 절대 PID 값 출력
pid_gains = [Kp, Ki, Kd]

# 개선: 상대적 게인 (목표별 조정)
normalized_action = action * scaling_factor(target_force)
```

**예상 소요 기간**: 2-3주

---

### Phase 5: 일반화 성능 검증 및 최종 평가
**목표**: 다양한 시나리오에서 성능 검증

#### 5.1 테스트 시나리오
1. **동일 목표 범위 내 (30-50N)**
   - 35N, 40N, 45N 등 고정 목표 테스트
   
2. **에피소드 내 동적 변경**
   - 2단계: -35N → -45N
   - 3단계: -35N → -46N → -40N
   - 불규칙: 랜덤 순서
   
3. **어댑테이션 테스트**
   - 한 범위에서 학습 → 다른 범위에서 테스트
   - Zero-shot vs Few-shot 비교

#### 5.1 평가 지표
**Phase 0-1**: 
- ✅ `RMSE`: 목표별 RMSE 비교
- ✅ `Settling Time`: 목표 도달 시간
- ✅ `Overshoot`: 과도응답 비율

**Phase 2-3**:
- ✅ `Generalization Gap`: 학습 범위 vs 테스트 범위 성능 차이
- ✅ `Adaptation Speed`: 목표 변경 후 얼마나 빠르게 적응?
- ✅ `Transfer Efficiency`: 이전 목표의 경험 활용도

**Phase 4-5**:
- ✅ `Multi-Target Performance`: 여러 목표에서 동시 성공율
- ✅ `Robustness`: 노이즈나 변화에 대한 강건성

---

## 핵심 지표와 모니터링

### 1. 학습 진행 지표
```python
# 에피소드별 저장
{
    "episode": 1,
    "target_force": -42.3,
    "pid_gains": [65.0, 90.0, 0.0],
    "reward": 0.85,
    "rmse": 2.5,
    "settling_time": 1.2,
    "overshoot": 5.0
}
```

### 2. 일반화 성능 지표
```python
# 여러 목표에 대한 성능 매트릭스
generalization_matrix = {
    (30, 50): {
        "test_targets": [35, 40, 45],
        "avg_rmse": [...],
        "success_rate": 0.95
    }
}
```

### 3. 실시간 모니터링
```python
# 각 목표별 실시간 성능 추적
force_tracking = {
    "current_force": [...],
    "desired_force": [...],  # 목표 히스토리
    "error": [...],
    "pid_output": [...]
}
```

---

## 전이학습 전략 (Transfer Learning)

### 핵심 개념
각 Phase를 지날 때마다 이전 Phase의 학습된 신경망 가중치를 전이하여 더 빠르고 안정적인 학습을 수행합니다.

### Phase별 전이학습 전략

#### Phase 0 → Phase 1 전이
```python
# Phase 0 완료 시 최고 성능 모델 저장
phase0_model = "saved_agents/phase0_best_model.pth"
agent.save_model(phase0_model)

# Phase 1 시작 시 전이학습 (학습률 10%로 감소)
agent.transfer_learning_setup(phase0_model, learning_rate_scale=0.1)
```

#### Phase 1 → Phase 2 전이
```python
# Phase 1 완료 시 최고 성능 모델 저장
phase1_model = "saved_agents/phase1_best_model.pth"
agent.save_model(phase1_model)

# Phase 2 시작 시 전이학습 (학습률 5%로 감소)
agent.transfer_learning_setup(phase1_model, learning_rate_scale=0.05)
```

#### Phase 2 → Phase 3 전이
```python
# Phase 2 완료 시 최고 성능 모델 저장
phase2_model = "saved_agents/phase2_best_model.pth"
agent.save_model(phase2_model)

# Phase 3 시작 시 전이학습 (학습률 5%로 감소)
agent.transfer_learning_setup(phase2_model, learning_rate_scale=0.05)
```

#### Phase 3 → Phase 4 전이
```python
# Phase 3 완료 시 최고 성능 모델 저장
phase3_model = "saved_agents/phase3_best_model.pth"
agent.save_model(phase3_model)

# Phase 4 시작 시 전이학습 (학습률 1%로 감소)
agent.transfer_learning_setup(phase3_model, learning_rate_scale=0.01)
```

### 전이학습 장점

1. **빠른 수렴**: 이전 Phase의 지식을 활용하여 더 빠른 학습
2. **안정성**: 이미 학습된 가중치로 시작하여 발산 위험 감소
3. **효율성**: 처음부터 학습하는 것보다 훨씬 적은 에피소드로 목표 달성
4. **일반화**: 점진적 확장으로 더 강건한 모델 생성

### 학습률 스케일링 전략

| Phase 전환 | 학습률 스케일 | 이유 |
|-----------|-------------|------|
| Phase 0 → 1 | 0.1 (10%) | 고정 목표에서 다중 목표로 확장 |
| Phase 1 → 2 | 0.05 (5%) | 단일 목표에서 동적 변경으로 확장 |
| Phase 2 → 3 | 0.05 (5%) | 알고리즘 개선으로 미세 조정 |
| Phase 3 → 4 | 0.01 (1%) | 최종 검증으로 매우 미세한 조정 |

### 모델 저장 전략

```python
# 각 Phase 완료 시 저장할 정보
checkpoint = {
    "actor": actor.state_dict(),
    "critic": critic.state_dict(), 
    "critic_target": critic_target.state_dict(),
    "actor_opt": actor_opt.state_dict(),
    "critic_opt": critic_opt.state_dict(),
    "total_steps": total_steps,
    "episode_rewards": episode_rewards,
    "cfg": cfg,  # 설정 정보
    "phase": current_phase,  # Phase 정보
    "performance": {  # 성능 정보
        "best_reward": best_reward,
        "best_rmse": best_rmse,
        "best_settling_time": best_settling_time
    }
}
```

### 전이학습 체크리스트

**Phase 완료 시**:
- [ ] 최고 성능 모델 저장
- [ ] 성능 지표 기록
- [ ] 다음 Phase 설정 준비

**Phase 시작 시**:
- [ ] 이전 Phase 모델 로드
- [ ] 학습률 조정
- [ ] 새로운 목표 설정
- [ ] 전이학습 성공 확인

---

## 구현 순서 및 체크리스트

### 현재 단계: Phase 0 - 기본 안정화

#### ✅ 체크리스트

**1. 현재 시스템 상태 파악 (1일)**
- [ ] 현재 코드 실행 후 로그 파일 확인
- [ ] episode_rewards.csv 파일 분석
- [ ] 현재 RMSE, settling time, overshoot 값 확인
- [ ] 학습이 진행 중인지, 정체되었는지 판단

**2. 성능 기준 명확화 (1일)**
- [ ] 목표 RMSE 결정: ___ N
- [ ] 목표 Settling Time 결정: ___ 초
- [ ] 목표 Overshoot 비율 결정: ___ %
- [ ] 안정성 기준 결정: 에피소드 ___ 개 이상 지속

**3. 학습 진행 모니터링 (일일)**
```python
# 매일 확인할 지표
- 현재 에피소드: ___
- 최근 10 에피소드 평균 RMSE: ___
- 최근 10 에피소드 평균 Reward: ___
- 학습 추세: 증가 / 정체 / 감소
```

**4. 문제 발생 시 대응**
- [ ] 성능이 개선되지 않는 경우: 하이퍼파라미터 조정
- [ ] 발산하는 경우: 학습률 감소, 재시작
- [ ] 진동 발생: PID 범위 조정 또는 보상 함수 수정

### 다음 단계 예정

**Phase 1 달성 시**:
- [ ] Phase 2로 진행할지 결정
- [ ] 학습된 모델 저장 및 백업
- [ ] 성능 리포트 작성

**Phase 2 달성 시**:
- [ ] Phase 3로 진행할지 결정
- [ ] 멀티-스테이지 시스템 설계
- [ ] 기존 코드와 통합 방법 검토

**Phase 3 달성 시**:
- [ ] 최종 일반화 성능 검증
- [ ] 다양한 시나리오 테스트
- [ ] 최종 리포트 작성

---

## 주의사항

### 1. 안정성 우선
- 각 단계마다 충분한 검증
- 이전 단계가 안정적이면 다음 단계로

### 2. 데이터 수집
- 모든 실험 결과 기록
- 실패 케이스도 분석

### 3. 점진적 확장
- 한 번에 모든 것을 바꾸지 말 것
- Incremental development 권장

### 4. 로봇 안전
- 진폭이나 속도 제한 확인
- 이상 탐지 및 안전 모드 포함

---

## 추천 참고 자료

1. **Curriculum Learning**: 쉬운 목표 → 어려운 목표로 점진적 학습
2. **Meta-Learning**: 빠른 어댑테이션을 위한 MAML 등
3. **Domain Randomization**: 다양한 조건에서 일반화 학습
4. **Adaptive Control Theory**: 적응적 제어 이론 참고

---

## 예상 전체 소요 시간
- **Phase 0**: 2-3주 (기본 안정화) ← **현재 단계**
- **Phase 1**: 2-3주 (고정 목표에서 성능 확보)
- **Phase 2**: 2-3주 (30-50N 범위로 확장)
- **Phase 3**: 3-4주 (에피소드 내 목표 변경)
- **Phase 4**: 2-3주 (알고리즘 개선)
- **Phase 5**: 1-2주 (최종 검증)

**총 예상**: 12-18주 (약 3-4개월)

## 다음 할 일 (즉시)

1. **현재 학습 상태 확인**
   ```bash
   # 로그 디렉토리로 이동
   cd py_rl/pid_gain_rl/experiment_logs/
   
   # 최근 로그 파일 확인
   ls -lth
   
   # 에피소드별 보상 확인
   tail -20 episode_rewards.csv
   ```

2. **성능 지표 추출**
   - 에피소드별 RMSE 추이 그래프
   - Settling time 추이 그래프
   - 현재까지의 최고 성능 확인

3. **학습 진행 상황 판단**
   - 수렴 중: 계속 진행
   - 정체: 하이퍼파라미터 조정
   - 발산: 재시작 필요
