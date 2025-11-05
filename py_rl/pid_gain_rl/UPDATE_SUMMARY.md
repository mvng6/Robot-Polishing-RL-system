# 📋 업데이트 요약 (우선순위별)

## 🔴 최우선: 모순 수정 및 핵심 설정 변경

### 1. PID 범위 변경 (constants.py)
**파일**: `constants.py`  
**위치**: `DEFAULT_PID_RANGE` (16-20줄)

**현재:**
```python
DEFAULT_PID_RANGE = {
    "Kp": (35.0, 45.0),
    "Ki": (45.0, 55.0),
    "Kd": (1e-6, 1e-3),
}
```

**변경:**
```python
DEFAULT_PID_RANGE = {
    "Kp": (3.0, 40.0),      # 하한 확장: 35 → 3
    "Ki": (0.0, 40.0),      # 하한 0까지, 상한 낮춤: 55 → 40
    "Kd": (1e-4, 5e-2),     # 대폭 확장: 1e-6~1e-3 → 1e-4~5e-2 (로그-스페이스)
}
```

**검증**: 모든 문서에서 Kd 범위 통일 확인

---

### 2. 보상 함수 튜닝 (constants.py)
**파일**: `constants.py`  
**위치**: 스코어화 기반 보상 시스템 (97-109줄)

**변경:**
```python
# 현재
SCORE_W_MP = 0.25
SCORE_TAU_MP_PERCENT = 12.0

# 변경
SCORE_W_MP = 0.35           # 0.25 → 0.35 (오버슈트 가중치 강화)
SCORE_TAU_MP_PERCENT = 8.0  # 12.0 → 8.0 (더 민감하게, 8~10 범위)
```

---

### 3. 탐색/탐욕 비율 조정 (constants.py)
**파일**: `constants.py`  
**위치**: Fine-tuning 설정 (121-130줄)

**변경:**
```python
# 현재
ACTOR_LOG_STD_MAX = -1.0  # σ ≤ 0.37 (탐색 범위 좁음)
ACTOR_INITIAL_ALPHA = 0.02  # 고정값 (매우 작음)
STD_ANNEAL_FINAL = 0.3  # 추가 축소
STD_ANNEAL_END_EPISODE = 200

# 변경
ACTOR_LOG_STD_MAX = -0.3  # -1.0 → -0.3 (더 큰 탐색 범위)
ACTOR_INITIAL_ALPHA = 0.1  # 0.02 → 0.1 (더 큰 초기값)
STD_ANNEAL_FINAL = 0.5  # 0.3 → 0.5 (덜 축소)
STD_ANNEAL_END_EPISODE = 150  # 200 → 150 (더 빠르게)
```

**추가 필요**: `ACTOR_LOG_STD_MIN = -2.5` (하한 보장)

---

### 4. 학습 하이퍼파라미터 조정 (constants.py)
**파일**: `constants.py`  
**위치**: 통신 기본 설정 (29줄)

**변경:**
```python
# 현재
DEFAULT_UPDATES_PER_EPISODE = 10

# 변경
DEFAULT_UPDATES_PER_EPISODE = 35  # 10 → 35 (30~40 범위, 세그먼트 분할 대비)
```

---

## 🟡 즉시 구현: 코드 로직 추가

### 5. 초기 구간 피크 패널티 추가 (env.py)
**파일**: `env.py`  
**위치**: `calculate_episode_reward` 메서드 내부

**추가할 함수:**
```python
def _calculate_early_peak_penalty(self, force_data, target_force, episode_time, dt):
    """0~0.5초 구간 전용 피크 패널티"""
    if episode_time < 0.5:
        if target_force < 0:
            peak_force = float(np.min(force_data))
        else:
            peak_force = float(np.max(force_data))
        peak_detector = max(0.0, abs(peak_force - target_force) / abs(target_force))
        penalty = min(peak_detector * 0.2, 0.2)  # 상한 0.15~0.2
        return penalty
    return 0.0
```

**통합 위치**: `calculate_episode_reward` 메서드의 보상 계산 부분

---

### 6. Target Entropy 동적 조정 (agent.py)
**파일**: `agent.py`  
**위치**: `__init__` 메서드 및 새 메서드 추가

**변경 사항:**
1. `__init__`에서 target_entropy 초기값 설정
2. `update_target_entropy` 메서드 추가
3. 에피소드마다 호출하여 조정

**코드:**
```python
# __init__ 메서드 수정
if self.auto_entropy_tuning:
    self.action_dim_for_entropy = a_dim
    # 초기 100ep: 더 공격적 탐색
    self.target_entropy_initial = -1.2 * a_dim  # -3.6 (3차원)
    self.target_entropy_final = -1.0 * a_dim    # -3.0 (3차원)
    self.target_entropy = self.target_entropy_initial
    # ... 기존 코드

# 새 메서드 추가
def update_target_entropy(self, episode_num):
    """초기 100ep 동안 target_entropy를 점진적으로 조정"""
    if episode_num < 100:
        progress = episode_num / 100.0
        self.target_entropy = (
            self.target_entropy_initial + 
            progress * (self.target_entropy_final - self.target_entropy_initial)
        )
    else:
        self.target_entropy = self.target_entropy_final
```

**호출 위치**: `env.py`의 에피소드 루프에서 `agent.update_target_entropy(episode_num)` 호출

---

### 7. Warm-start 버퍼 초기화 (agent.py 또는 env.py)
**파일**: `agent.py` 또는 `env.py`  
**위치**: 에피소드 루프 시작 전

**추가할 함수:**
```python
def warm_start_buffer(self, num_samples=50):
    """라틴 하이퍼큐브 샘플링으로 버퍼 초기화"""
    try:
        from scipy.stats import qmc
    except ImportError:
        print("⚠️ scipy 없음, 랜덤 샘플링 사용")
        # 랜덤 샘플링으로 대체
        for _ in range(num_samples):
            action_np = np.random.uniform(-1.0, 1.0, size=3)
            pid_gains = scale_action_to_pid(action_np, self.cfg["PID_RANGE"])
            # 더미 transition 저장
            dummy_state = np.zeros(20, dtype=np.float32)
            dummy_next_state = np.zeros(20, dtype=np.float32)
            dummy_action = np.array([
                (pid_gains[0] - self.cfg["PID_RANGE"]["Kp"][0]) / 
                (self.cfg["PID_RANGE"]["Kp"][1] - self.cfg["PID_RANGE"]["Kp"][0]) * 2 - 1,
                (pid_gains[1] - self.cfg["PID_RANGE"]["Ki"][0]) / 
                (self.cfg["PID_RANGE"]["Ki"][1] - self.cfg["PID_RANGE"]["Ki"][0]) * 2 - 1,
                # Kd는 로그 스케일로 정규화
            ], dtype=np.float32)
            dummy_reward = 0.0
            dummy_done = False
            self.replay.push(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
        return
    
    # LHS 샘플링
    sampler = qmc.LatinHypercube(d=3)
    samples = sampler.random(n=num_samples)
    
    # Kp, Ki: 선형, Kd: 로그
    kp_samples = samples[:, 0] * (40 - 3) + 3
    ki_samples = samples[:, 1] * (40 - 0) + 0
    kd_log_samples = samples[:, 2] * (np.log10(5e-2) - np.log10(1e-4)) + np.log10(1e-4)
    kd_samples = 10 ** kd_log_samples
    
    # 각 샘플에 대해 더미 transition 저장
    for kp, ki, kd in zip(kp_samples, ki_samples, kd_samples):
        # 더미 상태/보상 생성
        dummy_state = np.zeros(20, dtype=np.float32)
        dummy_next_state = np.zeros(20, dtype=np.float32)
        # PID를 액션으로 변환
        action_np = np.array([
            (kp - self.cfg["PID_RANGE"]["Kp"][0]) / 
            (self.cfg["PID_RANGE"]["Kp"][1] - self.cfg["PID_RANGE"]["Kp"][0]) * 2 - 1,
            (ki - self.cfg["PID_RANGE"]["Ki"][0]) / 
            (self.cfg["PID_RANGE"]["Ki"][1] - self.cfg["PID_RANGE"]["Ki"][0]) * 2 - 1,
            # Kd는 로그 스케일로 정규화 (scale_action_to_pid의 역함수)
            (np.log10(kd) - np.log10(self.cfg["PID_RANGE"]["Kd"][0])) / 
            (np.log10(self.cfg["PID_RANGE"]["Kd"][1]) - np.log10(self.cfg["PID_RANGE"]["Kd"][0])) * 2 - 1,
        ], dtype=np.float32)
        dummy_reward = 0.0
        dummy_done = False
        self.replay.push(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
```

**호출 위치**: `env.py`의 `run_pid_optimization_training` 메서드 시작 부분

---

## 🟢 모니터링 및 로깅 추가

### 8. 보상 구성 요소 시각화 추가 (loggers/reward_breakdown.py)
**파일**: `loggers/reward_breakdown.py`  
**위치**: `generate_episode_reward_graph` 메서드 수정 또는 새 메서드 추가

**추가할 시각화:**
1. reward_score, r_centered, r_baseline 동일 x축 플롯
2. overshoot, settling_time, band_ratio와 함께 플롯
3. Moving average (10-ep) 적용

**새 메서드:**
```python
def generate_reward_components_graph(self, reward_data, performance_data):
    """보상 구성 요소와 성능 지표를 함께 시각화"""
    # reward_data: dict with keys ['reward_score', 'r_centered', 'r_baseline']
    # performance_data: dict with keys ['overshoot', 'settling_time', 'band_ratio']
    # Moving average (10-ep) 계산 및 플롯
```

---

### 9. 행동분포 가드 로깅 (agent.py)
**파일**: `agent.py`  
**위치**: 새 메서드 추가

**추가할 메서드:**
```python
def log_exploration_metrics(self, episode_num):
    """탐색 효과 모니터링"""
    # 최근 20 에피소드의 action std 계산
    # action std/range 비율 계산
    # Kd decade 커버리지 계산
    # CSV로 저장
```

**호출 위치**: `env.py`의 에피소드 루프에서 주기적으로 호출

---

### 10. 세그먼트별 지표 로깅 (loggers/control_performance.py)
**파일**: `loggers/control_performance.py`  
**위치**: 새 메서드 추가

**추가할 기능:**
- 세그먼트별 overshoot, settling_time, band_ratio 계산
- CSV로 저장 (segment_metrics.csv)

---

## 🔵 보상 중심화 A/B 실험 준비

### 11. 보상 중심화 실험 플래그 (env.py)
**파일**: `env.py`  
**위치**: `__init__` 메서드 및 `calculate_episode_reward` 메서드

**추가할 설정:**
```python
# __init__에 추가
self.reward_experiment_mode = "default"  # "default", "centered_off", "beta_095", "tanh_15"

# calculate_episode_reward에서 분기 처리
if self.reward_experiment_mode == "centered_off":
    reward = reward_score  # 원시 사용
elif self.reward_experiment_mode == "beta_095":
    beta = 0.95
    # ... 기존 로직
elif self.reward_experiment_mode == "tanh_15":
    gain = 1.5
    # ... 기존 로직
else:
    # 기존 로직 (beta=0.99, gain=2.0)
```

**설정 방법**: `config.py` 또는 명령줄 인자로 추가

---

## 📊 파일별 수정 요약

### constants.py
- [x] DEFAULT_PID_RANGE 변경
- [x] SCORE_W_MP, SCORE_TAU_MP_PERCENT 변경
- [x] ACTOR_LOG_STD_MAX, ACTOR_INITIAL_ALPHA 변경
- [x] STD_ANNEAL_FINAL, STD_ANNEAL_END_EPISODE 변경
- [x] DEFAULT_UPDATES_PER_EPISODE 변경
- [x] ACTOR_LOG_STD_MIN 추가

### agent.py
- [ ] target_entropy 동적 조정 메서드 추가
- [ ] warm_start_buffer 메서드 추가
- [ ] log_exploration_metrics 메서드 추가
- [ ] __init__에서 target_entropy 초기값 설정 수정

### env.py
- [ ] _calculate_early_peak_penalty 메서드 추가
- [ ] calculate_episode_reward에 초기 구간 피크 패널티 통합
- [ ] 에피소드 루프에서 agent.update_target_entropy 호출
- [ ] 에피소드 루프 시작 전 warm_start_buffer 호출
- [ ] 에피소드 루프에서 log_exploration_metrics 호출
- [ ] 보상 중심화 실험 플래그 추가 (선택적)

### loggers/reward_breakdown.py
- [ ] generate_reward_components_graph 메서드 추가
- [ ] Moving average (10-ep) 계산 및 플롯

### loggers/control_performance.py
- [ ] 세그먼트별 지표 계산 메서드 추가
- [ ] segment_metrics.csv 저장 기능 추가

### config.py (선택적)
- [ ] 보상 중심화 실험 모드 필드 추가

---

## 🎯 구현 순서 (권장)

### Phase 1: 즉시 적용 (코드 수정)
1. constants.py 수정 (PID 범위, 보상 가중치, 탐색 설정)
2. env.py에 초기 구간 피크 패널티 추가
3. agent.py에 target_entropy 동적 조정 추가
4. agent.py에 warm_start_buffer 추가
5. env.py에서 위 메서드들 호출

### Phase 2: 모니터링 추가
6. loggers/reward_breakdown.py에 보상 구성 요소 그래프 추가
7. agent.py에 log_exploration_metrics 추가
8. loggers/control_performance.py에 세그먼트별 지표 추가

### Phase 3: 실험 준비 (선택적)
9. 보상 중심화 A/B 실험 플래그 추가
10. 실험 매트릭스 준비

---

## ⚠️ 주의사항

1. **STD_ANNEAL과 log_std_max 중복**: 현재는 STD_ANNEAL만 조정하는 방식으로 해결 (옵션 2)

2. **컨트롤러 선조치**: 로봇제어PC 측 구현이 필요하므로, Python 코드에서는 로깅만 추가

3. **오버슈트 정의 통일**: 이미 구현되어 있지만, 검증 필요

4. **Warm-start**: scipy 의존성 확인 (없으면 랜덤 샘플링으로 대체)

---

## ✅ 검증 체크리스트

### 코드 수정 후
- [ ] PID 범위가 모든 파일에서 통일되어 있는지 확인
- [ ] 보상 계산에 초기 구간 피크 패널티가 통합되었는지 확인
- [ ] target_entropy가 에피소드마다 업데이트되는지 확인
- [ ] Warm-start가 버퍼 초기화 시 호출되는지 확인

### 실행 후
- [ ] 보상 구성 요소 그래프가 생성되는지 확인
- [ ] 탐색 메트릭이 로그에 기록되는지 확인
- [ ] 세그먼트별 지표가 CSV에 저장되는지 확인

---

준비되면 수정을 시작하겠습니다!

