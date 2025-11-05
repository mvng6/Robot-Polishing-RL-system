# 🛠️ 커리큘럼 학습 구현 가이드

## 1. constants.py에 커리큘럼 설정 추가

```python
# constants.py에 추가할 내용

class CurriculumPhase:
    """커리큘럼 학습 단계 정의"""
    PHASE_1_FIXED = 1           # 고정 -40N
    PHASE_2_RANDOM_EPISODE = 2  # 에피소드마다 무작위
    PHASE_3_ONE_CHANGE = 3      # 중간 1회 변경 (10초)
    PHASE_4_TWO_CHANGES = 4     # 중간 2회 변경 (10초, 20초) - 최종
    PHASE_5_FINAL = 5           # 최종 목표 (3회 변경)

class Constants:
    # ... 기존 코드 ...
    
    # ===== 커리큘럼 학습 설정 =====
    CURRENT_PHASE = CurriculumPhase.PHASE_1_FIXED  # 현재 단계
    
    # 목표 접촉력 범위
    TARGET_FORCE_MIN = -50.0
    TARGET_FORCE_MAX = -35.0
    TARGET_FORCE_DEFAULT = -40.0
    
    # Phase별 설정
    PHASE_2_EPISODE_LENGTH = 10.0
    PHASE_3_EPISODE_LENGTH = 20.0
    PHASE_4_EPISODE_LENGTH = 30.0
    
    # 목표 변경 시점 (초)
    PHASE_3_CHANGE_TIME = 10.0
    PHASE_4_CHANGE_TIMES = [10.0, 20.0]  # 2회 변경
    PHASE_5_CHANGE_TIMES = [0.0, 10.0, 20.0]  # 3회 변경 (시작 포함)
    
    # Phase 전환 조건
    PHASE_TRANSITION_MIN_EPISODES = 50  # 최소 에피소드 수
    PHASE_TRANSITION_MIN_REWARD = 0.3   # 최소 평균 보상
```

---

## 2. config.py에 커리큘럼 설정 추가

```python
# config.py에 추가할 내용

@dataclass
class Config:
    # ... 기존 코드 ...
    
    # 커리큘럼 학습 설정
    curriculum_phase: int = Constants.CURRENT_PHASE
    target_force_min: float = Constants.TARGET_FORCE_MIN
    target_force_max: float = Constants.TARGET_FORCE_MAX
    dynamic_target_enabled: bool = False  # 동적 목표 활성화 여부
    target_change_times: List[float] = field(default_factory=list)  # 변경 시점 리스트
```

---

## 3. env.py에 동적 목표 관리 클래스 추가

```python
# env.py 상단에 추가

class DynamicTargetManager:
    """동적 목표 접촉력 관리 클래스"""
    
    def __init__(self, phase: int, target_min: float, target_max: float, 
                 episode_length: float, change_times: List[float]):
        self.phase = phase
        self.target_min = target_min
        self.target_max = target_max
        self.episode_length = episode_length
        self.change_times = sorted(change_times)  # 정렬된 변경 시점
        
        # 현재 에피소드의 목표 히스토리
        self.current_episode_targets = []
        self.current_target_idx = 0
        
    def start_episode(self):
        """에피소드 시작 시 목표 시퀀스 생성"""
        if self.phase == CurriculumPhase.PHASE_1_FIXED:
            # Phase 1: 고정 목표
            self.current_episode_targets = [Constants.TARGET_FORCE_DEFAULT]
            
        elif self.phase == CurriculumPhase.PHASE_2_RANDOM_EPISODE:
            # Phase 2: 에피소드마다 무작위 (에피소드 내부는 고정)
            target = random.uniform(self.target_min, self.target_max)
            self.current_episode_targets = [target]
            
        elif self.phase == CurriculumPhase.PHASE_3_ONE_CHANGE:
            # Phase 3: 1회 변경 (10초)
            target_1 = random.uniform(self.target_min, self.target_max)
            target_2 = random.uniform(self.target_min, self.target_max)
            self.current_episode_targets = [target_1, target_2]
            
        elif self.phase == CurriculumPhase.PHASE_4_TWO_CHANGES:
            # Phase 4: 2회 변경 (10초, 20초)
            target_1 = random.uniform(self.target_min, self.target_max)
            target_2 = random.uniform(self.target_min, self.target_max)
            target_3 = random.uniform(self.target_min, self.target_max)
            self.current_episode_targets = [target_1, target_2, target_3]
            
        elif self.phase == CurriculumPhase.PHASE_5_FINAL:
            # Phase 5: 3회 변경 (0초, 10초, 20초)
            target_1 = random.uniform(self.target_min, self.target_max)
            target_2 = random.uniform(self.target_min, self.target_max)
            target_3 = random.uniform(self.target_min, self.target_max)
            self.current_episode_targets = [target_1, target_2, target_3]
        
        self.current_target_idx = 0
        return self.current_episode_targets[0]
    
    def get_current_target(self, episode_time: float) -> float:
        """현재 시간에 맞는 목표 접촉력 반환"""
        if not self.change_times:
            # 변경 시점 없음 (Phase 1, 2)
            return self.current_episode_targets[0]
        
        # 변경 시점에 따라 목표 인덱스 업데이트
        for i, change_time in enumerate(self.change_times):
            if episode_time >= change_time:
                if i + 1 < len(self.current_episode_targets):
                    self.current_target_idx = i + 1
        
        return self.current_episode_targets[self.current_target_idx]
    
    def get_all_targets(self) -> List[float]:
        """에피소드의 모든 목표 리스트 반환"""
        return self.current_episode_targets.copy()
    
    def get_target_history(self) -> Dict[str, Any]:
        """목표 히스토리 정보 반환 (상태 공간용)"""
        return {
            "current_target": self.current_episode_targets[self.current_target_idx],
            "previous_targets": self.current_episode_targets[:self.current_target_idx],
            "next_targets": self.current_episode_targets[self.current_target_idx+1:],
            "target_idx": self.current_target_idx,
            "num_targets": len(self.current_episode_targets),
        }
```

---

## 4. env.py의 PIDGainEnvironment 클래스 수정

```python
# env.py의 __init__ 메서드에 추가

def __init__(self, cfg=None):
    # ... 기존 코드 ...
    
    # ==== 커리큘럼 학습: 동적 목표 관리 ====
    phase = cfg.get("CURRICULUM_PHASE", Constants.CURRENT_PHASE)
    episode_length = cfg.get("EPISODE_SECONDS", Constants.DEFAULT_EPISODE_SECONDS)
    change_times = []
    
    if phase == CurriculumPhase.PHASE_3_ONE_CHANGE:
        change_times = [Constants.PHASE_3_CHANGE_TIME]
    elif phase == CurriculumPhase.PHASE_4_TWO_CHANGES:
        change_times = Constants.PHASE_4_CHANGE_TIMES
    elif phase == CurriculumPhase.PHASE_5_FINAL:
        change_times = Constants.PHASE_5_CHANGE_TIMES
    
    self.target_manager = DynamicTargetManager(
        phase=phase,
        target_min=cfg.get("TARGET_FORCE_MIN", Constants.TARGET_FORCE_MIN),
        target_max=cfg.get("TARGET_FORCE_MAX", Constants.TARGET_FORCE_MAX),
        episode_length=episode_length,
        change_times=change_times,
    )
    
    self.dynamic_target_enabled = (phase >= CurriculumPhase.PHASE_2_RANDOM_EPISODE)
```

---

## 5. 데이터 수집 루프 수정 (env.py의 run_pid_optimization_training)

```python
# 기존 데이터 수집 루프 (약 641줄 부근) 수정

# 에피소드 시작 시 목표 초기화
current_episode_target = self.target_manager.start_episode()
print(f"🎯 에피소드 시작 - 목표 접촉력: {current_episode_target:.1f}N")

# 목표 히스토리 로깅
if len(self.target_manager.get_all_targets()) > 1:
    targets_str = " → ".join([f"{t:.1f}N" for t in self.target_manager.get_all_targets()])
    print(f"📋 목표 시퀀스: {targets_str}")

while (time.perf_counter() - start_time) < self.cfg["EPISODE_SECONDS"]:
    episode_time = time.perf_counter() - start_time
    
    # 동적 목표 업데이트
    if self.dynamic_target_enabled:
        current_episode_target = self.target_manager.get_current_target(episode_time)
        
        # 목표 변경 감지 및 로깅
        if episode_time > 0.0:
            prev_target = self.target_manager.current_episode_targets[
                max(0, self.target_manager.current_target_idx - 1)
            ]
            if abs(current_episode_target - prev_target) > 0.1:
                print(f"🔄 목표 변경: {prev_target:.1f}N → {current_episode_target:.1f}N (시간: {episode_time:.1f}초)")
    else:
        # Phase 1: 고정 목표
        current_episode_target = self.cfg["TARGET_FORCE"]
    
    state, sander_active = self.comm.get_latest_state()
    if state is None:
        time.sleep(0.001)
        continue

    self.episode_force_data.append(state[0])
    self.episode_pi_output_data.append(state[5])
    data_count += 1
    
    # ... 기존 안전 체크 코드 ...
```

---

## 6. 보상 계산 시 동적 목표 사용 (env.py의 calculate_episode_reward)

```python
# calculate_episode_reward 메서드 수정
# target_force 파라미터를 동적으로 받도록 변경

def calculate_episode_reward(
    self, force_data, pi_output_data, target_force=None, episode_len_s=None,
    target_sequence=None, change_times=None
):
    """
    Args:
        target_sequence: 목표 히스토리 리스트 [target1, target2, ...]
        change_times: 목표 변경 시점 리스트 [10.0, 20.0, ...]
    """
    # 세그먼트별 보상 계산 시 각 세그먼트의 목표 힘 사용
    if target_sequence and change_times:
        # 구간별로 다른 목표 사용
        # ... 세그먼트 분할 시 각 구간의 목표 힘 적용
    else:
        # 기존 로직 (단일 목표)
        if target_force is None:
            target_force = self.cfg["TARGET_FORCE"]
    
    # ... 기존 보상 계산 코드 ...
```

---

## 7. 세그먼트별 보상 계산 수정

```python
# _build_segment_state 또는 세그먼트 보상 계산 부분 수정

for seg_idx in range(Constants.NUM_SEGMENTS):
    seg_start = seg_idx * Constants.SEGMENT_LENGTH_S
    seg_end = (seg_idx + 1) * Constants.SEGMENT_LENGTH_S
    
    # 해당 세그먼트의 목표 힘 결정
    if self.dynamic_target_enabled and change_times:
        seg_target = self.target_manager.get_current_target(seg_start)
    else:
        seg_target = current_episode_target
    
    seg_force = force_data[seg_start_idx:seg_end_idx]
    seg_pi = pi_output_data[seg_start_idx:seg_end_idx]
    
    # 세그먼트별 보상 계산
    seg_reward, seg_metrics = self.calculate_episode_reward(
        seg_force, seg_pi,
        target_force=seg_target,  # 세그먼트별 목표
        episode_len_s=Constants.SEGMENT_LENGTH_S
    )
```

---

## 8. 상태 공간 확장 (utils/math_utils.py)

```python
# create_initial_state 함수 수정 또는 새 함수 추가

def create_initial_state_with_target_history(
    force_data, target_manager, previous_pid_gains=None, 
    historical_errors=None, episode_history=None, dt_sec=None
):
    """목표 히스토리를 포함한 상태 생성"""
    base_state = create_initial_state(
        force_data, target_manager.get_current_target(0.0),
        previous_pid_gains, historical_errors, episode_history, dt_sec
    )
    
    # 목표 히스토리 정보 추가 (4차원)
    target_info = target_manager.get_target_history()
    target_features = [
        target_info["current_target"] / 50.0,  # 정규화
        target_info["target_idx"] / 3.0,      # 정규화 (최대 3개)
        len(target_info["previous_targets"]) / 3.0,  # 이전 목표 개수
        len(target_info["next_targets"]) / 3.0,     # 다음 목표 개수
    ]
    
    # 기존 20차원 + 목표 히스토리 4차원 = 24차원
    extended_state = np.concatenate([base_state, target_features])
    return extended_state
```

---

## 9. Phase 전환 로직 (선택적, 자동 전환)

```python
# env.py에 추가

def check_phase_transition(self, episode: int, avg_reward: float) -> bool:
    """Phase 전환 조건 확인"""
    if episode < Constants.PHASE_TRANSITION_MIN_EPISODES:
        return False
    
    if avg_reward < Constants.PHASE_TRANSITION_MIN_REWARD:
        return False
    
    return True

def advance_phase(self):
    """다음 Phase로 전환"""
    current_phase = self.target_manager.phase
    
    if current_phase < CurriculumPhase.PHASE_5_FINAL:
        new_phase = current_phase + 1
        print(f"\n{'='*60}")
        print(f"🎓 Phase 전환: {current_phase} → {new_phase}")
        print(f"{'='*60}\n")
        
        # Phase별 에피소드 길이 조정
        if new_phase == CurriculumPhase.PHASE_3_ONE_CHANGE:
            self.cfg["EPISODE_SECONDS"] = Constants.PHASE_3_EPISODE_LENGTH
        elif new_phase == CurriculumPhase.PHASE_4_TWO_CHANGES:
            self.cfg["EPISODE_SECONDS"] = Constants.PHASE_4_EPISODE_LENGTH
        
        # Target Manager 재초기화
        self.target_manager = DynamicTargetManager(...)
        
        return True
    return False
```

---

## 10. 사용 예시

### Phase 1 (현재 상태, 변경 없음)
```python
# constants.py
CURRENT_PHASE = CurriculumPhase.PHASE_1_FIXED
```

### Phase 2로 전환
```python
# constants.py
CURRENT_PHASE = CurriculumPhase.PHASE_2_RANDOM_EPISODE

# 실행
python3 -m pid_gain_rl
```

### Phase 4로 직접 이동 (테스트용)
```python
# constants.py
CURRENT_PHASE = CurriculumPhase.PHASE_4_TWO_CHANGES
```

---

## 11. 체크리스트

### Phase 2 구현
- [ ] `constants.py`에 `CurriculumPhase` 클래스 추가
- [ ] `config.py`에 커리큘럼 설정 필드 추가
- [ ] `DynamicTargetManager` 클래스 구현
- [ ] 에피소드 시작 시 목표 선택 로직
- [ ] 상태 공간에 목표 정보 포함 (선택적)

### Phase 3 구현
- [ ] 에피소드 길이 20초로 조정
- [ ] 10초 시점 목표 변경 로직
- [ ] 목표 변경 감지 및 로깅
- [ ] 세그먼트별 보상 계산에 목표 반영

### Phase 4 구현
- [ ] 에피소드 길이 30초로 조정
- [ ] 10초, 20초 시점 목표 변경
- [ ] 세그먼트 분할 재조정 (15개)
- [ ] 통신 프로토콜 확인 (목표 변경 신호 필요 시)

---

## 📝 구현 순서 권장사항

1. **Phase 2부터 시작** (가장 간단, 기존 코드와 호환성 좋음)
2. 단계별 테스트 및 검증
3. Phase 3, 4 순차 구현
4. 각 Phase 완료 후 성능 검증




