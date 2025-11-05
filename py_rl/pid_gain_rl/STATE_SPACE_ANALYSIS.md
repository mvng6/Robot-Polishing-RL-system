# 🔍 20차원 State Space 분석 및 최적화

## 📊 현재 State Space 구성

### 전체 구조 (20차원)

#### 1. Base State (12차원)
```
[0-2]   Kp, Ki, Kd                    # 현재 PID 게인
[3]     prev_reward                   # 이전 보상
[4]     target_force                  # 목표 힘
[5]     segment_len_s                 # 세그먼트 길이 (초)
[6-11]  Force 통계 (6차원):
        - mean(force)                 # 평균 힘
        - std(force)                  # 표준편차
        - min(force)                  # 최소 힘
        - max(force)                  # 최대 힘
        - mean(error)                 # 평균 오차
        - std(error)                  # 오차 표준편차
```

#### 2. Trajectory Features (8차원)
```
[12]    overshoot                     # 오버슈트 (%)
[13]    settling_time                 # 정착 시간 (초)
[14]    rmse                          # RMSE
[15]    band_ratio                    # 밴드 유지 비율
[16]    oscillation_freq              # 진동 주파수 (FFT)
[17]    oscillation_amp               # 진동 진폭 (FFT)
[18]    rise_time                     # 상승 시간 (10%→90%)
[19]    steady_state_error            # 정상 상태 오차
```

---

## ❌ 현재 State Space의 문제점

### 1. **초기 압력 급격한 변화 감지 불가** ⚠️ **매우 심각**

**문제:**
- 초기 접촉 압력 0 → 0.3MPa 급격한 변화를 감지할 정보가 **없음**
- 힘의 변화율(force rate, dF/dt) 정보가 없음
- 초기 구간 피크 정보가 별도로 없음 (overshoot는 전체 구간 기준)

**결과:**
- 에이전트가 초기 급격한 변화를 인지하지 못함
- 초기 오버슈트를 예방하는 PID 게인을 학습하기 어려움

---

### 2. **시간적 정보 부족**

**문제:**
- 힘의 통계 정보만 있어 시간적 변화를 인지하기 어려움
- 초기 구간과 후기 구간을 구분하는 정보가 없음
- 힘의 변화 추세 정보가 없음

---

### 3. **초기 구간 특화 정보 부족**

**문제:**
- 초기 구간(0~0.5초) 피크 정보가 별도로 없음
- 초기 구간 RMSE가 별도로 없음
- 초기 구간 오차 변화율 정보가 없음

---

## ✅ 개선 방안

### 방안 1: 초기 구간 정보 추가 (권장) 🎯

**목표:** 초기 압력 급격한 변화를 감지하고 대응

**추가할 State (3-4차원):**
```
[20]    early_peak_force              # 초기 구간(0~0.5초) 최대/최소 힘
[21]    early_force_rate              # 초기 구간 힘 변화율 (dF/dt 평균)
[22]    early_rmse                    # 초기 구간 RMSE
[23]    initial_force                 # 세그먼트 시작 힘 (또는 에피소드 시작 힘)
```

**구현:**
```python
# 초기 구간 (0~0.5초 또는 첫 20% 구간)
early_window_s = 0.5
early_samples = int(early_window_s * fs_hz)
early_force = force[:min(early_samples, len(force))]

# 초기 구간 피크
if target_force < 0:
    early_peak_force = float(np.min(early_force))
else:
    early_peak_force = float(np.max(early_force))

# 초기 구간 힘 변화율
if len(early_force) > 1:
    early_force_rate = float((early_force[-1] - early_force[0]) / early_window_s)
else:
    early_force_rate = 0.0

# 초기 구간 RMSE
early_errors = early_force - target_force
early_rmse = float(np.sqrt(np.mean(early_errors**2)))

# 초기 힘 값
initial_force = float(force[0])
```

**예상 효과:**
- ✅ 초기 압력 급격한 변화를 명확히 감지 가능
- ✅ 초기 오버슈트를 예방하는 PID 게인 학습 가능
- ✅ State space: 20 → 24차원

---

### 방안 2: 힘 변화율 정보 추가

**추가할 State (2차원):**
```
[20]    force_rate_mean               # 힘 변화율 평균 (dF/dt)
[21]    force_rate_max                # 힘 변화율 최대값
```

**구현:**
```python
# 힘 변화율 계산
force_diff = np.diff(force)
dt = 1.0 / fs_hz
force_rate = force_diff / dt

force_rate_mean = float(np.mean(force_rate))
force_rate_max = float(np.max(np.abs(force_rate)))
```

**예상 효과:**
- ✅ 힘의 변화 추세를 인지 가능
- ✅ 급격한 변화를 감지 가능
- ✅ State space: 20 → 22차원

---

### 방안 3: 초기 구간 + 변화율 결합 (권장) 🎯

**추가할 State (5차원):**
```
[20]    early_peak_force              # 초기 구간 피크 힘
[21]    early_force_rate              # 초기 구간 힘 변화율
[22]    early_rmse                    # 초기 구간 RMSE
[23]    initial_force                 # 세그먼트 시작 힘
[24]    force_rate_mean               # 전체 구간 힘 변화율 평균
```

**예상 효과:**
- ✅ 초기 구간 특화 정보 제공
- ✅ 힘 변화 추세 정보 제공
- ✅ 초기 압력 급격한 변화 감지 가능
- ✅ State space: 20 → 25차원

---

## 📋 구현 우선순위

### Phase 1: 즉시 적용 (초기 구간 정보 추가)
1. ✅ `_build_segment_state()` 메서드 수정
2. ✅ 초기 구간(0~0.5초) 피크 힘 계산
3. ✅ 초기 구간 힘 변화율 계산
4. ✅ 초기 구간 RMSE 계산
5. ✅ 초기 힘 값 추가
6. ✅ STATE_DIM: 20 → 24차원 업데이트

### Phase 2: 추가 개선 (필요 시)
1. 전체 구간 힘 변화율 정보 추가
2. 힘 변화율 최대값 추가
3. State space: 24 → 26차원

---

## 🔧 구현 코드 예시

```python
def _build_segment_state(self, prev_pid_gains, prev_reward, force_segment, target_force, segment_len_s):
    # ... 기존 코드 ...
    
    # 🆕 초기 구간 정보 추가 (0~0.5초)
    early_window_s = 0.5
    early_samples = int(early_window_s * fs_hz)
    early_samples = min(early_samples, len(force))
    
    if early_samples > 0:
        early_force = force[:early_samples]
        early_errors = early_force - target_force
        
        # 초기 구간 피크 힘
        if target_force < 0:
            early_peak_force = float(np.min(early_force))
        else:
            early_peak_force = float(np.max(early_force))
        
        # 초기 구간 힘 변화율
        if len(early_force) > 1:
            early_force_rate = float((early_force[-1] - early_force[0]) / early_window_s)
        else:
            early_force_rate = 0.0
        
        # 초기 구간 RMSE
        early_rmse = float(np.sqrt(np.mean(early_errors**2)))
        
        # 초기 힘 값
        initial_force = float(force[0])
    else:
        early_peak_force = 0.0
        early_force_rate = 0.0
        early_rmse = 0.0
        initial_force = 0.0
    
    # 기존 trajectory_features에 추가
    trajectory_features = [
        overshoot,
        settling_time,
        rmse,
        band_ratio,
        oscillation_freq,
        oscillation_amp,
        rise_time,
        steady_state_error,
        # 🆕 초기 구간 정보 추가
        early_peak_force,      # 20
        early_force_rate,      # 21
        early_rmse,            # 22
        initial_force,          # 23
    ]
    
    # STATE_DIM: 20 → 24차원
    state = np.array(base_state + trajectory_features, dtype=np.float32)
    return state
```

---

## 🎯 결론

### 현재 State Space 평가
- ❌ **초기 압력 급격한 변화 감지 불가** (매우 심각)
- ❌ 시간적 정보 부족
- ❌ 초기 구간 특화 정보 부족

### 권장 개선 방안
1. ✅ **초기 구간 정보 추가 (4차원)**: 즉시 적용 권장
2. ✅ 힘 변화율 정보 추가 (2차원): 추가 개선

### 예상 효과
- ✅ 초기 압력 급격한 변화를 명확히 감지
- ✅ 초기 오버슈트를 예방하는 PID 게인 학습 가능
- ✅ 학습 성능 개선 기대

---

**초기 구간 정보 추가를 권장합니다!** 🚀

