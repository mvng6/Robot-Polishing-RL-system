# ✅ 구현 체크리스트 (GPT 최종 검토 반영)

## 🔴 최우선: 모순/누락 수정

### 1. Kd 범위 통일 (문서/코드/체크리스트)
- [ ] **constants.py**: `DEFAULT_PID_RANGE["Kd"] = (1e-4, 5e-2)`
- [ ] **config.py**: PID 범위 설정 확인
- [ ] **문서**: 모든 Kd 언급을 [1e-4, 5e-2]로 통일
- [ ] **체크리스트**: [1e-4, 1e-2] → [1e-4, 5e-2] 수정 완료

### 2. 오버슈트 정의 통일
- [ ] **env.py**: 오버슈트 계산식 확인
  ```python
  # 음수 목표 힘 (압축력)
  if target_force < 0:
      extreme_force = float(np.min(force_array))
      overshoot_ratio = max(0.0, (target_force - extreme_force) / abs(target_force))
  else:
      extreme_force = float(np.max(force_array))
      overshoot_ratio = max(0.0, (extreme_force - target_force) / abs(target_force))
  overshoot_pct = float(overshoot_ratio * 100.0)
  ```
- [ ] **loggers/control_performance.py**: 동일한 공식 사용 확인
- [ ] **대시보드**: 수치 일관성 검증 (타임시리즈 vs 대시보드)

---

## 🟡 즉시 구현: 컨트롤러 선조치

### 3. 컨트롤러 플래그 설정 (로봇제어PC 또는 comm.py)

```python
# 컨트롤러 설정 (구현 위치 확인 필요)
controller_config = {
    # Anti-windup
    "anti_windup_mode": "conditional",  # 'backcalc' | 'conditional'
    "anti_windup_gain": 1.0,  # backcalc 사용 시
    
    # Derivative
    "derivative_on_measurement": True,  # 필수
    "derivative_filter_Tf": 0.02,  # 초기값 (0.01~0.03 범위)
    "derivative_filter_N": 30,  # 또는 필터 계수 (20~40)
    
    # Rate limit
    "du_rate_limit": 0.05,  # MPa/s (0.04~0.07 범위)
    
    # Leaky-I
    "leaky_I_lambda": 0.03,  # /s (0.02~0.05 범위)
    
    # Bumpless transfer
    "bumpless_transfer_enabled": True,
}
```

**로그 기록:**
- [ ] 에피소드별로 위 모든 플래그와 값을 CSV/로그 파일에 기록
- [ ] 재현성 확보를 위한 영구 보존

---

## 🟢 보상 함수 및 학습 설정

### 4. 보상 함수 튜닝

```python
# constants.py 수정
SCORE_W_MP = 0.35  # 0.25 → 0.35
SCORE_TAU_MP_PERCENT = 8.0  # 12.0 → 8.0 (8~10 범위)

# env.py에 초기 구간 피크 패널티 추가
def calculate_early_peak_penalty(force_data, target_force, episode_time, dt):
    """0~0.5초 구간 전용 피크 패널티"""
    if episode_time < 0.5:
        peak_force = max(force_data) if target_force > 0 else min(force_data)
        peak_detector = max(0.0, abs(peak_force - target_force) / abs(target_force))
        penalty = min(peak_detector * 0.2, 0.2)  # 상한 0.15~0.2
        return penalty
    return 0.0
```

- [ ] constants.py 수정
- [ ] env.py에 초기 구간 피크 패널티 추가
- [ ] 보상 계산에 통합

### 5. 보상 중심화 A/B 실험 계획

**실험 설정:**
```python
# 실험 A: centered off
reward = reward_score  # 원시 사용

# 실험 B: beta=0.95
beta = 0.95  # 0.99 → 0.95

# 실험 C: tanh gain 1.5
gain = 1.5  # 2.0 → 1.5
reward = tanh(gain * r_centered)
```

**실험 계획:**
- [ ] 각 실험 최소 100ep씩 실행
- [ ] reward_score 절대추세 기록
- [ ] 성능지표 추세(Spearman ρ) 기록
- [ ] 동시에 로그/플롯하여 비교

---

## 🔵 학습 효율 개선

### 6. 하이퍼파라미터 조정

```python
# constants.py 수정
DEFAULT_UPDATES_PER_EPISODE = 35  # 10 → 30~40 (중간값 35)
```

- [ ] constants.py 수정

### 7. 세그먼트 균형 샘플링 또는 PER

**옵션 1: 세그먼트 균형 샘플링**
- [ ] 각 세그먼트(1~5)에서 균등하게 샘플링
- [ ] 버퍼 분포 급변 방지

**옵션 2: 우선순위 경험재생(PER)**
- [ ] PER 구현 (α≈0.6, β-anneal)
- [ ] TD-error 기반 우선순위

### 8. Warm-start

- [ ] LHS/Grid 30~50점으로 버퍼 초기화
- [ ] 본학습 시작 전 버퍼 품질 확보

---

## 🟣 모니터링 및 로깅

### 9. 추가 모니터링 로그

**CSV 로그 추가:**
- [ ] `reward_components.csv`: reward_score, r_centered, r_baseline
- [ ] `controller_state.csv`: integrator_state(I), sat 비율(%), |Δu/Δt|
- [ ] `controller_config.csv`: D 필터 파라미터(Tf/N), rate_limit, leaky_I_lambda
- [ ] `segment_metrics.csv`: 세그먼트별 overshoot, settling, band_ratio

**시각화 추가:**
- [ ] reward_score / r_centered / r_baseline + overshoot/ts/band_ratio 동일 x축 MA(10ep)
- [ ] Spearman ρ 상관관계 그래프
- [ ] 추세선(OLS) 그래프

---

## 🔬 실험 매트릭스 (GPT 권장)

### 소규모 실험 순서

| 실험 | 차이점 | 기대 효과 |
|------|--------|----------|
| **E1** | Baseline (현 설정 + 새 PID 박스 + 보호막) | 기준선 성능 |
| **E2** | E1 + centered off | 보상 우하향 원인 분리 |
| **E3** | E1 + beta=0.95 | 기준선 추종 가속 |
| **E4** | E1 + tanh gain 1.5 | 포화 완화 |
| **E5** | E1 + Kd 상위쿼타 확대 (샘플링 log bias↑) | 초기 피크 감쇠 향상 |
| **E6** | E1 + PER(α=0.6, β-anneal) | 학습안정/수렴속도 개선 |

**실행 순서:**
1. E1 실행 → 기준선 성능 확보
2. E2, E3, E4 병렬 또는 순차 실행 → 보상 함수 최적화
3. E5, E6 실행 → 추가 개선

---

## 📋 세그먼트-지표 정합

### 10. 세그먼트별 지표 계산 통일

**각 2초 세그먼트마다:**
- [ ] **(a) peak(0~0.5 s)**: 세그먼트 시작 0.5초 내 최대/최소 힘
- [ ] **(b) band_ratio**: 세그먼트 전체 밴드 유지율
- [ ] **(c) mini-settling(hold=0.3~0.5 s)**: 세그먼트 내 정착 시간

**보상 계산에도 동일 정의 적용:**
- [ ] 세그먼트별 보상 계산 시 위 지표 사용
- [ ] 에피소드 전체 보상과 일관성 유지

---

## 🎯 최종 체크포인트

### 실행 전 확인

1. ✅ **Kd 범위 통일**: 문서/코드/체크리스트 모두 [1e-4, 5e-2]
2. ✅ **오버슈트 정의 통일**: 보상/대시보드/로그 동일 공식
3. ✅ **컨트롤러 플래그 명시**: 모든 설정값을 로그로 기록
4. ✅ **보상 중심화 실험 계획**: A/B 실험 최소 100ep씩
5. ✅ **세그먼트-지표 정합**: 계산식 통일

### 실행 후 검증

1. ✅ 보상 우하향 원인 분석 (centered off vs baseline)
2. ✅ 오버슈트 수치 일관성 확인 (타임시리즈 vs 대시보드)
3. ✅ 컨트롤러 플래그 재현성 확인
4. ✅ 세그먼트별 지표 일관성 확인

---

## 📝 참고: 현재 코드 상태

### 이미 구현된 부분
- ✅ 액션→PID 매핑: Kd 로그 매핑 구현됨 (`utils/math_utils.py`)
- ✅ 물리 클램프: 후단 적용 확인됨
- ✅ 오버슈트 계산: 음수 목표 힘 처리 구현됨 (`env.py`)

### 확인 필요
- ❓ 로봇제어PC에서 D 필터링 구현 여부
- ❓ Anti-windup 구현 방식 (backcalc vs conditional)
- ❓ Rate limit 구현 여부
- ❓ Leaky-I 구현 여부

---

**이 체크리스트를 따라 진행하면, 초기 피크·링잉 억제와 보상곡선 해석 혼선을 동시에 해결할 수 있습니다.**

