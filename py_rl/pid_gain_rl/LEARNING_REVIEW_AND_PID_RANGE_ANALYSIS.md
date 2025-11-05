# 🔍 강화학습 결과 리뷰 및 PID 범위 변경 분석

> **⚠️ GPT 최종 검토 반영**: 모순/누락/리스크 수정 완료
> - Kd 범위 통일: [1e-4, 5e-2] (문서/코드/체크리스트)
> - 오버슈트 정의 통일: 보상/대시보드/로그 동일 공식
> - 컨트롤러 플래그 명시: 구현 디테일 명확화
> - 보상 중심화 A/B 실험 계획: 최소 100ep씩
> - 세그먼트-지표 정합: 계산식 통일

## 📊 현재 학습 결과 분석

### 1. 그래프에서 관찰된 현상

**첫 번째/두 번째 그래프 (시계열 힘 추적):**
- ✅ 초기 스텝 직후 큰 오버슈트 (60N대, 목표 40N 대비 57-60%)
- ✅ 강한 진동 후 감쇠
- ✅ 이후 밴드(±1.5N) 내 안정 유지

**세 번째 그래프 (제어 성능 대시보드):**
- ⚠️ OVERSHOOT: 높은 값 (90~105% 범위, 완만한 하향 경향 가능성)
  - **주의**: 이동평균 없이 단정하지 말 것, 10-ep 이동평균과 중앙값/사분위 확인 필요
- ⚠️ SETTLING_TIME: 높은 변동성 (1~5초)
- ⚠️ BAND_RATIO: 특정 시점(35, 78) 급락
- ⚠️ RMSE: 후반부 증가 추세로 보이지만 노이즈 범위 내 변동일 가능성
  - **검증 필요**: 추세선(OLS)이나 Spearman ρ로 상관 수치화
- ⚠️ IAE, TOTAL_VARIATION: 특정 시점 큰 스파이크

**네 번째 그래프 (에피소드 보상):**
- ❌ 우하향 추세 (에피소드 25 이후)
- ❌ 큰 변동성 (0.2 → -0.35)
- ⚠️ 평균 보상: 0.14 (낮음)

---

## 🤔 학습이 제대로 되고 있는가?

### GPT의 분석: "보상 우하향 = 학습 실패"는 성급함

**✅ GPT의 핵심 지적 (타당함):**

현재 보상 함수 구조:
```python
reward_score = (0~1 범위)  # 원시 스코어
r_centered = reward_score - rew_baseline  # 기준선 중심화
reward = tanh(2.0 * r_centered)  # [-1, 1] 소프트클립
```

**문제점:**
- `rew_baseline`이 EWMA(beta=0.99)로 천천히 상승
- 실제 성능이 개선되어도 기준선이 따라가면 `r_centered`가 낮아짐
- 보상 곡선이 우하향처럼 보일 수 있음

**✅ 검증 필요:**
```python
# 에피소드별로 다음을 함께 플롯:
1. reward_score (원시 스코어)
2. r_centered (중심화된 값)
3. rew_baseline (기준선)
4. overshoot, settling_time, band_ratio (실제 성능 지표)
```

**결론**: 보상 우하향이 실제 성능 악화인지, 기준선 중심화 부작용인지 **먼저 분리해야 함**

---

### 🔍 현재 코드 확인 결과

**보상 함수 구조 (env.py):**
```python
# 1. 원시 스코어 계산 (0~1)
reward_score = (
    0.30 * S_ts +       # 정착시간
    0.25 * S_mp +       # 오버슈트 (현재)
    0.20 * S_ess +      # 정상상태 오차
    0.15 * S_band +     # 밴드 유지
    0.05 * S_u -        # 제어 노력
    -0.15 * P_fail +    # 추종 실패
    +0.10 * progress +  # PBRS
    +0.10 * I_improve   # 기준선 대비 개선
)

# 2. 기준선 중심화
beta = 0.99  # 매우 느린 추적
self._rew_baseline = beta * self._rew_baseline + (1.0 - beta) * reward_score
r_centered = reward_score - self._rew_baseline

# 3. tanh 소프트클립
reward = tanh(2.0 * r_centered)
```

**현재 로깅:**
- ✅ `reward_score`, `r_centered`, `r_baseline`이 metrics에 포함됨
- ❌ **하지만 시각화가 부족함**: Episode별로 함께 플롯되지 않음

**권장: 추가 로깅/시각화**
```python
# reward_breakdown.py에 추가
- reward_score, r_centered, r_baseline의 시계열 그래프
- reward_score vs overshoot/settling_time 상관관계
- Moving average (10-ep) 적용
```

---

## 🎯 제안된 PID 범위 분석

### 제안: Kp[5,40], Ki[10,50], Kd[1e-9,1e-8]

#### ✅ 타당한 부분

1. **Kp 하한 확장 (35→5)**: ✅ **올바름**
   - 오버슈트 감소를 위해 비례 게인을 낮추는 것은 맞음
   - 현재 Kp=35~45는 너무 높아 초기 피크를 유발

2. **Ki 하한 확장 (45→10)**: ✅ **올바름**
   - 적분 게인을 낮추면 초기 진동 감소 가능
   - 단, Ki=0도 포함해야 함 (조건부 적분 off)

#### ❌ 문제가 있는 부분

3. **Kd 범위 축소 (1e-6~1e-3 → 1e-9~1e-8)**: ❌ **심각한 오류**

**GPT의 지적 (완전히 타당):**
- Kd는 오버슈트와 진동을 잡는 **핵심 댐핑 요소**
- 1e-9~1e-8은 사실상 **0에 가까워 댐핑 효과 없음**
- 현재도 1e-6~1e-3은 매우 작았는데, 더 줄이면 오히려 악화

**문제의 핵심:**
```
초기 피크 + 진동 → 댐핑 필요 → D(미분) 게인
현재 Kd 범위도 작음 → 더 줄이면 댐핑 효과 사라짐
```

---

## 💡 올바른 PID 범위 제안

### GPT의 권장안 (비판적 검토)

**제안:**
- Kp: [3, 40] ✅ (하한 더 내림, 상한 유지)
- Ki: [0, 40] ✅ (하한 0까지, 상한 낮춤)
- Kd: [1e-4, 5e-2] ✅ (대폭 증가, 로그 균일)

**비판적 검토:**

#### ✅ Kp[3, 40]: 타당
- 하한 5→3: 더 보수적 접근 가능
- 상한 40 유지: 고속 응답 필요 시 활용

#### ✅ Ki[0, 40]: 타당하지만 주의
- Ki=0 포함: 조건부 적분 off 가능 (좋음)
- 상한 55→40: 과도한 적분 누적 방지
- ⚠️ **주의**: Ki=0 근처에서는 안정성 검증 필요

#### ⚠️ Kd[1e-4, 5e-2]: 방향은 맞지만 GPT 지적이 타당함

**GPT의 지적 (타당함):**
- ❌ **내 제안이 너무 보수적**: Kd[1e-4, 1e-2]는 부족할 수 있음
- ✅ **초기 피크+링잉 문제**: 상한을 5e-2까지 열어두는 것이 안전
- ✅ **시스템 스케일 고려**: 1e-2가 부족할 수 있음

**수정된 권장안:**
```
Kd: [1e-4, 5e-2] 로그-스페이스 (필터와 rate-limit 전제)
```

**우려사항:**
- ⚠️ **시스템 의존성**: Kd 단위는 시스템에 따라 다름
- ⚠️ **노이즈 민감도**: D는 측정값 미분이므로 노이즈 증폭 가능
- ⚠️ **필터링 필수**: D-on-measurement + 1차 LPF 필수

**구체적 필터링 가이드:**
```
1 kHz 샘플 기준:
- 미분 저역통과: Tf = 0.01~0.03 s (컷오프 ~5-16 Hz)
- 또는 필터 계수: N = 20~40
- u̇ 제한: 0.04~0.07 MPa/s부터 탐색
```

---

## 🛠️ 추가 개선 사항

### 1. 컨트롤러 측 선조치 (GPT 권장 - 타당함)

#### ✅ Anti-windup (필수)
```python
# 현재 코드 확인 필요
# PI_OUTPUT_MAX = 0.4 근처에서 sat 시
# integral 누적이 커져 스텝 해제 시 피크 발생
```

**구현 방법:**
1. **Back-calculation**: aw_gain 사용
2. **Conditional integration**: 포화/대오차 구간에서 I 정지
   ```python
   # |e| > 2·BAND_TOL 또는 sat일 때 I 정지
   if abs(error) > 2 * BAND_TOLERANCE_N or abs(pi_output) >= PI_OUTPUT_MAX * 0.95:
       # I term 업데이트 정지
   ```

#### ✅ Leaky-I (누설 적분)
```python
# 정착 후 I_leak = -λ·I (λ≈0.02~0.05/s)
# 과도한 적분 누적 방지
if abs(error) < BAND_TOLERANCE_N:
    I_term = I_term * (1.0 - lambda_leak * dt)  # lambda_leak = 0.02~0.05
```

#### ✅ Bumpless transfer
```python
# 접촉 직후 또는 목표 변경 시 integrator 초기값 프리셋
# 최근 steady-state 추정값으로 초기화
if target_changed or contact_just_started:
    I_term = estimate_steady_state_I()  # 최근 steady-state 추정
```

#### ✅ 입력 rate limit
```python
# |Δu/Δt| 상한: 0.04~0.07 MPa/s
# 초기 가압 속도 제한 → 피크 억제
max_rate = 0.05  # MPa/s
if abs(du_dt) > max_rate:
    u = u_prev + sign(du_dt) * max_rate * dt
```

#### ✅ D 필터링 (필수, 구체적 가이드 추가)
```python
# D-on-measurement + 1차 LPF (필수)
# 1 kHz 샘플 기준:
#   - Tf = 0.01~0.03 s (컷오프 ~5-16 Hz)
#   - 또는 필터 계수 N = 20~40
# u̇ 제한: 0.04~0.07 MPa/s부터 탐색
```

**현재 코드 확인:**
- ❓ 로봇제어PC에서 D 필터링 구현 여부 확인 필요
- ❓ D-on-measurement vs D-on-error 확인 필요

---

### 2. 보상 함수 튜닝 (GPT 권장 - 타당함)

#### ✅ Overshoot 가중치 강화
```python
# 현재
SCORE_W_MP = 0.25
SCORE_TAU_MP_PERCENT = 12.0

# 제안
SCORE_W_MP = 0.35  # 0.25 → 0.35
SCORE_TAU_MP_PERCENT = 8.0  # 12.0 → 8.0 (더 민감하게)
```

**타당성:** ✅ 오버슈트가 주요 문제이므로 가중치 강화는 맞음

#### ✅ 초기 구간 피크 패널티 추가 (GPT 지적: 상한 필요)
```python
# 0~0.5초 구간에만 작동하는 피크 패널티
peak_detector = max(0.0, (peak_force - target_force) / abs(target_force))
if episode_time < 0.5:
    penalty = min(peak_detector * 0.2, 0.15~0.2)  # 상한 0.15~0.2
```

**GPT의 지적 (타당함):**
- ⚠️ **안전·실패 패널티 상대량**: SCORE_W_FAIL=0.15와 SCORE_W_MP=0.35 관계
- ⚠️ 큰 피크가 한번만 나도 전체 스코어를 과도하게 깎는지 검토 필요
- ✅ 초기구간 전용 패널티는 0.15~0.2 상한
- ✅ 대신 지속적 링잉을 TOTAL_VARIATION/u_rms 가중으로 천천히 깎기

**타당성:** ✅ 내 제안보다 GPT 제안이 더 안전하고 균형적

#### ⚠️ 기준선 중심화 실험 (GPT 지적: tanh gain 조정 추가)
```python
# 옵션 1: r_centered 비활성화 (원시 reward_score 사용)
# 옵션 2: beta=0.99 → 0.95 (기준선 반응 빠르게)
# 옵션 3: tanh gain 조정 (현재 2.0 → 1.5~2.0 범위 스윕)
```

**GPT의 지적 (타당함):**
- ⚠️ **보상 스케일/포화**: tanh(gain=2.0)는 중심화 이후 작은 차이를 과도하게 압축
- ✅ **A/B 실험**: gain 1.5~2.0 범위 스윕, 혹은 초기 50ep는 tanh 제거(선형)

**타당성:** ✅ GPT 제안이 더 구체적이고 실험적 접근

---

### 3. 학습 하이퍼파라미터 튜닝

#### ✅ Updates per episode 증가 (GPT 지적: 더 구체적 필요)
```python
# 현재
DEFAULT_UPDATES_PER_EPISODE = 10

# GPT 제안 (타당함)
DEFAULT_UPDATES_PER_EPISODE = 30~40  # 세그먼트 5개 대비 증가
```

**GPT의 추가 지적 (타당함):**
- ⚠️ 세그먼트 5개(2초) 구조에서 버퍼 분포가 급변
- ✅ **우선순위 경험재생(PER)** 또는 **세그먼트 균형 샘플링** 필요
- ✅ **Warm-start**: 짧은 LHS/Grid 30~50점으로 버퍼 품질 확보

**타당성:** ✅ 내 제안보다 GPT 제안이 더 구체적이고 타당

#### ⚠️ 탐험 (α, log_std) 조정
```python
# 현재
ACTOR_INITIAL_ALPHA = 0.02  # 매우 보수적
ACTOR_LOG_STD_MAX = -1.0   # 탐험 축소

# 제안
# 초반 50~100ep 동안 α 더 크게 또는 자동온도
```

**타당성:** ⚠️ **조건부**
- 새로운 PID 범위(Kp 하한↓, Kd↑) 탐색을 위해 초반 탐험 증가는 타당
- 하지만 과도한 탐험은 불안정성 초래 가능

---

## 📊 추가 필요한 로그/자료

### GPT가 제안한 항목 (모두 타당)

1. **pi_output, integrator_state(I term), sat 여부/시간비율**
   - Anti-windup 검증 필요

2. **|Δu/Δt| 시계열**
   - Rate limit 검증 필요

3. **D 필터 파라미터(Tf 또는 N)**
   - D 필터링 효과 확인

4. **세그먼트별 overshoot/ts/band_ratio**
   - 현재는 에피소드 전체만 계산
   - 세그먼트별 분석 필요

5. **reward_score, r_centered, r_baseline의 raw 값 CSV**
   - 보상 우하향 원인 분석

6. **노이즈 스펙(힘 센서 RMS, cutoff)**
   - D 필터링 설계 필요

---

## 🎯 최종 권장사항

### 즉시 실행 (우선순위 높음)

#### 1. 보상 우하향 원인 분석
```python
# reward_breakdown.py에 추가
- reward_score, r_centered, r_baseline 시계열 그래프
- overshoot, settling_time, band_ratio와 상관관계
- Moving average (10-ep) 적용
```

#### 2. PID 범위 재설정 (GPT 지적 반영)
```python
# ✅ GPT 검토 후 수정된 제안
Kp: [3, 40]  # 하한 확장
Ki: [0, 40]  # 하한 0까지, 상한 낮춤 (조건부 적분 + leaky-I)
Kd: [1e-4, 5e-2]  # 로그-스페이스 (필터·rate-limit 전제)
```

**GPT 지적 반영:**
- ❌ **내 제안이 너무 보수적**: Kd[1e-4, 1e-2]는 부족할 수 있음
- ✅ **초기 피크+링잉 문제**: 상한을 5e-2까지 열어두는 것이 안전
- ✅ **액션 매핑**: Kd만 로그 매핑, 물리 클램프는 후단에 적용 (이미 구현됨)

#### 3. 컨트롤러 선조치
- ✅ Anti-windup 구현
- ✅ D-on-measurement + LPF
- ✅ 입력 rate limit

### 단계적 실행 (우선순위 중간)

#### 4. 보상 함수 튜닝
```python
SCORE_W_MP = 0.35  # 0.25 → 0.35
SCORE_TAU_MP_PERCENT = 8.0  # 12.0 → 8.0
# 초기 구간 피크 패널티 추가
```

#### 5. 학습 하이퍼파라미터 (GPT 지적 반영)
```python
DEFAULT_UPDATES_PER_EPISODE = 30~40  # 10 → 30~40 (GPT 제안)
# 우선순위 경험재생(PER) 또는 세그먼트 균형 샘플링
# Warm-start: LHS/Grid 30~50점으로 버퍼 품질 확보
# 초반 50ep 동안 탐험 증가 (선택적)
```

### 검증 후 실행 (우선순위 낮음)

#### 6. 기준선 중심화 조정 (GPT 지적 반영)
- 보상 우하향 원인 분석 후 결정
- 옵션 1: beta=0.99 → 0.95
- 옵션 2: tanh gain 2.0 → 1.5~2.0 범위 스윕
- 옵션 3: 초기 50ep는 tanh 제거(선형) A/B 실험

---

## 📝 한 줄 결론 (GPT 검토 반영)

1. ✅ **보상 우하향 = 학습 실패 단정은 성급함**: 기준선 중심화 부작용 가능성
2. ✅ **Kp/Ki 하한 확장은 타당**: 오버슈트 억제 목적
3. ✅ **Kd 범위 확장 필요**: [1e-4, 5e-2] 로그-스페이스 (필터·rate-limit 전제)
   - ❌ **내 제안 [1e-4, 1e-2]는 너무 보수적**: GPT 지적이 타당
4. ✅ **컨트롤러 선조치 필수**: Anti-windup, Leaky-I, D 필터링(Tf=0.01~0.03s), rate limit
5. ✅ **보상 함수 튜닝**: Overshoot 가중치 강화, 초기 구간 피크 패널티(상한 0.15~0.2)
6. ✅ **학습 효율 개선**: UPDATES_PER_EPISODE=30~40, PER 또는 세그먼트 균형 샘플링, Warm-start
7. ⚠️ **그래프 해석 정확도**: 이동평균(10-ep)과 추세선(OLS)로 정량화 필요

---

## 🔬 검증 체크리스트

### 학습 결과 분석 (GPT 지적: 정량화 필요)
- [ ] **보상 구성 요소 시각화**
  - [ ] reward_score, r_centered, r_baseline 동일 x축 플롯
  - [ ] overshoot, settling_time, band_ratio와 함께 플롯
  - [ ] Moving average (10-ep) 적용
- [ ] **상관관계 정량 분석**
  - [ ] Spearman ρ로 보상과 성능지표 상관 수치화
  - [ ] 추세선(OLS)로 RMSE 후반부 증가 검증
- [ ] **오버슈트 수치 일관성 검증**
  - [ ] 타임시리즈 피크(~65N, 목표 40N)와 대시보드(90~105%) 비교
  - [ ] 오버슈트 정의/단위 재검증: `overshoot_pct = 100 * max(0, (max(force) - target)/|target|)`
  - [ ] 음수 목표 힘 케이스 처리 확인
- [ ] **추가 모니터링 로그**
  - [ ] integrator_state(I), sat 비율(%), |Δu/Δt| 시계열
  - [ ] D 필터 파라미터(Tf/N) 기록
  - [ ] 세그먼트별 지표 CSV (overshoot, settling, band_ratio)

### PID 범위 변경 (GPT 지적: 통일 필요)
- [ ] Kp[3, 40] 설정
- [ ] Ki[0, 40] 설정
- [ ] **Kd[1e-4, 5e-2] 설정** (로그-스페이스, 필터·rate-limit 전제)
- [ ] D 필터링 구현 확인
- [ ] **문서/코드/체크리스트 모두 동일한 범위 사용 확인**

### 컨트롤러 검증 (GPT 지적: 구현 디테일 명시 필요)
- [ ] **Anti-windup 구현 확인**
  - [ ] `anti_windup_mode = {'backcalc' | 'conditional'}` 설정
  - [ ] Conditional I: `|e| > 2·BAND_TOL` 또는 `sat` 시 적분 정지
- [ ] **D 필터링 구현 확인**
  - [ ] `derivative_on_measurement = True` 설정
  - [ ] `Tf = 0.01~0.03 s` (초기값 0.02) 또는 `N = 20~40`
  - [ ] 필터 파라미터를 로그로 기록
- [ ] **입력 rate limit 확인**
  - [ ] `du_rate_limit = 0.04~0.07 MPa/s` (초기값 0.05)
  - [ ] rate limit 값을 로그로 기록
- [ ] **Bumpless transfer 확인**
  - [ ] 접촉/목표변경 시 I 프리셋 구현
- [ ] **Leaky-I 확인**
  - [ ] `leaky_I_lambda = 0.02~0.05/s` (초기값 0.03)
  - [ ] 정착 후 I_leak = -λ·I 적용
- [ ] **모든 컨트롤러 플래그와 값을 로그로 영구 보존**

### 보상 함수 튜닝 (GPT 지적: 실험 계획 명확화 필요)
- [ ] **오버슈트 가중치 강화**
  - [ ] SCORE_W_MP = 0.35 (0.25 → 0.35)
  - [ ] SCORE_TAU_MP_PERCENT = 8~10 (12.0 → 8~10)
- [ ] **초기 구간 피크 패널티 추가**
  - [ ] 0~0.5초 구간 전용 패널티
  - [ ] 상한 0.15~0.2로 제한
- [ ] **보상 중심화 A/B 실험 (최소 100ep씩)**
  - [ ] (A) centered off (원시 reward_score 사용)
  - [ ] (B) beta=0.95 (기준선 추종 가속)
  - [ ] (C) tanh gain 1.5 (포화 완화)
  - [ ] 각 실험에서 reward_score 절대추세와 성능지표 추세(Spearman ρ) 기록
- [ ] **오버슈트 정의 통일 확인**
  - [ ] 보상/대시보드/로그에서 동일한 공식 사용
  - [ ] 음수 목표 힘 케이스 처리 확인

### 하이퍼파라미터 (GPT 지적: 수정 필요)
- [ ] **UPDATES_PER_EPISODE = 30~40** (10 → 30~40)
- [ ] **세그먼트 균형 샘플링 또는 PER(α≈0.6, β-anneal)**
- [ ] **Warm-start: LHS/Grid 30~50점으로 버퍼 품질 확보**
- [ ] 초반 탐험 증가 (선택적)

---

## 📚 GPT 검토 반영 요약

### ✅ GPT가 맞게 지적한 부분 (모두 타당)

1. **그래프 해석 정확도**: OVERSHOOT "100% 근처"는 과장 가능, 이동평균 필요
2. **Kd 상한이 너무 보수적**: [1e-4, 1e-2] → [1e-4, 5e-2] 필요
3. **D 설계 상세 부족**: 구체 수치 가이드 추가 (Tf=0.01~0.03s, N=20~40)
4. **Integral 운용 디테일**: Leaky-I, 조건부 적분 상세 필요
5. **행동공간 매핑**: 로그 균일은 이미 구현됨, 물리 클램프 순서 명확화 필요
6. **학습 데이터 유효성**: UPDATES_PER_EPISODE=30~40, PER, Warm-start 필요
7. **보상 스케일/포화**: tanh gain 조정 (1.5~2.0 범위 스윕) 필요
8. **밴드/정착 정의**: 세그먼트별 일치성 확인 필요
9. **안전·실패 패널티**: 초기 구간 피크 패널티 상한 0.15~0.2 필요

### 🔄 수정된 최종 권장사항

**A. PID 탐색 박스 v2:**
- Kp: [3, 40]
- Ki: [0, 40] (조건부 적분 + leaky-I)
- Kd: 로그-스페이스 [1e-4, 5e-2] (필터·rate-limit 전제)

**B. 컨트롤러 선조치 (필수):**
- Anti-windup (back-calc 또는 conditional-I)
- Derivative on measurement + LPF: Tf=0.01~0.03s
- 입력 rate-limit: |Δu/Δt| ≤ 0.04~0.07 MPa/s
- Bumpless transfer: 접촉/목표변경 시 I 프리셋
- Leaky-I: 정착 후 I_leak = -λ·I (λ≈0.02~0.05/s)

**C. 학습/보상:**
- UPDATES_PER_EPISODE = 30~40, PER 또는 세그먼트 균형샘플
- Overshoot 가중 강화: W_MP = 0.35, TAU_MP = 8~10
- 초기 0~0.5s 피크 패널티(상한 0.15~0.2)
- 보상 중심화 A/B: (a) centered-off, (b) beta=0.95, (c) tanh-gain 1.5
- 액션→PID 매핑: Kd만 로그, 물리 클램프는 후단에 적용

**D. 분석/로그 (추가 수집):**
- reward_score / r_centered / r_baseline + 지표 동일 x축 MA(10ep)
- integrator_state(I), sat 비율, |Δu/Δt|, Kd 필터 파라미터
- 세그먼트별 overshoot/settling/band
- 센서 노이즈 RMS & 필터 컷오프

### 🎯 결론

**GPT의 검토가 매우 타당하며, 내 분석의 부족한 부분을 잘 보완함:**
- Kd 상한을 5e-2까지 확장하는 것이 올바름
- D 필터링, Integral 운용, 학습 효율에 대한 구체적 가이드 필요
- 그래프 해석의 정확도 향상 필요 (이동평균, 추세선)

