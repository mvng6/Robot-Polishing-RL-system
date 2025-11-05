# PID Gain RL - 모듈화된 강화학습 시스템

## 📋 개요

공압 폴리싱 로봇의 PID 게인을 **Soft Actor-Critic (SAC) 강화학습**으로 최적화하는 시스템입니다.
**3876줄 → 16개 모듈**로 리팩토링하여 유지보수성과 확장성을 향상했습니다.

### 핵심 특징
- 🎯 **Fine-tuning 최적화**: 국소 탐색 범위로 빠른 수렴
- 📊 **세그먼트 분할 학습**: 1 에피소드 → 5개 transition으로 학습 효율 향상
- 🧠 **20차원 상태 공간**: 기존 12차원 + 궤적 요약 8차원
- 🔒 **안전 중심 설계**: 안전 위반 자동 감지 및 재시작
- 📈 **제어공학 통합**: 10개 핵심 지표 계산 및 논문용 그래프 생성

### 최종 목표
- **목표 접촉력**: -30N까지 위치 제어 가능
- **동적 목표 환경**: -35N ~ -50N 범위에서 무작위로 변경 (한 에피소드에 3번 변경)
- **즉각 대응**: 목표 접촉력 변경 시 즉시 최적 PID 게인 출력

---

## 🏗️ 시스템 아키텍처

### 전체 시스템 구조

```
┌───────────────────────────────────────────────────────────┐
│                    강화학습 PC (Python)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │__main__/main │→ │     env      │→ │    agent     │     │
│  │  (엔트리)      │  │  (학습 환경)  │  │  (SAC RL)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │           │
│         │                  ↓                  │           │
│         │          ┌──────────────┐           │           │
│         │          │   comm.py    │           │           │
│         │          │ (TCP/IP 통신) │           │           │
│         │          └──────────────┘           │           │
│         │                  │                  │           │
│         └──────────────────┼──────────────────┘           │
│                            │                              │
│                     ┌──────▼──────┐                       │
│                     │   monitor   │                       │
│                     │  (GUI 실시간)│                       │
│                     └─────────────┘                       │
└────────────────────────────┬──────────────────────────────┘
                             │
                      TCP/IP (포트 8888)
                     1kHz 데이터 교환
                             │
┌────────────────────────────▼──────────────────────────────┐
│                 로봇제어 PC (C++)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   PID 제어    │→ │  공압 액추에이터│→ │  힘 센서       │     │
│  │  (RL 최적화)   │  │  (폴리싱 툴)  │  │  (1kHz)      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────────────────────────────────────────┘
```

### 컴포넌트 간 상호작용

1. **강화학습 PC** → **로봇제어 PC**: PID gain 전송 (에피소드당 1회)
2. **로봇제어 PC** → **강화학습 PC**: 상태 데이터 수신 (1kHz 지속)
3. **환경(env)** ↔ **에이전트(agent)**: 상태/액션/보상 교환
4. **모니터(monitor)**: 실시간 Force/보상 그래프 표시

### `__main__.py` vs `main.py` 차이점

| 항목 | `__main__.py` | `main.py` |
|------|--------------|-----------|
| **실행 방법** | `python3 -m pid_gain_rl` | `python3 main.py [옵션]` |
| **설정 방식** | 코드 내 하드코딩 | argparse 명령줄 인자 |
| **설정 변경** | 코드 수정 필요   | 실행 시 옵션으로 변경 |
| **용도** | 간단한 실행, 기본 설정 사용 | 다양한 실험, 유연한 설정 |
| **예시** | `python3 -m pid_gain_rl` | `python3 main.py --episodes 500 --target-force -40.0` |

**결론**: 두 파일 모두 동일한 학습 로직을 사용하지만, `__main__.py`는 간단한 실행용, `main.py`는 명령줄로 유연하게 설정할 수 있는 인터페이스입니다.

---

## 🚀 실행 방법

### 방법 1: 모듈 방식 (권장)

```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl
python3 -m pid_gain_rl
```

**특징:**
- 기본 설정 사용
- 간단한 실행
- `__main__.py`에서 설정 변경 가능

### 방법 2: 명령줄 인터페이스 (유연한 설정)

```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl
python3 main.py --episodes 500 --target-force -40.0 --batch-size 64
```

**지원 옵션:**
```bash
python3 main.py \
  --episodes 500              # 학습할 에피소드 수
  --batch-size 64             # 배치 크기
  --lr 1e-4                   # 기본 학습률
  --lr-actor 1e-4             # Actor 학습률
  --lr-critic 2e-4            # Critic 학습률
  --target-force -40.0        # 목표 힘 (N)
  --episode-seconds 10.0      # 에피소드 길이 (초)
  --recv-freq 1000            # 수신 주파수 (Hz)
  --load-model /path/to/model.pth  # 모델 로드
  --log-dir /path/to/logs     # 로그 디렉토리
  --model-dir /path/to/models # 모델 저장 디렉토리
```

---

## 🔄 전체 실행 흐름

### 단계별 실행 과정

#### Phase 1: 초기화 (프로그램 시작 시)

```
1. 설정 생성 (create_config)
   ├─ 수신 주파수: 1000 Hz
   ├─ 에피소드 길이: 10.0초
   └─ 기타 상수 로드 (constants.py)

2. 환경 생성 (PIDGainEnvironment)
   ├─ 에이전트 생성 (PIDGainSACAgent)
   ├─ 통신 객체 생성 (PIDGainCommunicator)
   └─ 로거 초기화 (ControlPerformanceLogger, RewardBreakdownLogger)

3. TCP/IP 연결 대기
   ├─ 로봇제어PC 연결 대기 (포트 8888)
   └─ 연결 성공 확인

4. RL 활성화 대기
   ├─ sander_active 플래그 감지
   └─ 최대 5분 타임아웃

5. 실시간 모니터 시작
   └─ 별도 프로세스에서 GUI 실행 (10 Hz 갱신)
```

#### Phase 2: 에피소드 실행 루프 (각 에피소드마다 반복)

```
에피소드 N 시작
│
├─ 1. 초기 상태 생성
│   ├─ 이전 PID gain 정보 활용
│   ├─ 에피소드 히스토리 활용 (최근 5개)
│   └─ 20차원 상태 벡터 생성
│
├─ 2. PID gain 선택
│   ├─ 첫 에피소드: 기준 PID (40, 50, 0)
│   └─ 이후 에피소드: SAC 에이전트 선택 (또는 ε-greedy)
│
├─ 3. PID gain 전송
│   ├─ 로봇제어PC로 전송
│   └─ episode_done=False 신호 (에피소드 시작)
│
├─ 4. sander_active 상승 에지 대기 (2번째 에피소드부터)
│   └─ False → True 전환 감지
│
├─ 5. 데이터 수집 (10초간, 1kHz)
│   ├─ 힘 데이터 (force)
│   ├─ PI 출력 데이터 (pi_output)
│   ├─ 실시간 모니터 전송 (10 Hz)
│   └─ 안전 위반 감지 (힘 > ±100N)
│      └─ 즉시 종료 및 랜덤 PID로 재시작
│
├─ 6. 세그먼트 분할 및 보상 계산
│   ├─ 10초 데이터 → 5개 세그먼트 (2초씩)
│   ├─ 각 세그먼트별 보상 계산
│   │   ├─ 정착시간 (Settling Time)
│   │   ├─ 오버슈트 (Overshoot %)
│   │   ├─ 정상상태 오차 (ESS)
│   │   ├─ 밴드 유지율
│   │   ├─ 제어 입력 RMS
│   │   └─ PBRS (Potential-Based Reward Shaping)
│   └─ 최종 보상: [-1.0, 1.0] 범위 (tanh 소프트클립)
│
├─ 7. 경험 저장 (Replay Buffer)
│   ├─ 각 세그먼트별 transition 저장
│   └─ 1 에피소드 = 5개 transition
│
├─ 8. 강화학습 업데이트
│   ├─ 최소 버퍼 크기 확인 (32개)
│   ├─ 배치 샘플링
│   ├─ Critic 업데이트 (Dual Q-Network)
│   ├─ Actor 업데이트 (정책 개선)
│   ├─ 표준편차 Annealing 업데이트
│   └─ 타겟 네트워크 업데이트 (Soft Update, τ=0.01)
│
├─ 9. 다음 에피소드 PID gain 계산
│   ├─ ε-greedy: 10 에피소드마다 5% 확률 랜덤
│   └─ 그 외: SAC 에이전트 선택
│
├─ 10. 에피소드 완료 신호 전송
│    ├─ 다음 PID gain과 함께 전송
│    └─ episode_done=True 신호
│
└─ 11. 로봇 리셋 대기 (2초, 모니터링 지속)
    └─ 다음 에피소드 준비
```

#### Phase 3: 학습 완료

```
1. 최종 데이터 저장
   ├─ 제어 성능 지표 (10개 핵심 지표)
   ├─ 보상 분석 그래프
   └─ 에피소드 요약 CSV

2. 학습 종료 신호 전송
   └─ learning_done=True

3. 모니터 종료
   └─ GUI 창 닫기

4. 통신 종료
   └─ TCP/IP 연결 해제
```

---

## 🧠 학습 알고리즘 상세

### Soft Actor-Critic (SAC) 알고리즘

#### 1. 상태 공간 (State Space): 20차원

```
기존 12차원:
├─ 로봇PC 6차원: current_force, target_force, force_error, 
│                force_error_dot, force_error_int, pi_output
└─ 강화학습PC 6차원: Kp, Ki, Kd, prev_reward, target_force, 
                     episode_seconds, 통계 6개 (평균, 표준편차, 최소, 최대, 평균 오차, 오차 표준편차)

추가 궤적 요약 8차원:
├─ overshoot (%)           # 오버슈트 비율
├─ settling_time (초)      # 정착시간
├─ rmse (N)               # RMSE
├─ band_ratio (0~1)       # 밴드 유지율
├─ oscillation_freq (Hz)  # 진동 주파수 (FFT)
├─ oscillation_amp (N)    # 진동 진폭
├─ rise_time (초)         # 상승시간 (10%→90%)
└─ steady_state_error (N) # 정상상태 오차
```

#### 2. 액션 공간 (Action Space): 3차원

```
액션: [a_Kp, a_Ki, a_Kd] ∈ [-1, 1]³

PID gain 매핑:
├─ Kp: 선형 매핑 → [3.0, 40.0] (소수점 2자리)
├─ Ki: 선형 매핑 → [0.0, 40.0] (소수점 2자리)
└─ Kd: 로그 매핑 → [1e-4, 5e-2] (소수점 6자리)
```

#### 3. 신경망 구조

**Actor 네트워크:**
```
입력: 20차원 상태
├─ FC1: 20 → 128 (ReLU)
├─ FC2: 128 → 128 (ReLU)
├─ Mean Head: 128 → 3 (tanh 클램핑: [-10, 10])
└─ Log Std Head: 128 → 3 (클램핑: [-2.5, -0.3])
```

**Critic 네트워크 (Dual Q-Network):**
```
입력: 20차원 상태 + 3차원 액션 = 23차원
├─ Q1:
│   ├─ FC1: 23 → 128 (ReLU)
│   ├─ FC2: 128 → 128 (ReLU)
│   └─ FC3: 128 → 1
└─ Q2:
    ├─ FC1: 23 → 128 (ReLU)
    ├─ FC2: 128 → 128 (ReLU)
    └─ FC3: 128 → 1
```

#### 4. 학습 파라미터

```
학습률:
├─ Actor: 1e-4
└─ Critic: 2e-4

할인율: γ = 0.99

Soft Update: τ = 0.01

Entropy Coefficient:
├─ 초기값: α = 0.1
└─ 자동 조정: True (target_entropy 동적 조정)

표준편차 Annealing:
├─ 초기: std_scale = 1.0
├─ 최종: std_scale = 0.5
└─ 범위: 0~150 에피소드 (선형 감소)

리플레이 버퍼:
└─ 크기: 10,000개 (세그먼트별 저장, 5배 증가)
```

---

## 🎯 보상 함수 구성 요약

### 최종 출력
**reward ∈ [-1.0, 1.0]** (tanh 소프트클립)

### 계산 단계
1. **핵심 지표 계산**: RMSE, Overshoot (%), Band Ratio (±1.5N, 전체 구간), Settling Time (±1.0N 범위 내 0.5초 연속 유지, 0.2초 이후 검사), ESS, Input RMS
2. **지수 스코어 변환 (0~1)**: S_ts=exp(-settling_time/5.0초), S_mp=exp(-overshoot/8.0%), S_ess=exp(-|ESS|/1.0N), S_band=band_ratio, S_u=exp(-u_rms/0.5)
3. **PBRS**: 순간 개선 신호 (γ=0.99, 0.2초 이후 데이터)
4. **추종 실패 패널티**: RMSE>5.0N 또는 Band Ratio<0.3 시 [0,1]
5. **초기 구간 피크 패널티**: 0~0.5초 구간 전용 패널티
6. **기준선 대비 개선량**: EWMA 기준선 대비 개선 (정착시간 40%, 오버슈트 30%, ESS 30%)
7. **통합 보상 스코어**: 0.30×S_ts + 0.35×S_mp + 0.20×S_ess + 0.15×S_band + 0.05×S_u - 0.15×P_fail - P_early + 0.10×progress + 0.10×I_improve
   (P_early: 초기 구간 피크 패널티, 최대값 0.2, 가중치 없이 직접 차감)
8. **기준선 중심화 + tanh**: r_centered = reward_score - rew_baseline, reward = tanh(2.0 × r_centered)
9. **안전 위반**: 힘 > ±100N → reward = -1.0

### 가중치
- 정착시간(0.30), 오버슈트(0.35), ESS(0.20), 밴드유지(0.15), 제어입력(0.05), 추종실패(-0.15), 초기피크(직접 차감, 최대 -0.20), PBRS(+0.10), 기준선개선(+0.10)

---

## 🎯 보상 시스템 상세

### 보상 계산 단계

#### Step 1: 핵심 지표 계산

```
1. RMSE (Root Mean Square Error)
   └─ 전체 추종 오차의 제곱평균제곱근

2. Overshoot (%)
   └─ 목표값 대비 최대 편차 (음수 목표: 더 큰 음수)

3. Settling Time (초)
   └─ ±1.0N 밴드 내 0.5초 연속 유지 시작 시간 (0.2초 이후 검사)

4. 정상상태 오차 (ESS, N)
   └─ 마지막 10% 구간 평균 오차

5. 밴드 유지율 (0~1)
   └─ ±1.5N 범위 내 유지 비율

6. 제어 입력 RMS (0~1)
   └─ PI 출력 RMS 정규화
```

#### Step 2: 지수 스코어 변환 (0~1)

```
각 지표를 지수 함수로 변환:

S_ts   = exp(-settling_time / 5.0)      # τ=5.0s
S_mp   = exp(-overshoot / 8.0)           # τ=8.0% (더 민감하게)
S_ess  = exp(-|ESS| / 1.0)              # τ=1.0N
S_band = band_ratio                      # 직접 사용
S_u    = exp(-u_rms / 0.5)              # τ=0.5

→ 모든 스코어는 [0, 1] 범위
```

#### Step 3: PBRS (Potential-Based Reward Shaping)

```
Potential 함수: φ(s) = -|error| / |target_force|

개선분: F = γ * φ(s') - φ(s)  (γ=0.99)

Progress = mean(max(F, 0))  # 양수만 사용

→ 순간 개선 신호 제공 (그라디언트 향상)
```

#### Step 4: 초기 구간 피크 패널티

```
초기 구간 (0~0.5초) 전용 패널티:
- early_peak_penalty = max(0, (peak_force - target_force) / |target_force|)
- P_early = min(early_peak_penalty * SCALE, MAX)

→ 초기 압력 급격한 변화에 대한 명시적 페널티
```

#### Step 5: 기준선 중심화

```
기준선 업데이트 (EWMA):
baseline_ts  = 0.9 * baseline_ts  + 0.1 * settling_time
baseline_mp  = 0.9 * baseline_mp  + 0.1 * overshoot
baseline_ess = 0.9 * baseline_ess + 0.1 * |ESS|

개선량 계산:
I_ts  = clip((baseline_ts - settling_time) / τ_ts, -1, 1)
I_mp  = clip((baseline_mp - overshoot) / τ_mp, -1, 1)
I_ess = clip((baseline_ess - |ESS|) / τ_ess, -1, 1)

I_improve = 0.4 * I_ts + 0.3 * I_mp + 0.3 * I_ess
```

#### Step 6: 통합 보상 스코어

```
reward_score = 
    0.30 * S_ts          # 정착시간
  + 0.35 * S_mp          # 오버슈트 (가중치 강화)
  + 0.20 * S_ess         # 정상상태 오차
  + 0.15 * S_band        # 밴드 유지
  + 0.05 * S_u           # 제어 노력
  - 0.15 * P_fail        # 추종 실패 패널티
  - P_early              # 초기 구간 피크 패널티 (최대값 0.2, 가중치 없이 직접 차감)
  + 0.10 * progress      # PBRS
  + 0.10 * I_improve     # 기준선 대비 개선

→ [0, 1] 범위 (대략, P_early 최대값 0.2 고려 시)
```

#### Step 7: 최종 보상 변환

```
보상 기준선 업데이트 (EWMA, β=0.99):
rew_baseline = 0.99 * rew_baseline + 0.01 * reward_score

중심화:
r_centered = reward_score - rew_baseline

tanh 소프트클립:
reward = tanh(2.0 * r_centered)

→ 최종 범위: [-1.0, 1.0]
```

#### Step 8: 안전 위반 처리

```
if |extreme_force| > 100N:
    reward = -1.0  # 하드 패널티
    에피소드 즉시 종료
    랜덤 PID로 재시작
```

---

## 📊 세그먼트 분할 학습

### 개념

기존: **1 에피소드 = 1 transition** (비효율적)
개선: **1 에피소드 = 5 transition** (학습 효율 5배 향상)

### 구현 방법

```
에피소드 (10초, 10000개 샘플)
│
├─ 세그먼트 1: [0~2초, 2000개] → transition 1
├─ 세그먼트 2: [2~4초, 2000개] → transition 2
├─ 세그먼트 3: [4~6초, 2000개] → transition 3
├─ 세그먼트 4: [6~8초, 2000개] → transition 4
└─ 세그먼트 5: [8~10초, 2000개] → transition 5
```

### 각 세그먼트 상태 생성

```
세그먼트 i의 상태:
├─ 기존 12차원: PID gain, 목표 힘, 통계 등
└─ 궤적 요약 8차원: 해당 세그먼트의 성능 지표
   ├─ overshoot, settling_time, rmse, band_ratio
   ├─ oscillation_freq, oscillation_amp
   └─ rise_time, steady_state_error
```

### 장점

1. **학습 효율 향상**: 같은 데이터로 5배 학습
2. **시간적 세밀한 학습**: 에피소드 내부 동작 패턴 학습
3. **안정적 수렴**: 더 많은 경험으로 안정적 학습

---

## 📁 리팩토링된 모듈 구조 (3876줄 → 16개 모듈)

**원본**: `experiment/JY_PID_Gain_SAC_MDP_monitor_3_reset.py` (3876줄 단일 파일)  
**결과**: 16개 모듈로 분리하여 유지보수성 및 확장성 향상

**핵심 실행 모듈**: `__main__.py`는 모듈 실행 엔트리 포인트로 기본 설정 사용 및 시그널 핸들러 설치, `main.py`는 argparse 기반 명령줄 인터페이스로 유연한 설정 지원.

**설정 관리**: `config.py` (165줄)는 Config 데이터클래스로 타입 안전한 설정 관리(신경망/PID/에피소드/통신/저장 설정), `constants.py` (131줄)는 모든 상수 중앙 집중(PID 범위 Kp[3-40]/Ki[0-40]/Kd[1e-4~5e-2], 보상 가중치, 세그먼트 분할 NUM_SEGMENTS=5, 물리 상수 SETTLING_BAND_TOLERANCE=1.0N).

**핵심 RL 컴포넌트**: `agent.py` (422줄)는 SAC 알고리즘 구현으로 Actor(128-128 MLP, log_std∈[-2.5,-0.3])/Critic(Dual Q-Network)/ReplayBuffer(10,000개), 표준편차 Annealing(1.0→0.5), PID gain 액션 선택, `env.py` (1356줄)는 학습 환경 메인 루프로 보상 계산(지수 스코어+PBRS→[-1,1]), 세그먼트 분할 처리, 안전 위반 처리, 20차원 상태 생성, `comm.py` (357줄)는 TCP/IP 통신 관리로 1kHz 데이터 수신(29바이트 패킷, CRC16 체크섬), PID gain 전송(19바이트), 스레드 기반 수신 루프.

**모니터링**: `monitor.py` (188줄)는 실시간 GUI 모니터로 Force/보상 그래프(10Hz, 30초 롤링 윈도우), 멀티프로세싱 분리, TkAgg/Agg 백엔드 자동 전환.

**유틸리티**: `utils/math_utils.py` (120줄)는 scale_action_to_pid(액션[-1,1]³→PID gain, Kd 로그 스케일), create_initial_state(20차원 상태 생성), `utils/data_saver.py` (59줄)는 데이터 저장 래퍼로 reward breakdown/제어 성능 지표 일괄 저장, `utils/signals.py` (76줄)는 Ctrl+C 안전 종료 처리로 SIGINT/SIGTERM 핸들러, 데이터 저장 및 learning_done 신호 전송.

**로깅 시스템**: `loggers/base_logger.py` (28줄)는 AppLogger 이모지 지원 로거(INFO/SUCCESS/WARNING/ERROR/DEBUG), `loggers/control_performance.py` (906줄)는 제어공학 지표 계산 및 저장으로 10개 핵심 지표(RMSE/Overshoot/Settling Time/IAE 등), CSV/PNG 출력(300 DPI), `loggers/reward_breakdown.py` (267줄)는 스텝별 보상 분석, 에피소드별 보상 통계, CSV/PNG 시각화, `loggers/learning_done.py` (21줄)는 학습 완료 타임스탬프 폴더 생성(learning_done_YYMMDD_HHhMMm/).

---

## 🔧 각 모듈 상세 역할

### 핵심 모듈

#### 1. **__main__.py** / **main.py** (실행 엔트리 포인트)

**역할:**
- 프로그램 시작점
- 설정 생성 및 환경 초기화
- 학습 루프 실행

**주요 기능:**
- `create_config()`: 설정 객체 생성
- `PIDGainEnvironment()`: 환경 객체 생성
- `env.run_pid_optimization_training()`: 메인 학습 루프 호출
- 시그널 핸들러 설치 (Ctrl+C 안전 종료)

#### 2. **config.py** (설정 관리)

**역할:**
- 타입 안전한 설정 관리 (dataclass 사용)
- 설정 생성 및 변환

**주요 클래스:**
- `Config`: 모든 설정을 담는 데이터클래스
  - 신경망 설정 (state_dim, action_dim, hidden_dim, lr 등)
  - PID 설정 (pid_range)
  - 에피소드 설정 (episode_seconds, target_force, episodes)
  - 통신 설정 (recv_freq_hz, host, port)
  - 저장 설정 (model_save_dir, log_dir)

**주요 함수:**
- `create_config()`: 기본 설정 생성
- `change_episode_length()`: 에피소드 길이 동적 변경

#### 3. **constants.py** (상수 정의)

**역할:**
- 모든 하드코딩된 값 중앙 집중 관리

**주요 상수:**
- **신경망**: 학습률, 은닉층 크기, 감쇠율 등
- **PID 범위**: Kp[3.0, 40.0], Ki[0.0, 40.0], Kd[1e-4, 5e-2]
- **세그먼트 분할**: NUM_SEGMENTS=5, SEGMENT_LENGTH_S=2.0
- **보상 가중치**: 각 지표별 가중치 (W_TS=0.30, W_MP=0.35, W_ESS=0.20 등)
- **물리 상수**: BAND_TOLERANCE_N=1.5, SAFETY_FORCE_LIMIT=100.0N
- **탐색 설정**: ACTOR_LOG_STD_MAX=-0.3, ACTOR_INITIAL_ALPHA=0.1, STD_ANNEAL_FINAL=0.5

#### 4. **agent.py** (SAC 에이전트)

**역할:**
- Soft Actor-Critic 알고리즘 구현
- PID gain 액션 선택
- 신경망 학습

**주요 클래스:**

**Actor:**
- 입력: 20차원 상태
- 출력: 액션 평균(mean), 로그 표준편차(log_std)
- 네트워크: 128-128 2층 MLP
- 탐험 범위: log_std ∈ [-2.5, -0.3]

**Critic (Dual Q-Network):**
- 입력: 23차원 (상태 20 + 액션 3)
- 출력: Q 값 2개 (Q1, Q2)
- 네트워크: 128-128 2층 MLP × 2

**ReplayBuffer:**
- 크기: 10,000개
- 저장 형식: (state, action, reward, next_state, done)

**PIDGainSACAgent:**
- 액션 선택: `select_action()` (탐험), `select_action_random()` (안전 위반 시)
- 경험 저장: `store_transition()` (세그먼트별)
- 학습 업데이트: `update_parameters_one_step()`
- 표준편차 Annealing: `update_std_scale()` (0~150 에피소드, 1.0→0.5)
- Warm-start: LHS 샘플링으로 초기 버퍼 시딩
- 동적 target_entropy: 초기 100ep는 -1.2×action_dim, 이후 -1.0×action_dim

#### 5. **env.py** (학습 환경) ⭐ 가장 큰 파일

**역할:**
- 메인 학습 루프 관리
- 보상 계산
- 세그먼트 분할 처리
- 안전 위반 처리

**주요 클래스:** `PIDGainEnvironment`

**주요 메서드:**

**`run_pid_optimization_training()`**:
- 전체 학습 루프 실행
- 에피소드별 PID gain 선택 → 데이터 수집 → 보상 계산 → 학습

**`calculate_episode_reward()`**:
- 보상 계산 (지수 스코어 + PBRS + 기준선 중심화)
- 최종 범위: [-1.0, 1.0]
- 안전 위반 감지
- 초기 구간 피크 패널티 계산

**`_build_segment_state()`**:
- 세그먼트별 20차원 상태 벡터 생성
- 궤적 요약 8차원 추가

#### 6. **comm.py** (TCP/IP 통신)

**역할:**
- 로봇제어PC와의 실시간 통신 관리

**주요 클래스:** `PIDGainCommunicator`

**주요 기능:**
- **수신** (1kHz):
  - 패킷 구조: 29바이트 (SOF + 힘 데이터 + 상태 데이터 + 체크섬)
  - CRC16 체크섬 검증
  - 별도 스레드에서 지속 수신
- **송신** (에피소드당 1회):
  - 패킷 구조: 19바이트 (SOF + PID gains + 플래그 + 체크섬)
  - PID gain 전송
  - episode_done, learning_done 플래그

**통신 프로토콜:**
```
수신 패킷 (29바이트):
[SOF(2)] [current_force(4)] [target_force(4)] [force_error(4)]
[force_error_dot(4)] [force_error_int(4)] [pi_output(4)]
[sander_active(1)] [checksum(2)]

송신 패킷 (19바이트):
[SOF(2)] [Kp(4)] [Ki(4)] [Kd(4)]
[timing_accurate(1)] [episode_done(1)] [learning_done(1)] [checksum(2)]
```

#### 7. **monitor.py** (실시간 GUI)

**역할:**
- 학습 진행 상황 실시간 시각화

**주요 클래스:** `RLRealtimeMonitor`

**주요 기능:**
- Force 추적 그래프: 현재 힘 vs 목표 힘 (롤링 윈도우 30초)
- 보상 그래프: 에피소드별 보상 추이
- 멀티프로세싱: 별도 프로세스에서 실행 (메인 프로세스 블로킹 방지)
- 백엔드 감지: TkAgg (GUI 가능) → Agg (헤드리스) 자동 전환
- 갱신 주파수: 10 Hz

#### 8. **utils/math_utils.py** (수학 유틸리티)

**역할:**
- PID 액션 스케일링
- 상태 벡터 생성

**주요 함수:**

**`scale_action_to_pid()`**:
- 액션 [-1, 1]³ → 실제 PID gains
- Kp, Ki: 선형 매핑
- Kd: 로그 스케일 매핑 (1e-4~5e-2 범위)

**`create_initial_state()`**:
- 20차원 초기 상태 벡터 생성
- 기존 12차원 + 궤적 요약 8차원 (초기값 0)

#### 9. **utils/data_saver.py** (데이터 저장 래퍼)

**역할:**
- 모든 데이터 저장 통합 관리

**주요 클래스:** `DataSaver`

**주요 함수:**
- `save_all_data()`: reward breakdown, 제어 성능 지표 일괄 저장

#### 10. **utils/signals.py** (시그널 처리)

**역할:**
- Ctrl+C 안전 종료 처리

**주요 함수:**
- `install_signal_handlers()`: SIGINT, SIGTERM 핸들러 설치
- `signal_handler()`: 종료 시 데이터 저장 및 learning_done 신호 전송

#### 11. **loggers/** (로깅 시스템)

**base_logger.py**:
- `AppLogger`: 이모지 지원 로거
- 로그 레벨: INFO, SUCCESS, WARNING, ERROR, DEBUG

**control_performance.py** ⭐ 큰 파일:
- `ControlPerformanceLogger`: 제어공학 지표 계산 및 저장
- **10개 핵심 지표**:
  1. RMSE
  2. Steady-State Error
  3. Rise Time
  4. Settling Time
  5. Overshoot
  6. IAE
  7. Input RMS
  8. Total Variation
  9. Band Ratio
  10. Error Variance
- 출력: CSV (에피소드별 값), PNG (추이 그래프, 300 DPI)

**reward_breakdown.py**:
- `RewardBreakdownLogger`: 스텝별 보상 분석
- 출력: CSV (스텝별 구성), PNG (에피소드별 추이)
- 보상 구성 요소 시각화: reward_score, r_centered, r_baseline

**learning_done.py**:
- `LearningDoneLogger`: 학습 완료 시 타임스탬프 폴더 생성
- 폴더명: `learning_done_YYMMDD_HHhMMm/`

---

## 📊 출력 파일

### experiment_logs/learning_done_YYMMDD_HHhMMm/

#### control_performance/ - 제어 성능 지표

**CSV 파일 (에피소드별 값):**
- `rmse.csv`, `overshoot.csv`, `settling_time.csv`
- `steady_state_error.csv`, `rise_time.csv`, `iae.csv`
- `input_rms.csv`, `total_variation.csv`
- `band_ratio.csv`, `error_variance.csv`

**PNG 그래프 (300 DPI, 논문용):**
- 각 지표별 추이 그래프 (에피소드별)
- `comprehensive_dashboard.png`: 10개 지표 종합 대시보드
- `step_dashboard.png`: Step 축 지표 대시보드
- `force_tracking_curve.png`: 힘 추적 곡선
- `error_time_series.png`: 오차 시계열
- `control_input_series.png`: 제어 입력 시계열
- `reward_breakdown_step.png`: 스텝별 보상 구성

**요약 파일:**
- `performance_summary.csv`: 전체 통계 (평균, 표준편차, 최소, 최대)

#### reward_breakdown/ - 보상 분석

**CSV 파일:**
- `reward_breakdown.csv`: 스텝별 보상 구성 요소
- `episode_rewards.csv`: 에피소드별 총 보상

**PNG 그래프:**
- `episode_rewards.png`: 에피소드별 보상 추이
- `reward_breakdown_prog_ep1-45.png`: Progress 보상 추이
- `reward_breakdown_inband_ep1-45.png`: In-band 보상 추이
- `reward_breakdown_edot_ep1-45.png`: Error dot 추이
- `reward_breakdown_du_ep1-45.png`: Control input 변화 추이
- `reward_breakdown_reward_ep1-45.png`: 총 보상 추이

### saved_agents/

**모델 파일:**
- `best_pid_agent_episode_{N}_reward_{R}.pth`: 최고 성능 에이전트
- 저장 내용: Actor/Critic 가중치, 옵티마이저 상태, 에피소드 보상 히스토리

---

## ⚙️ 설정 변경

### __main__.py에서 기본 설정 변경

```python
def main():
    # ========== 설정 변경 포인트 ==========
    RECV_FREQUENCY_HZ = 1000           # 수신 주파수 (Hz)
    EPISODE_LENGTH_SECONDS = 10.0      # 에피소드 길이 (초)
    # ====================================
    ...
```

### main.py에서 명령줄 인자로 변경

```bash
python3 main.py \
  --episodes 500 \
  --target-force -40.0 \
  --episode-seconds 10.0
```

### constants.py에서 상수 변경

```python
class Constants:
    # 학습 설정
    DEFAULT_EPISODES = 500
    DEFAULT_LR_ACTOR = 1e-4
    DEFAULT_LR_CRITIC = 2e-4
    
    # PID 범위
    DEFAULT_PID_RANGE = {
        "Kp": (3.0, 40.0),
        "Ki": (0.0, 40.0),
        "Kd": (1e-4, 5e-2),
    }
    
    # 보상 가중치
    SCORE_W_TS = 0.30      # 정착시간
    SCORE_W_MP = 0.35      # 오버슈트 (강화)
    SCORE_W_ESS = 0.20     # 정상상태 오차
    SCORE_W_BAND = 0.15    # 밴드 유지
    SCORE_W_U = 0.05       # 제어 노력
```

---

## 🔍 주요 개선사항

### 리팩토링 전 ❌
- 3876줄 단일 파일
- 코드 찾기 어려움
- 유지보수 어려움
- 재사용/테스트 불가

### 리팩토링 후 ✅
- 16개 모듈로 분리
- 명확한 구조
- 독립 테스트/재사용 가능
- 기존 코드 호환

### 핵심 기술적 개선
1. **세그먼트 분할 학습**: 1 에피소드 = 5 transition (학습 효율 5배)
2. **20차원 상태 공간**: 궤적 요약 정보 추가
3. **Fine-tuning 최적화**: 국소 탐색 범위, 탐험 축소
4. **표준편차 Annealing**: 점진적 탐험 감소
5. **안전 중심 설계**: 안전 위반 자동 감지 및 재시작
6. **PID 범위 확장**: Kp[3-40], Ki[0-40], Kd[1e-4~5e-2]
7. **보상 함수 강화**: 오버슈트 가중치 0.35, 초기 구간 피크 패널티 추가
8. **탐색 전략 개선**: log_std_max=-0.3, 초기 alpha=0.1, 동적 target_entropy

---

## 📊 실험 결과 및 문제점 분석

### 200 에피소드 학습 결과 (수정된 PID 범위 적용)

#### 핵심 문제점

**1. 오버슈트가 모든 에피소드에서 90% 이상 유지** ⚠️ **매우 심각**

**관찰:**
- 초기 (0-25ep): 20% → 90% 이상으로 급격히 상승
- 전체 (25-200ep): 90% 이상을 유지하며 **전혀 개선되지 않음**
- 원인: 초기 접촉 압력 0 → 0.3MPa 급격한 변화

**2. 보상이 개선되지 않음**

**관찰:**
- `reward_score (MA10)`: 0 근처에서 정체
- `r_centered (MA10)`: -0.5 ~ -0.3 사이에서 변동
- 실제 성능 개선이 없어 기준선 중심화만 작동

**3. 제어 성능 지표 전반적으로 악화 또는 정체**

**관찰:**
- 정착 시간: 0.7초 이상 유지하며 개선 없음
- 밴드 유지 비율: 20-30% 사이에서 매우 낮은 수준 유지
- RMSE: 4.5-10.5 사이에서 높은 변동성
- Rise Time: 0.05-0.30 사이에서 극심한 변동성

#### 학습 진행 여부 판단

**❌ 학습이 되지 않는 증거:**
1. 보상 정체: `reward_score (MA10)`가 0 근처에서 개선 없음
2. 제어 성능 악화: 오버슈트가 90% 이상으로 유지
3. 높은 변동성: 모든 지표에서 변동성이 줄어들지 않음
4. 100ep 이후 정체: 추가 개선 없음

**🔍 원인 분석:**
1. **환경 난이도**: 초기 접촉 압력 0 → 0.3MPa 급격한 변화가 PID 제어의 한계를 넘어섬
2. **보상 함수**: 오버슈트 페널티가 있지만, 에이전트가 이를 피할 수 있는 PID 게인을 찾지 못함
3. **탐색 범위**: 현재 PID 범위 내에서 최적해가 없을 수 있음
4. **초기 피크 패널티**: `early_peak_penalty`가 있지만 충분하지 않을 수 있음

---

## 💡 해결 방안 및 개선 계획

### ✅ 즉시 적용: 초기 접촉 압력 변경 (사용자 제안)

**초기 접촉 압력을 0.1MPa로 올리기**

**기대 효과:**
- 초기 과도 응답 강도 감소
- 오버슈트 감소 가능성 높음
- 에이전트가 더 쉽게 학습 가능한 환경 제공

**구현 방법:**
- 로봇 제어 PC 측에서 초기 접촉 압력을 0.1MPa로 설정
- 에피소드 시작 시 즉시 0.1MPa로 설정된 상태에서 시작

---

### 🔍 State Space 개선 필요

#### 현재 State Space의 문제점

**1. 초기 압력 급격한 변화 감지 불가** ⚠️ **매우 심각**

**문제:**
- 초기 접촉 압력 0 → 0.3MPa 급격한 변화를 감지할 정보가 **없음**
- 힘의 변화율(force rate, dF/dt) 정보가 없음
- 초기 구간 피크 정보가 별도로 없음 (overshoot는 전체 구간 기준)

**결과:**
- 에이전트가 초기 급격한 변화를 인지하지 못함
- 초기 오버슈트를 예방하는 PID 게인을 학습하기 어려움

#### 개선 방안: 초기 구간 정보 추가 (4차원)

**추가할 State:**
```
[20] early_peak_force      # 초기 구간(0~0.5초) 최대/최소 힘
[21] early_force_rate      # 초기 구간 힘 변화율 (dF/dt)
[22] early_rmse            # 초기 구간 RMSE
[23] initial_force          # 세그먼트 시작 힘
```

**예상 효과:**
- ✅ 초기 압력 급격한 변화를 명확히 감지 가능
- ✅ 초기 오버슈트를 예방하는 PID 게인 학습 가능
- ✅ State space: 20 → 24차원

**구현 필요:**
- `_build_segment_state()` 메서드 수정
- STATE_DIM: 20 → 24차원 업데이트
- Actor/Critic 네트워크 입력 차원 자동 조정

---

### 🎯 커리큘럼 학습 계획

#### 최종 목표
- **목표 접촉력**: -35N ~ -50N 범위에서 무작위로 변경
- **변경 횟수**: 한 에피소드에 3번 변경 (예: 0초, 10초, 20초)
- **즉각 대응**: 목표 변경 시 즉시 최적 PID 게인 출력

#### Phase별 계획

**Phase 1: 고정 목표 (현재)**
- 목표 접촉력: -40N 고정
- 완료 조건: 평균 보상 > 0.5 (최근 50 에피소드)
- 전이학습: 없음 (처음부터 학습)

**Phase 2: 무작위 목표 (단일 에피소드)**
- 목표 접촉력: -35N ~ -50N 범위에서 무작위 선택 (에피소드당 1개)
- 완료 조건: 평균 보상 > 0.4
- 전이학습: Phase 1 모델 로드
- State space: 20차원 → 22차원 (목표 범위 정보 추가)

**Phase 3: 2번 변경**
- 목표 접촉력: -35N ~ -50N 범위에서 무작위 변경 (에피소드당 2번)
- 변경 시점: 0초, 5초
- 완료 조건: 평균 보상 > 0.35
- 전이학습: Phase 2 모델 로드
- State space: 22차원 → 26차원 (목표 히스토리 정보 추가)

**Phase 4: 3번 변경 (최종 목표)**
- 목표 접촉력: -35N ~ -50N 범위에서 무작위 변경 (에피소드당 3번)
- 변경 시점: 0초, 10초, 20초 (30초 에피소드)
- 완료 조건: 평균 보상 > 0.3
- 전이학습: Phase 3 모델 로드
- State space: 26차원 유지

**Phase 5: -30N까지 확장 (선택적)**
- 목표 접촉력: -30N ~ -50N 범위로 확장
- 변경 횟수: 3번 유지
- 완료 조건: 평균 보상 > 0.25
- 전이학습: Phase 4 모델 로드

#### 전이학습 전략
- 각 Phase 완료 시 모델 저장 (`saved_agents/phase_{N}/`)
- 다음 Phase 시작 시 이전 모델 로드
- 네트워크 구조 변경 최소화 (입력 차원만 변경)
- Fine-tuning: 초기 학습률을 낮춰서 점진적 조정

---

## 🧠 신경망 구조 확장 필요성 분석

### 현재 구조
- **Actor**: 128-128 2층 MLP (20차원 입력 → 3차원 출력)
- **Critic**: Dual Q-Network, 각각 128-128 2층 MLP (23차원 입력 → 1차원 출력)

### Phase별 구조 변경 계획

**Phase 1 (현재)**
- 상태: 20차원
- Actor: 128-128 MLP (20 → 128 → 128 → 3)
- Critic: 128-128 MLP (23 → 128 → 128 → 1) × 2

**Phase 2-3: 상태 공간만 확장 (권장)**
- 상태: 22차원 → 26차원 (목표 정보 추가)
- Actor: 입력 차원만 변경 (20 → 22 → 26)
- Critic: 입력 차원만 변경 (23 → 25 → 29)
- **구조 변경 없음, 전이학습 가능!**

**LSTM 도입 검토:**
- ❌ **현재는 불필요**: 명시적 상태 정보로 충분
- ✅ **필요 시 검토 조건**:
  - 명시적 상태 정보로 해결 불가능
  - 장기 의존성 필요 (10초 이상)
  - Phase 3+에서도 성능 부족 시

**결론:**
- **상태 공간 확장만으로 충분**: 20차원 → 26차원
- **네트워크 구조 변경 최소화**: 전이학습 유지
- **LSTM은 마지막 수단**: Phase 3+에서도 성능 부족 시에만 검토

---

## 🐛 문제 해결

### ImportError: No module named 'pid_gain_rl'
**원인**: 잘못된 경로에서 실행  
**해결**:
```bash
# 올바른 경로로 이동
cd /home/katech/Robot-Polishing-RL-system/py_rl

# 방법 1 (권장)
python3 -m pid_gain_rl

# 방법 2
cd pid_gain_rl
python3 main.py
```

### AttributeError: partially initialized module 'logging'
**원인**: `logging` 디렉토리가 Python 내장 모듈과 충돌  
**해결**: ⚠️ **이미 해결됨** - `logging/` → `loggers/`로 이름 변경 완료

### ModuleNotFoundError
**원인**: 절대 경로가 sys.path에 없음  
**해결**: 반드시 위 경로에서 실행하거나 `PYTHONPATH` 설정
```bash
export PYTHONPATH=/home/katech/Robot-Polishing-RL-system/py_rl:$PYTHONPATH
```

### 연결 실패
- 로봇 PC 실행 확인
- 포트 8888 확인
- 네트워크 연결 확인

### GUI 미표시
- 헤드리스 모드 자동 감지 (정상)
- PNG 파일로 저장됨

---

## 📋 다음 단계 체크리스트

### Phase 1 개선 (즉시 적용)

- [ ] **초기 접촉 압력 변경**: 0 → 0.1MPa로 설정
- [ ] **State Space 확장**: 20차원 → 24차원 (초기 구간 정보 추가)
  - [ ] `_build_segment_state()` 메서드 수정
  - [ ] STATE_DIM 업데이트
  - [ ] Actor/Critic 네트워크 입력 차원 자동 조정 확인
- [ ] **200 에피소드 재학습**
- [ ] **결과 분석**: 보상 구성 요소 그래프 재분석
  - [ ] reward_score, r_centered, r_baseline 추이 확인
  - [ ] 오버슈트 개선 여부 확인
  - [ ] 제어 성능 지표 개선 여부 확인

### Phase 1 추가 개선 (필요 시)

- [ ] **보상 함수 강화**
  - [ ] 초기 구간 피크 패널티 강화 (0.2 → 0.4)
  - [ ] 오버슈트 가중치 추가 강화 (0.35 → 0.45)
- [ ] **PID 범위 재조정** (필요 시)
  - [ ] Kd 범위 축소 (1e-4~5e-2 → 1e-3~1e-2)
  - [ ] Kp, Ki 상한 축소 (40 → 30)
- [ ] **컨트롤러 측 사전 조치** (장기)
  - [ ] Anti-windup 구현
  - [ ] D-on-measurement with LPF 구현
  - [ ] Input rate limit 구현
  - [ ] Bumpless transfer 구현

### Phase 2 준비 (Phase 1 완료 후)

- [ ] **Phase 1 모델 저장**
- [ ] **State Space 확장**: 20차원 → 22차원 (목표 범위 정보)
- [ ] **목표 접촉력 무작위 선택**: -35N ~ -50N 범위
- [ ] **전이학습 구현**: Phase 1 모델 로드
- [ ] **200 에피소드 학습**

---

## 📝 참고

- 원본: JY_PID_Gain_SAC_MDP_monitor_3_reset.py (3876줄)
- 리팩토링: 2024-10-30 (지속 업데이트)
- 구조 유지: experiment/, experiment_logs/, saved_agents/
- 최종 목표: -30N까지 위치 제어, 동적 목표 환경 (-35N ~ -50N, 3번 변경)

---

**리팩토링 완료!** 🎉  
3876줄 → 16개 모듈로 분리 완료

**현재 상태**: Phase 1 학습 중, 오버슈트 문제 해결 진행 중
