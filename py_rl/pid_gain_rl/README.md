# PID Gain Optimization using Reinforcement Learning

## 📋 연구 개요

본 연구는 **강화학습(Reinforcement Learning)을 활용한 공압 연마 시스템의 PID 게인 자동 최적화**를 목표로 합니다. 기존의 수동 튜닝 방식이나 경험 기반 PID 게인 설정 방식을 대체하여, 실시간 제어 성능 피드백을 기반으로 최적의 PID 게인을 자동으로 학습합니다.

### 🎯 핵심 목표
- **자동 PID 게인 튜닝**: 수작업 튜닝 과정 자동화
- **제어 성능 최적화**: 목표 힘(45N) 유지 성능 향상
- **안정성 보장**: 오버슈트, 진동, 포화 최소화
- **빠른 정착**: 목표 힘으로의 빠른 수렴 시간 달성

### 🔬 연구 방법
- **알고리즘**: Soft Actor-Critic (SAC) - 연속 행동 공간에 적합한 Off-policy 알고리즘
- **제어 대상**: 공압 연마 로봇의 힘 제어 시스템
- **학습 방식**: 에피소드 기반 학습 (각 에피소드 15초)
- **통신 방식**: TCP/IP 소켓 통신 (1000Hz 수신)

---

## 🗂️ 폴더 구조

```
pid_gain_rl/
├── experiment/
│   └── JY_PID_Gain_SAC_1_test.py    # 메인 학습 스크립트
├── experiment_logs/                  # 학습 결과 로그
│   └── learning_done_YYMMDD_HHhMMm/
│       ├── control_performance/      # 제어 성능 지표 (CSV, PNG)
│       ├── reward_breakdown/         # 보상 분해 분석 (CSV, PNG)
│       ├── episode_rewards.csv       # 에피소드별 보상
│       └── episode_rewards.png       # 보상 추이 그래프
├── saved_agents/                     # 학습된 모델 저장
│   └── test_best_agent_episode_X_reward_Y.pth
├── tele/
│   └── test_server.py               # 테스트용 서버
└── README.md                         # 본 문서
```

---

## 📄 코드 구조 및 주요 컴포넌트

### 1. 핵심 클래스

#### 🧠 `PIDGainSACAgent`
- **역할**: SAC 강화학습 에이전트
- **구성요소**:
  - Actor Network: PID 게인(Kp, Ki, Kd) 행동 결정
  - Critic Network (2개): Q-value 추정 (Double Q-learning)
  - Target Networks: 안정적 학습을 위한 타겟 네트워크
  - Replay Buffer: 경험 저장 및 배치 샘플링
- **특징**:
  - 자동 엔트로피 조정 (α 자동 튜닝)
  - Tanh activation으로 PID 게인 범위 제한

#### 🌍 `PIDGainOptimizationEnvironment`
- **역할**: 강화학습 환경 관리 및 보상 계산
- **주요 기능**:
  - 에피소드 실행 및 데이터 수집
  - 18개 제어공학 지표 기반 보상 계산
  - 학습 진행 모니터링
  - 최고 성능 모델 저장
- **핵심 메서드**:
  - `run_pid_optimization_training()`: 메인 학습 루프
  - `calculate_episode_reward()`: 보상 계산 (8개 정규화 지표)
  - `collect_episode_data()`: 에피소드 데이터 수집

#### 📡 `PIDGainCommunicator`
- **역할**: 로봇 제어 PC와 TCP/IP 통신
- **기능**:
  - PID 게인 전송 (에피소드 시작 시)
  - 힘 데이터 수신 (1000Hz)
  - 통신 오류 처리 및 재연결

#### 📊 로깅 시스템
- **`RewardBreakdownLogger`**: 보상 구성 요소별 분석 및 저장
- **`ControlPerformanceLogger`**: 제어 성능 지표 18개 추적 및 시각화
- **`LearningDoneLogger`**: 학습 완료 시 통합 데이터 저장

---

### 2. 상태 공간 (State Space)

총 **12차원** 상태 벡터:

```python
state = [
    # [0-5] 로봇 제어 PC에서 전송
    force_error,           # 힘 오차 (정규화)
    force_error_dot,       # 힘 오차 변화율
    current_force,         # 현재 힘 (정규화)
    pi_output,             # PI 출력 (정규화)
    sander_active,         # 샌더 활성 상태 (0/1)
    episode_progress,      # 에피소드 진행률 (0~1)
    
    # [6-11] 강화학습 PC에서 계산
    prev_Kp, prev_Ki, prev_Kd,  # 이전 PID 게인 (정규화)
    prev_rmse,             # 이전 RMSE (정규화)
    prev_band_ratio,       # 이전 밴드 유지 비율
    history_best_band_ratio # 히스토리 최고 밴드 유지 비율
]
```

---

### 3. 행동 공간 (Action Space)

총 **3차원** 연속 행동 벡터:

```python
action = [Kp, Ki, Kd]  # 각각 [-1, 1] 범위로 출력 후 스케일링

# 스케일링 범위 (기준값 P=80, I=130, D=0 중심 ±30%)
Kp: [56.0, 104.0]
Ki: [91.0, 169.0]
Kd: [0.0, 15.0]
```

---

### 4. 보상 함수 (Reward Function)

**8개 정규화 지표 기반 연속형 보상**:

```python
reward = (
    # 긍정적 보상 (최대화)
    + 1.5 × band_ratio              # ±0.5N 밴드 유지 비율 (핵심)
    
    # 부정적 페널티 (최소화)
    - 1.2 × rmse_normalized         # RMSE (제어 정확도)
    - 0.7 × overshoot_normalized    # 오버슈트 (과도 응답)
    - 0.6 × settling_normalized     # 정착 시간
    - 0.4 × variance_normalized     # 오차 분산 (진동성)
    - 0.3 × u_rms_normalized        # PI 출력 RMS (제어 노력)
    - 0.3 × du_rms_normalized       # PI 출력 변화율
    - 0.4 × saturation_ratio        # 포화 비율
    
    # 이상적 성능 보너스 (+1~+3)
    + ideal_bonus  # 6개 조건 달성 시
    
    # 안전 위반 시 큰 페널티 (-100)
)
```

#### 제어 성능 임계값
- **밴드 허용 오차**: ±0.5N (논문 스펙)
- **정착 판정**: ±0.5N 또는 ±1% 범위에서 1초 연속 유지
- **안전 힘 제한**: 100N
- **PI 출력 최대값**: 0.4 MPa
- **포화 임계값**: 95% 이상

---

### 5. 학습 파라미터

```python
# 신경망
STATE_DIM = 12
ACTION_DIM = 3
HIDDEN_DIM = 256
LEARNING_RATE = 3e-4

# 강화학습
GAMMA = 0.99                    # 할인 인수
TAU = 0.01                      # 소프트 업데이트 계수
BATCH_SIZE = 128                # 배치 크기
REPLAY_BUFFER_SIZE = 2000       # 리플레이 버퍼 크기
UPDATES_PER_EPISODE = 16        # 에피소드당 업데이트 횟수

# 에피소드
EPISODES = 250                  # 총 에피소드 수
EPISODE_SECONDS = 15.0          # 에피소드 길이 (초)
TARGET_FORCE = 45.0             # 목표 힘 (N) - 고정값

# 통신
RECV_FREQ_HZ = 1000             # 수신 주파수 (1kHz)
PORT = 8888                     # TCP 포트
```

---

## 🔄 학습 프로세스

### 에피소드 실행 흐름

```
1. 에피소드 시작
   ├─ 상태 초기화 (이전 PID, 히스토리 활용)
   ├─ Actor가 PID 게인 결정
   └─ 로봇 제어 PC로 PID 게인 전송

2. 데이터 수집 (15초)
   ├─ 1000Hz로 힘 데이터 수신
   ├─ PI 출력 데이터 수집
   └─ 실시간 모니터링

3. 에피소드 종료
   ├─ 보상 계산 (18개 제어 지표)
   ├─ 성능 지표 분석
   ├─ Replay Buffer 저장
   └─ 신경망 업데이트 (16회)

4. 학습 진행
   ├─ 최고 성능 모델 저장
   ├─ 로그 기록 (CSV, PNG)
   └─ 다음 에피소드로
```

### 주요 데이터 흐름

```
강화학습 PC (Python)         로봇 제어 PC (C++)
      ↓                            ↓
[SAC Agent]                  [PID Controller]
      ↓                            ↓
  PID 게인 ─────전송(TCP)────→  힘 제어
      ↑                            ↓
힘 데이터 ←────수신(TCP)─────  센서 측정
      ↓
[Environment]
      ↓
  보상 계산
      ↓
[Replay Buffer]
      ↓
  신경망 학습
```

---

## 📊 제어 성능 지표 (18개)

### 1. 기본 제어 지표 (8개)
1. **RMSE** (Root Mean Square Error): 제어 정확도
2. **오버슈트** (%OS): 목표값 초과 정도
3. **정착 시간** (Ts): 목표 범위 도달 시간
4. **밴드 유지 시간**: ±0.5N 범위 내 시간
5. **밴드 이탈 시간**: ±0.5N 범위 밖 시간
6. **오차 분산**: 진동성/안정성 지표
7. **최대 힘**: 안전성 확인
8. **샘플 수**: 데이터 품질 확인

### 2. 정규화 지표 (7개)
9. **RMSE (정규화)**: 0~1 범위
10. **오버슈트 (정규화)**: 0~1 범위
11. **정착 시간 (정규화)**: 0~1 범위
12. **밴드 유지 비율**: 0~1 범위
13. **오차 분산 (정규화)**: 0~1 범위
14. **PI 출력 RMS (정규화)**: 제어 노력
15. **PI 출력 변화율 (정규화)**: 제어 부드러움

### 3. 제어 품질 지표 (3개)
16. **포화 비율**: PI 출력 포화 발생률
17. **이상 조건 개수**: 달성한 이상 조건 수
18. **이상 보너스**: 이상 성능 달성 보너스

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 필요 라이브러리 설치
pip install torch numpy matplotlib

# 프로젝트 디렉토리로 이동
cd /home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl/experiment
```

### 2. 학습 실행

```bash
# 메인 학습 스크립트 실행
python JY_PID_Gain_SAC_1_test.py
```

### 3. 학습 파라미터 수정

```python
# JY_PID_Gain_SAC_1_test.py 파일 내에서 수정

# 수신 주파수 변경 (line ~2754)
RECV_FREQUENCY_HZ = 1000  # 기본값: 1000Hz

# 에피소드 수 변경 (line ~96)
DEFAULT_EPISODES = 250  # 기본값: 250

# 에피소드 길이 변경 (line ~98)
DEFAULT_EPISODE_SECONDS = 15.0  # 기본값: 15초

# 목표 힘 변경 (line ~99)
DEFAULT_TARGET_FORCE = 45.0  # 기본값: 45N
```

### 4. 학습 중단

```bash
# Ctrl+C로 안전하게 중단
# 자동으로 데이터 저장 및 종료 신호 전송
```

---

## 📈 학습 결과 확인

### 1. 로그 파일 위치

```
experiment_logs/learning_done_YYMMDD_HHhMMm/
├── control_performance/
│   ├── performance_summary.csv       # 전체 성능 요약
│   ├── episode_*.csv                 # 개별 에피소드 상세 데이터
│   └── *.png                         # 성능 지표 그래프 (7개)
├── reward_breakdown/
│   ├── reward_breakdown.csv          # 보상 분해 데이터
│   └── reward_breakdown_*.png        # 보상 구성 그래프 (5개)
├── episode_rewards.csv                # 에피소드별 총보상
└── episode_rewards.png                # 보상 추이 그래프
```

### 2. 주요 분석 그래프

1. **Episode Rewards**: 학습 진행에 따른 보상 추이
2. **RMSE**: 제어 정확도 개선 추이
3. **Band Ratio**: 밴드 유지 성능 추이
4. **Overshoot**: 오버슈트 감소 추이
5. **Settling Time**: 정착 시간 단축 추이
6. **Control Effort**: 제어 노력 최적화 추이
7. **Saturation Ratio**: 포화 발생률 감소 추이

### 3. 최고 성능 모델

```
saved_agents/
└── test_best_agent_episode_X_reward_Y.pth
```

- `X`: 최고 성능 달성 에피소드 번호
- `Y`: 최고 보상값

---

## 🔧 향후 개선 방향

### 1. 알고리즘 개선
- [ ] **TD3 알고리즘 적용**: Deterministic policy로 제어 안정성 향상
- [ ] **PPO 알고리즘 비교**: On-policy 알고리즘 성능 비교
- [ ] **Domain Randomization**: 다양한 조건에서의 강건성 확보
- [ ] **Transfer Learning**: 다른 목표 힘으로의 전이 학습

### 2. 상태 공간 확장
- [ ] **속도 정보 추가**: 힘 변화 속도 직접 관측
- [ ] **가속도 정보 추가**: 진동 특성 더 정확히 파악
- [ ] **히스토리 윈도우 확장**: 시계열 패턴 학습
- [ ] **LSTM/GRU 적용**: 시계열 특성 명시적 모델링

### 3. 보상 함수 개선
- [ ] **적응형 가중치**: 학습 단계별 가중치 자동 조정
- [ ] **다목적 최적화**: Pareto-optimal 솔루션 탐색
- [ ] **안전 제약 강화**: Constrained RL 적용
- [ ] **실시간 피드백**: 스텝별 중간 보상 추가

### 4. 학습 효율성 향상
- [ ] **Curriculum Learning**: 쉬운 조건부터 점진적 학습
- [ ] **Prioritized Experience Replay**: 중요한 경험 우선 학습
- [ ] **Hindsight Experience Replay**: 실패 경험도 학습 자원으로 활용
- [ ] **분산 학습**: 여러 로봇 동시 학습으로 데이터 효율 증대

### 5. 실용화 개선
- [ ] **실시간 모니터링 대시보드**: 웹 기반 학습 진행 모니터링
- [ ] **자동 하이퍼파라미터 튜닝**: Optuna 등 활용
- [ ] **모델 압축**: 실시간 추론 속도 향상
- [ ] **A/B 테스트 프레임워크**: 기존 제어기와 성능 비교

### 6. 안전성 강화
- [ ] **Safe RL 적용**: 학습 중 안전 보장
- [ ] **이상 탐지**: 비정상 동작 실시간 감지
- [ ] **페일세이프 메커니즘**: 통신 두절 시 안전 모드
- [ ] **Sim-to-Real 검증**: 시뮬레이션 선행 학습 후 실제 적용

### 7. 데이터 분석 강화
- [ ] **Feature Importance 분석**: 상태 변수 중요도 분석
- [ ] **Ablation Study**: 보상 구성 요소별 기여도 분석
- [ ] **Sensitivity Analysis**: 하이퍼파라미터 민감도 분석
- [ ] **벤치마크 데이터셋**: 표준 성능 평가 기준 마련

---

## 🐛 알려진 이슈

### 1. 통신 안정성
- **문제**: 간헐적 통신 두절 발생
- **해결책**: 자동 재연결 및 타임아웃 처리 구현됨
- **개선 필요**: 더 강건한 에러 핸들링

### 2. 초기 탐색 불안정
- **문제**: 초기 에피소드에서 높은 분산
- **해결책**: 히스토리 기반 상태 초기화 적용
- **개선 필요**: 더 나은 초기화 전략

### 3. 메모리 사용량
- **문제**: 장시간 학습 시 메모리 증가
- **해결책**: 주기적 로그 플러시 및 버퍼 크기 제한
- **개선 필요**: 더 효율적인 메모리 관리

---

## 📚 참고 자료

### 논문
- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) - Haarnoja et al., 2018
- [PID Controller Tuning using RL](https://ieeexplore.ieee.org/) - 관련 연구들

### 코드 레퍼런스
- PyTorch 공식 문서: https://pytorch.org/docs/
- OpenAI Spinning Up: https://spinningup.openai.com/

---

## 👥 개발자 정보

- **프로젝트**: Robot Polishing RL System
- **목적**: 공압 연마 로봇의 지능형 PID 게인 최적화
- **환경**: Python 3.x, PyTorch, NumPy, Matplotlib
- **시스템**: Linux (Ubuntu), 1000Hz 실시간 통신

---

## 📝 변경 이력

### Version: JY_PID_Gain_SAC_1_test.py
- SAC 알고리즘 기반 PID 게인 최적화
- 18개 제어 성능 지표 통합
- 에피소드 기반 학습 (15초)
- 정규화된 보상 함수 적용
- 히스토리 기반 상태 표현
- 종합 로깅 시스템 (3개 Logger)
- 안전 메커니즘 강화
- 재현성 보장 (시드 고정)

---

## 🔐 라이센스

내부 연구용 프로젝트

---

## 📧 문의

프로젝트 관련 문의사항은 개발팀에 연락 바랍니다.

---

**Last Updated**: 2025-09-30
**Version**: 1.0
