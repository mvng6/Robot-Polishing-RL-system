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
- **통신 방식**: TCP/IP 소켓 통신 (1kHz 데이터 교환)

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
  - 에피소드당 1개 transition 저장 (Episodic RL)

#### 🌍 `PIDGainOptimizationEnvironment`
- **역할**: 강화학습 환경 관리 및 보상 계산
- **주요 기능**:
  - 에피소드 실행 및 데이터 수집 (1kHz)
  - 18개 제어공학 지표 기반 보상 계산
  - 학습 진행 모니터링
  - 최고 성능 모델 저장
- **핵심 메서드**:
  - `run_pid_optimization_training()`: 메인 학습 루프
  - `calculate_episode_reward()`: 보상 계산 (8개 정규화 지표)
  - 에피소드당 15초 동안 힘/PI 출력 데이터 수집

#### 📡 `RobotCommunication`
- **역할**: 로봇 제어 PC와 TCP/IP 통신
- **기능**:
  - **PID 게인 전송**: 
    - 첫 에피소드: 시작 시 전송
    - 이후 에피소드: 종료 시 다음 에피소드 PID 미리 전송
  - **상태 데이터 수신**: 1kHz로 힘, 에러, PI 출력 등 수신
  - **플래그 전송**:
    - `episode_done`: 에피소드 종료 시 True
    - `learning_done`: 전체 학습 완료 시 True
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
    # [0-5] 로봇 제어 PC에서 전송 (1kHz)
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

**첫 에피소드는 기준값 사용**: `Kp=80, Ki=130, Kd=0`

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

#### 보상 함수 특징
- **주파수 독립적**: 샘플링 주파수 자동 계산 (fs = n_samples / episode_len_s)
- **모든 지표 정규화**: [0,1] 범위로 통일하여 가중치 해석 용이
- **연속형 보상**: 불연속 보너스 최소화하여 학습 안정성 향상

---

### 5. 학습 파라미터

```python
# 신경망
STATE_DIM = 12
ACTION_DIM = 3
HIDDEN_DIM = 256
LEARNING_RATE = 3e-4

# 강화학습
GAMMA = 0.99                    # 할인 인수 (한 스텝 MDP이므로 영향 없음)
TAU = 0.01                      # 소프트 업데이트 계수
BATCH_SIZE = 128                # 배치 크기
REPLAY_BUFFER_SIZE = 4000       # 리플레이 버퍼 크기 (500 에피소드 × 8배)
UPDATES_PER_EPISODE = 16        # 에피소드당 업데이트 횟수 (동적 조정)

# 에피소드
EPISODES = 500                  # 총 에피소드 수
EPISODE_SECONDS = 15.0          # 에피소드 길이 (초)
TARGET_FORCE = 45.0             # 목표 힘 (N) - 고정값

# 통신
RECV_FREQ_HZ = 1000             # 수신 주파수 (1kHz)
HOST = "0.0.0.0"                # 서버 주소
PORT = 8888                     # TCP 포트
```

---

## 🔄 학습 프로세스

### 에피소드 실행 흐름

```
1. 에피소드 N 시작
   ├─ 상태 초기화 (이전 PID, 히스토리 활용)
   ├─ N=0: PID 전송 안 함 (로봇제어PC 자체 PID 80, 130, 0 사용)
   └─ N≥1: 이전 에피소드 종료 시 받은 PID 사용 (전송 안 함)

2. 데이터 수집 (15초) - 모든 에피소드 동일
   ├─ 1kHz로 힘 데이터 수신 및 저장
   ├─ 1kHz로 PI 출력 데이터 수집
   ├─ 총 약 15,000개 데이터 포인트 수집
   └─ ⭐ 에피소드 1도 동일하게 데이터 수집 (PID 전송만 안 함)

3. 에피소드 종료
   ├─ 15초 데이터로 보상 계산 (18개 제어 지표)
   ├─ 성능 지표 분석 및 로깅
   ├─ Replay Buffer에 1개 transition 저장
   ├─ 신경망 학습 시도:
   │   └─ Buffer < 2개: 학습 건너뜀 (에피소드 1)
   │   └─ Buffer ≥ 2개: 학습 수행 (에피소드 2부터, 동적 배치크기)
   ├─ 다음 에피소드용 PID 계산 (현재 Actor 네트워크 사용)
   │   └─ 에피소드 1 후: 초기 네트워크로 계산
   │   └─ 에피소드 2 후: 학습된 네트워크로 계산 (계속 개선됨)
   └─ 다음 에피소드용 PID + episode_done=True 전송

4. 로봇제어PC의 PID 관리
   ├─ 수신한 PID를 임시 변수에 보관 (m_nextKp, m_nextKi, m_nextKd)
   └─ 다음 에피소드 시작 시 현재 변수로 전환 후 컨트롤러 적용

5. 학습 진행
   ├─ 최고 성능 모델 자동 저장
   ├─ 로그 기록 (CSV, PNG)
   └─ 다음 에피소드로

6. 학습 완료
   ├─ 모든 에피소드 완료 시 learning_done=True 전송
   └─ 최종 데이터 저장 및 연결 종료
```

### 주요 데이터 흐름

```
강화학습 PC (Python)                    로봇 제어 PC (C++)
      ↓                                       ↓
[SAC Agent]                             [PID Controller]
      ↓                                       ↓
  에피소드 1 시작                         자체 PID (80,130,0) 사용
      ↓                                       ↓
힘/PI 출력 ←──────수신(TCP, 1kHz)──────  센서 측정 (15초)
      ↓                                       
  보상 계산                                 
      ↓                                       
  학습 (16회)                               
      ↓                                       
  다음 PID 계산                             
      ↓                                       ↑
  PID + episode_done=True ─────전송────→  다음 PID 저장
      ↓                                       ↓
  에피소드 2 시작                         저장된 PID 적용
      ↓                                       ↓
  (반복)                                  센서 측정 (15초)
      ↓
  ...
      ↓
  모든 에피소드 완료
      ↓
  learning_done=True ──────전송─────→  제어 종료
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
12. **밴드 유지 비율**: 0~1 범위 (핵심 지표)
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
# JY_PID_Gain_SAC_1_test.py 파일 내 Constants 클래스에서 수정

# 에피소드 수 변경
DEFAULT_EPISODES = 500  # 기본값: 500

# 에피소드 길이 변경
DEFAULT_EPISODE_SECONDS = 15.0  # 기본값: 15초

# 목표 힘 변경
DEFAULT_TARGET_FORCE = 45.0  # 기본값: 45N

# 밴드 허용 오차 변경
BAND_TOLERANCE_N = 0.5  # 기본값: ±0.5N

# 보상 가중치 조정
REWARD_WEIGHT_BAND = 1.5       # 밴드 유지
REWARD_WEIGHT_RMSE = 1.2       # RMSE
REWARD_WEIGHT_OVERSHOOT = 0.7  # 오버슈트
# ... (기타 가중치)
```

### 4. 학습 중단

```bash
# Ctrl+C로 안전하게 중단
# 자동으로:
# - learning_done=True 신호 전송
# - 데이터 저장 (CSV, PNG)
# - 최종 모델 저장
# - 연결 종료
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

## 🤖 로봇제어PC 구현 가이드

### 필수 구현 사항

강화학습 PC와 통신하기 위해 로봇제어PC(C++)에서 구현해야 할 핵심 기능들입니다.

#### 1. PID 게인 패킷 수신 구조체

```cpp
// Protocol.h에 추가
#pragma pack(push, 1)
struct PIDGainPacket {
    unsigned short  sof;              // 0xCCCC (2 bytes)
    float           Kp;               // P 게인 (4 bytes)
    float           Ki;               // I 게인 (4 bytes)
    float           Kd;               // D 게인 (4 bytes)
    bool            timing_accurate;  // 타이밍 정확성 (1 byte)
    bool            episode_done;     // 에피소드 종료 플래그 (1 byte)
    bool            learning_done;    // 학습 종료 플래그 (1 byte)
    unsigned short  checksum;         // CRC-16 체크섬 (2 bytes)
};
#pragma pack(pop)

bool UnpackPIDGainPacket(const char* buffer, int length, PIDGainPacket& outPacket);
```

#### 2. PID 게인 관리 클래스

```cpp
class PIDGainManager {
private:
    // 기본 PID (첫 에피소드용 - 하드코딩, 변수로 변경 가능)
    const float DEFAULT_KP = 80.0f;
    const float DEFAULT_KI = 130.0f;
    const float DEFAULT_KD = 0.0f;
    
    float m_currentKp, m_currentKi, m_currentKd;  // 현재 에피소드에서 사용 중인 PID
    float m_nextKp, m_nextKi, m_nextKd;           // 다음 에피소드에서 사용할 PID (미리 받아둠)
    bool m_hasNextGain;                           // 다음 PID를 받았는지 여부
    bool m_episodeActive;
    bool m_learningActive;
    std::mutex m_gainMutex;

public:
    PIDGainManager() 
        : m_currentKp(DEFAULT_KP), m_currentKi(DEFAULT_KI), m_currentKd(DEFAULT_KD),
          m_hasNextGain(false), m_episodeActive(false), m_learningActive(true) {}
    
    // 강화학습PC로부터 PID 게인 수신 (에피소드 종료 시)
    // → m_nextKp, m_nextKi, m_nextKd에 보관
    void ReceivePIDGain(const PIDGainPacket& packet);
    
    // 에피소드 시작 시 호출: m_next → m_current 전환
    void StartEpisode();
    
    // 기본 PID 사용 (첫 에피소드용)
    void UseDefaultGains();
    
    // 현재 사용 중인 PID 가져오기
    void GetCurrentGains(float& kp, float& ki, float& kd);
    
    // 다음 PID를 받았는지 확인
    bool HasNextGain() const;
    
    // 학습 종료 확인
    bool IsLearningDone() const;
};
```

#### 3. TCP 수신 처리

```cpp
// TcpClient의 수신 스레드에서
if (packet_sof == 0xCCCC) {  // PID 게인 패킷
    PIDGainPacket pidPacket;
    if (UnpackPIDGainPacket(buffer, length, pidPacket)) {
        m_pidGainManager.ReceivePIDGain(pidPacket);
    }
}
```

#### 4. 메인 제어 루프

```cpp
// 에피소드 관리
bool episode_active = false;
double episode_start_time = 0.0;
const double EPISODE_DURATION = 15.0;  // 15초
int current_episode = 0;

// 메인 루프 (1kHz)
while (control_active) {
    // 에피소드 시작 감지
    if (!episode_active) {
        current_episode++;
        
        if (current_episode == 1) {
            // 첫 에피소드: 하드코딩된 기본 PID 사용
            m_pidGainManager.UseDefaultGains();
            printf("에피소드 1: 기본 PID (80, 130, 0) 사용\n");
        } 
        else if (m_pidGainManager.HasNextGain()) {
            // 2번째 에피소드부터: 보관된 PID를 현재로 전환
            m_pidGainManager.StartEpisode();  // m_current = m_next
            printf("에피소드 %d: 보관된 PID를 현재 PID로 전환\n", current_episode);
        }
        
        // PID 컨트롤러에 적용
        float kp, ki, kd;
        m_pidGainManager.GetCurrentGains(kp, ki, kd);
        m_pidctrl.setGains(kp, ki, kd);
        printf("PID 적용: Kp=%.2f, Ki=%.2f, Kd=%.2f\n", kp, ki, kd);
        
        episode_active = true;
        episode_start_time = GetCurrentTime();
    }
    
    // 센서 데이터 읽기
    float currentForce = ReadForceZ();
    float error = target_force - currentForce;
    
    // PID 제어
    float pid_output = m_pidctrl.compute(error, dt);
    ApplyControl(pid_output);
    
    // 상태 패킷 전송 (1kHz)
    std::vector<char> packet = PackRobotStatus(
        currentForce, target_force, error, 
        error_dot, error_integral, pid_output, sander_active
    );
    m_tcpClient.Send(packet.data(), packet.size());
    
    // 에피소드 종료 감지 (15초)
    if (episode_active && 
        (GetCurrentTime() - episode_start_time) >= EPISODE_DURATION) {
        episode_active = false;
    }
    
    // 학습 종료 확인
    if (m_pidGainManager.IsLearningDone()) {
        StopControl();
        break;
    }
    
    Sleep(1);  // 1ms
}
```

#### 5. PIDGainManager 주요 메서드 구현 예시

```cpp
void PIDGainManager::ReceivePIDGain(const PIDGainPacket& packet) {
    std::lock_guard<std::mutex> lock(m_gainMutex);
    
    // 다음 에피소드용 PID를 임시 변수에 보관
    m_nextKp = packet.Kp;
    m_nextKi = packet.Ki;
    m_nextKd = packet.Kd;
    m_hasNextGain = true;
    
    // 에피소드 종료 플래그 확인
    if (packet.episode_done) {
        printf("에피소드 종료 신호 수신 + 다음 PID 보관: Kp=%.2f, Ki=%.2f, Kd=%.2f\n",
               m_nextKp, m_nextKi, m_nextKd);
    }
    
    // 학습 종료 플래그 확인
    if (packet.learning_done) {
        m_learningActive = false;
        printf("학습 완료 신호 수신! 제어 종료 준비\n");
    }
}

void PIDGainManager::StartEpisode() {
    std::lock_guard<std::mutex> lock(m_gainMutex);
    
    if (m_hasNextGain) {
        // 보관된 PID를 현재 PID로 전환
        m_currentKp = m_nextKp;
        m_currentKi = m_nextKi;
        m_currentKd = m_nextKd;
        m_hasNextGain = false;
        
        printf("보관된 PID를 현재로 전환: Kp=%.2f, Ki=%.2f, Kd=%.2f\n",
               m_currentKp, m_currentKi, m_currentKd);
    }
}

void PIDGainManager::UseDefaultGains() {
    std::lock_guard<std::mutex> lock(m_gainMutex);
    
    // 기본 PID 사용 (첫 에피소드)
    m_currentKp = DEFAULT_KP;
    m_currentKi = DEFAULT_KI;
    m_currentKd = DEFAULT_KD;
    
    printf("기본 PID 사용: Kp=%.2f, Ki=%.2f, Kd=%.2f\n",
           m_currentKp, m_currentKi, m_currentKd);
}

void PIDGainManager::GetCurrentGains(float& kp, float& ki, float& kd) {
    std::lock_guard<std::mutex> lock(m_gainMutex);
    kp = m_currentKp;
    ki = m_currentKi;
    kd = m_currentKd;
}

bool PIDGainManager::HasNextGain() const {
    return m_hasNextGain;
}

bool PIDGainManager::IsLearningDone() const {
    return !m_learningActive;
}
```

#### 6. 상태 패킷 전송 (1kHz)

```cpp
// 이미 구현된 PythonCommPacket 사용
std::vector<char> packet = PackRobotStatus(
    current_forceZ,       // 현재 힘
    target_forceZ,        // 목표 힘
    error_forceZ,         // 힘 오차
    error_forceZ_dot,     // 힘 오차 미분
    error_forceZ_int,     // 힘 오차 적분
    cur_PID_output,       // PI 출력
    sander_active_flag    // 샌더 활성 상태
);
m_tcpClient.Send(packet.data(), packet.size());
```

### 구현 체크리스트

#### A. 데이터 구조
- [ ] `Protocol.h`에 `PIDGainPacket` 구조체 추가
- [ ] `UnpackPIDGainPacket()` 함수 구현
- [ ] CRC-16 체크섬 검증

#### B. PID 관리
- [ ] `PIDGainManager` 클래스 구현
  - [ ] 기본 PID 하드코딩 (80, 130, 0)
  - [ ] 현재 PID 변수 (`m_currentKp`, `m_currentKi`, `m_currentKd`)
  - [ ] 다음 PID 변수 (`m_nextKp`, `m_nextKi`, `m_nextKd`)
  - [ ] `ReceivePIDGain()`: 수신한 PID를 m_next*에 보관
  - [ ] `StartEpisode()`: m_next를 m_current로 전환
  - [ ] `UseDefaultGains()`: 기본 PID 사용 (첫 에피소드)
  - [ ] 스레드 안전성 (mutex)

#### C. TCP 통신
- [ ] TCP 수신 처리에 PID 게인 패킷 핸들링 추가 (SOF=0xCCCC)
- [ ] 상태 패킷 전송 (1kHz, SOF=0xAAAA)

#### D. 메인 제어 루프
- [ ] 에피소드 카운터 추가
- [ ] 첫 에피소드: `UseDefaultGains()` 호출
- [ ] 2번째 이후: `StartEpisode()` 호출 (m_next → m_current)
- [ ] PID 컨트롤러 적용: `setGains()`
- [ ] 15초 타이머 구현
- [ ] learning_done 플래그 처리
- [ ] 로깅 및 디버깅 메시지

### 통신 프로토콜 요약

| 방향 | 패킷 종류 | SOF | 주파수 | 내용 |
|------|----------|-----|--------|------|
| RL→Robot | PID Gain | 0xCCCC | 에피소드 종료 시 (첫 에피소드 제외) | Kp, Ki, Kd, episode_done, learning_done |
| Robot→RL | State | 0xAAAA | 1kHz | Force, Error, PI output, Sander active |

**중요**: 첫 에피소드는 로봇제어PC가 하드코딩된 기본 PID를 사용하므로 강화학습PC에서 PID를 전송하지 않습니다.

### 에피소드 타이밍 다이어그램

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
에피소드 1 (기준 성능 측정)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[시작]
  [Robot] 변수 상태: m_currentKp=80, m_currentKi=130, m_currentKd=0 (하드코딩 초기값)
  [Robot] PID 컨트롤러 적용: setGains(80, 130, 0)
  [RL]    PID 전송 안 함 (로봇제어PC 자체 값 사용)
  
[실행 - 15초]
  [Robot] 매 1ms마다: 힘 센서 읽기 → PID 제어 계산 → 상태 패킷 전송 (1kHz)
  [RL]    매 1ms마다: 상태 패킷 수신 → 힘/PI출력 데이터 저장
  [RL]    총 15,000개 데이터 포인트 수집 (힘, 에러, PI출력 등)
  
[종료]
  [RL] 15,000개 데이터 분석: RMSE, 오버슈트, 정착시간 등 18개 지표 계산
  [RL] 보상 계산: 기준 성능 평가
  [RL] Replay Buffer 저장: transition 1개 추가 (총 1개)
  [RL] 통계 업데이트: 보상 기록, 성능 지표 로깅
  [RL] 신경망 학습 시도: 건너뜀 (replay buffer=1개 < 최소 2개)
  [RL] 에피소드 2용 PID 계산: 초기 Actor 네트워크로 계산 (학습 전 상태)
  [RL] 패킷 전송: Kp=85.3, Ki=127.8, Kd=2.1, episode_done=True
  [Robot] 패킷 수신 → m_nextKp=85.3, m_nextKi=127.8, m_nextKd=2.1로 보관
  
  ⚠️ 에피소드 2는 초기 네트워크의 PID 사용 (학습 전)
  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
에피소드 2 (강화학습 시작)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[시작]
  [Robot] m_currentKp = m_nextKp (85.3으로 전환)
  [Robot] m_currentKi = m_nextKi (127.8로 전환)
  [Robot] m_currentKd = m_nextKd (2.1로 전환)
  [Robot] PID 컨트롤러 적용: setGains(85.3, 127.8, 2.1)
  [RL]    PID 전송 안 함 (이미 에피소드 1 종료 시 전송했음)
  
[실행 - 15초]
  [Robot] 매 1ms마다: 힘 센서 읽기 → PID 제어 계산 → 상태 패킷 전송 (1kHz)
  [RL]    매 1ms마다: 상태 패킷 수신 → 힘/PI출력 데이터 저장
  [RL]    총 15,000개 데이터 포인트 수집
  
[종료]
  [RL] 15,000개 데이터 분석: 18개 제어 지표 계산
  [RL] 보상 계산
  [RL] Replay Buffer 저장: transition 1개 추가 (총 2개)
  [RL] 통계 업데이트: 보상 기록, 성능 지표 로깅
  [RL] 신경망 학습: ✅ 시작! (replay buffer=2개, 배치크기 2, 1회 업데이트)
  [RL] 에피소드 3용 PID 계산: ✅ 학습된 네트워크로 계산! (학습 후)
  [RL] 패킷 전송: Kp=82.1, Ki=131.5, Kd=1.8, episode_done=True
  [Robot] 패킷 수신 → m_nextKp=82.1, m_nextKi=131.5, m_nextKd=1.8로 보관
  
  ✅ 에피소드 3부터 학습된 네트워크의 PID 사용!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
에피소드 3~500 (반복)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
마지막 에피소드 종료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [RL] 패킷 전송: Kp=0, Ki=0, Kd=0, learning_done=True
  [Robot] 학습 완료 플래그 확인 → 제어 루프 종료
```

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
- [ ] **ITAE 추가**: 시간 가중 절대 오차 적분 (선택적)

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
- [Soft Actor-Critic Algorithms](https://arxiv.org/abs/1812.05905) - Haarnoja et al., 2019
- PID Controller Tuning using RL - 관련 연구들

### 코드 레퍼런스
- PyTorch 공식 문서: https://pytorch.org/docs/
- OpenAI Spinning Up: https://spinningup.openai.com/
- Stable Baselines3: https://stable-baselines3.readthedocs.io/

---

## 👥 개발자 정보

- **프로젝트**: Robot Polishing RL System - PID Gain Optimization
- **목적**: 공압 연마 로봇의 지능형 PID 게인 자동 최적화
- **환경**: Python 3.x, PyTorch, NumPy, Matplotlib
- **시스템**: Linux (Ubuntu), 1kHz 실시간 통신
- **알고리즘**: Soft Actor-Critic (SAC)

---

## 📝 변경 이력

### Version: JY_PID_Gain_SAC_1_test.py (2025-09-30)
- ✅ SAC 알고리즘 기반 PID 게인 최적화
- ✅ 18개 제어 성능 지표 통합
- ✅ 에피소드 기반 학습 (15초, 1kHz 수집)
- ✅ 정규화된 보상 함수 적용
- ✅ 주파수 독립적 보상 계산
- ✅ 히스토리 기반 상태 표현 (최근 5개 에피소드)
- ✅ 종합 로깅 시스템 (3개 Logger)
- ✅ 에피소드 종료 신호 전송 (episode_done)
- ✅ 학습 완료 신호 전송 (learning_done)
- ✅ Ctrl+C 안전 종료 메커니즘
- ✅ 재현성 보장 (시드 고정: 42)
- ✅ 로봇제어PC 통신 프로토콜 정의

---

## 🔐 라이센스

내부 연구용 프로젝트

---

**Last Updated**: 2025-09-30  
**Version**: 1.0  
**Status**: Production Ready ✅