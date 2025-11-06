# 📋 코드 기능 상세 분석 - Part 1: 실행 및 설정 모듈

## 📁 모듈 목록

1. `__main__.py` - 모듈 실행 엔트리 포인트
2. `main.py` - 명령줄 인터페이스
3. `config.py` - 설정 관리
4. `constants.py` - 상수 정의

---

## 1. `__main__.py` - 모듈 실행 엔트리 포인트

**파일 경로**: `py_rl/pid_gain_rl/__main__.py`  
**줄 수**: 115줄  
**역할**: Python 모듈로 실행할 때의 엔트리 포인트

### 주요 기능

#### 1.1 `main()` 함수
**역할**: 프로그램 메인 실행 함수

**세부 기능**:

1. **설정 변경 포인트**
   ```python
   RECV_FREQUENCY_HZ = 1000           # 수신 주파수 (Hz)
   EPISODE_LENGTH_SECONDS = 10.0      # 에피소드 길이 (초)
   ```
   - 사용자가 직접 수정 가능한 설정값
   - 기본값 사용, 간단한 실행에 적합

2. **설정 생성**
   ```python
   config = create_config(RECV_FREQUENCY_HZ, EPISODE_LENGTH_SECONDS)
   config_dict = config.to_dict()
   ```
   - `create_config()` 함수로 설정 객체 생성
   - 딕셔너리로 변환 (호환성 유지)

3. **프로그램 정보 출력**
   - 프로그램 버전 정보
   - 수신 주파수, 목표 힘, 에피소드 길이 출력
   - 구분선 출력

4. **재현성 시드 설정**
   ```python
   np.random.seed(42)
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```
   - NumPy, PyTorch, CUDA 시드 고정
   - CUDNN 결정론적 모드 활성화
   - 재현 가능한 결과 보장

5. **환경 생성**
   ```python
   env = PIDGainEnvironment(config_dict)
   _global_env = env
   ```
   - 환경 객체 생성
   - 전역 변수에 저장 (시그널 핸들러에서 사용)

6. **시그널 핸들러 설치**
   ```python
   install_signal_handlers()
   ```
   - SIGINT, SIGTERM 핸들러 설치
   - Ctrl+C 안전 종료 지원

7. **학습 루프 실행**
   ```python
   env.run_pid_optimization_training(config.episodes)
   ```
   - 메인 학습 루프 실행

8. **예외 처리**
   - **KeyboardInterrupt (Ctrl+C)**:
     - 학습 중단 신호 전송 (`learning_done=True`)
     - 데이터 저장 (DataSaver.save_all_data)
     - 통신 종료
   - **일반 Exception**:
     - 오류 로그 출력
     - 학습 오류 종료 신호 전송
     - 데이터 저장 및 통신 종료

9. **데이터 저장**
   ```python
   env.rlogger.flush_if_needed(..., force=True, ...)
   DataSaver.save_all_data(env)
   ```
   - 보상 분석 데이터 저장
   - 제어 성능 지표 저장

### 전역 변수

- `_global_env`: 전역 환경 객체 (시그널 핸들러에서 접근)

### 실행 방법

```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl
python3 -m pid_gain_rl
```

---

## 2. `main.py` - 명령줄 인터페이스

**파일 경로**: `py_rl/pid_gain_rl/main.py`  
**줄 수**: 230줄  
**역할**: argparse 기반 명령줄 인터페이스로 유연한 설정 지원

### 주요 기능

#### 2.1 `main()` 함수
**역할**: 명령줄 인자 파싱 및 실행

**세부 기능**:

1. **argparse 인자 파싱**
   - `--episodes`: 학습할 에피소드 수 (기본: 500)
   - `--batch-size`: 배치 크기 (기본: 64)
   - `--lr`: 기본 학습률 (기본: 1e-4)
   - `--lr-actor`: Actor 학습률 (기본: 1e-4)
   - `--lr-critic`: Critic 학습률 (기본: 2e-4)
   - `--target-force`: 목표 힘 (기본: -40.0)
   - `--episode-seconds`: 에피소드 길이 (기본: 10.0)
   - `--recv-freq`: 수신 주파수 (기본: 1000)
   - `--load-model`: 로드할 모델 경로 (.pth 파일)
   - `--log-dir`: 로그 디렉토리
   - `--model-dir`: 모델 저장 디렉토리

2. **설정 생성 및 덮어쓰기**
   ```python
   config = create_config(recv_freq_hz=args.recv_freq, episode_seconds=args.episode_seconds)
   config.target_force = args.target_force
   config.episodes = args.episodes
   # ... 기타 설정 덮어쓰기
   ```
   - 기본 설정 생성
   - 명령줄 인자로 덮어쓰기

3. **시작 정보 출력**
   - 상세한 설정 정보 출력
   - STATE_DIM, 세그먼트 분할, 탐험 설정 등 출력

4. **재현성 시드 설정**
   - `__main__.py`와 동일한 시드 설정

5. **디렉토리 생성**
   ```python
   os.makedirs(args.log_dir, exist_ok=True)
   os.makedirs(args.model_dir, exist_ok=True)
   ```

6. **환경 생성 및 시그널 핸들러 설치**
   - `__main__.py`와 동일

7. **모델 로드 (선택적)**
   ```python
   if args.load_model:
       if env.agent.load_model(args.load_model):
           print("✅ 모델 로드 완료")
       else:
           print("⚠️ 모델 로드 실패 - 새로 학습 시작")
   ```
   - 전이학습 또는 학습 재개 지원

8. **학습 루프 실행 및 예외 처리**
   - `__main__.py`와 동일한 로직

### 차이점: `__main__.py` vs `main.py`

| 항목 | `__main__.py` | `main.py` |
|------|--------------|-----------|
| **설정 방식** | 코드 내 하드코딩 | argparse 명령줄 인자 |
| **실행 방법** | `python3 -m pid_gain_rl` | `python3 main.py [옵션]` |
| **설정 변경** | 코드 수정 필요 | 실행 시 옵션으로 변경 |
| **모델 로드** | 지원 안 함 | `--load-model` 옵션 지원 |
| **용도** | 간단한 실행, 기본 설정 | 다양한 실험, 유연한 설정 |

### 실행 방법

```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl
python3 main.py --episodes 500 --target-force -40.0 --batch-size 64
```

---

## 3. `config.py` - 설정 관리

**파일 경로**: `py_rl/pid_gain_rl/config.py`  
**줄 수**: 171줄  
**역할**: 타입 안전한 설정 관리 (dataclass 사용)

### 주요 클래스

#### 3.1 `Config` 클래스 (dataclass)

**역할**: 모든 설정을 담는 데이터클래스

**필드 구성**:

1. **신경망 설정**
   - `state_dim`: 상태 차원 (기본: 12, 실제로는 20차원으로 동적 설정됨)
   - `action_dim`: 액션 차원 (기본: 3)
   - `hidden_dim`: 은닉층 크기 (기본: 128)
   - `lr`: 기본 학습률 (기본: 1e-4)
   - `lr_actor`: Actor 학습률 (기본: 1e-4)
   - `lr_critic`: Critic 학습률 (기본: 2e-4)
   - `gamma`: 할인율 (기본: 0.99)
   - `tau`: Soft update 계수 (기본: 0.01)
   - `auto_entropy`: Entropy 자동 튜닝 여부 (기본: True)

2. **PID 설정**
   - `pid_range`: PID 범위 딕셔너리
     - `Kp`: (3.0, 40.0)
     - `Ki`: (0.0, 40.0)
     - `Kd`: (1e-4, 5e-2)

3. **에피소드 설정**
   - `episode_seconds`: 에피소드 길이 (기본: 10.0초)
   - `target_force`: 목표 힘 (기본: -40.0N)
   - `updates_per_episode`: 에피소드당 업데이트 횟수 (기본: 35)
   - `episodes`: 총 에피소드 수 (기본: 500)

4. **통신 설정**
   - `recv_freq_hz`: 수신 주파수 (기본: 1000 Hz)
   - `recv_interval_sec`: 수신 간격 (계산됨: 1.0 / recv_freq_hz)
   - `batch_size`: 배치 크기 (기본: 64)
   - `host`: 호스트 주소 (기본: "0.0.0.0")
   - `port`: 포트 번호 (기본: 8888)
   - `recv_timeout_sec`: 수신 타임아웃 (기본: 0.5초)
   - `recv_loop_timeout_sec`: 수신 루프 타임아웃 (기본: 0.05초)

5. **실패 처리 설정**
   - `comm_fail_max`: 통신 실패 최대 횟수 (기본: 3)
   - `comm_retry_delay`: 재시도 지연 (기본: 0.1초)

6. **저장 설정**
   - `model_save_dir`: 모델 저장 디렉토리
   - `log_dir`: 로그 디렉토리

7. **학습 설정**
   - `max_episode_rewards_history`: 최대 보상 히스토리 (기본: 1000)
   - `replay_buffer_size`: 리플레이 버퍼 크기 (기본: 10000)

**메서드**:

1. **`__post_init__()`**
   - 초기화 후 처리
   - `recv_interval_sec` 계산: `1.0 / recv_freq_hz`

2. **`to_dict()` → dict**
   - 딕셔너리로 변환 (호환성 유지)
   - 기존 코드와의 호환을 위해 대문자 키 사용
   - 예: `STATE_DIM`, `ACTION_DIM`, `HIDDEN` 등

3. **`from_dict(cls, config_dict: dict) → Config`**
   - 딕셔너리로부터 Config 객체 생성
   - 기본값 처리 포함
   - 전이학습 시 기존 설정 로드에 사용

### 주요 함수

#### 3.2 `create_config(recv_freq_hz=None, episode_seconds=None) → Config`

**역할**: 기본 설정 생성 함수

**매개변수**:
- `recv_freq_hz`: 수신 주파수 (선택적, None이면 기본값 사용)
- `episode_seconds`: 에피소드 길이 (선택적, None이면 기본값 사용)

**검증**:
- `recv_freq_hz`: 0 < recv_freq_hz <= 10000
- `episode_seconds`: episode_seconds > 0

**반환값**: Config 객체

**예외**: ValueError (잘못된 값일 때)

#### 3.3 `change_episode_length(config: Config, new_length_seconds: float) → Config`

**역할**: 에피소드 길이를 동적으로 변경

**매개변수**:
- `config`: 현재 설정
- `new_length_seconds`: 새로운 에피소드 길이 (초)

**검증**:
- `new_length_seconds` > 0

**반환값**: 업데이트된 Config 객체

**출력**: 변경된 에피소드 길이 및 목표 데이터 개수 출력

---

## 4. `constants.py` - 상수 정의

**파일 경로**: `py_rl/pid_gain_rl/constants.py`  
**줄 수**: 147줄  
**역할**: 모든 하드코딩된 값 중앙 집중 관리

### 클래스: `Constants`

**역할**: 모든 상수 정의

### 상수 분류

#### 4.1 신경망 기본 설정

```python
DEFAULT_HIDDEN_DIM = 128
DEFAULT_LR = 1e-4
DEFAULT_LR_ACTOR = 1e-4
DEFAULT_LR_CRITIC = 2e-4
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.01
```

**설명**:
- `DEFAULT_HIDDEN_DIM`: 은닉층 크기 (128)
- `DEFAULT_LR`: 기본 학습률 (1e-4)
- `DEFAULT_LR_ACTOR`: Actor 학습률 (1e-4)
- `DEFAULT_LR_CRITIC`: Critic 학습률 (2e-4, Actor보다 2배)
- `DEFAULT_GAMMA`: 할인율 (0.99)
- `DEFAULT_TAU`: Soft update 계수 (0.01)

#### 4.2 PID 범위

```python
DEFAULT_PID_RANGE = {
    "Kp": (3.0, 40.0),      # 하한 확장: 35 → 3 (오버슈트 억제)
    "Ki": (0.0, 40.0),      # 하한 0까지, 상한 낮춤: 55 → 40 (조건부 적분 off 가능)
    "Kd": (1e-4, 5e-2),     # 대폭 확장: 1e-6~1e-3 → 1e-4~5e-2 (댐핑 효과)
}
```

**설명**:
- **Kp**: 비례 게인 범위 (3.0 ~ 40.0)
  - 하한 35 → 3으로 확장 (오버슈트 억제)
- **Ki**: 적분 게인 범위 (0.0 ~ 40.0)
  - 하한 0 포함 (조건부 적분 off 가능)
  - 상한 55 → 40으로 낮춤 (과도한 적분 누적 방지)
- **Kd**: 미분 게인 범위 (1e-4 ~ 5e-2)
  - 대폭 확장 (1e-6~1e-3 → 1e-4~5e-2)
  - 댐핑 효과 확보
  - 로그 스케일로 매핑됨

#### 4.3 통신 기본 설정

```python
DEFAULT_RECV_FREQ = 1000
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPISODES = 500
MIN_EPISODES_FOR_SAVING = 50
DEFAULT_EPISODE_SECONDS = 10.0
DEFAULT_TARGET_FORCE = -40.0
DEFAULT_UPDATES_PER_EPISODE = 35
```

**설명**:
- `DEFAULT_RECV_FREQ`: 수신 주파수 (1kHz)
- `DEFAULT_BATCH_SIZE`: 배치 크기 (64)
- `DEFAULT_EPISODES`: 기본 에피소드 수 (500)
- `MIN_EPISODES_FOR_SAVING`: 모델 저장 시작 에피소드 (50, 초반 lucky reward 방지)
- `DEFAULT_EPISODE_SECONDS`: 에피소드 길이 (10.0초)
- `DEFAULT_TARGET_FORCE`: 목표 힘 (-40.0N)
- `DEFAULT_UPDATES_PER_EPISODE`: 에피소드당 업데이트 횟수 (35, 세그먼트 분할 대응)

#### 4.4 통신 설정

```python
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888
DEFAULT_RECV_TIMEOUT = 0.5
DEFAULT_RECV_LOOP_TIMEOUT = 0.05
DEFAULT_COMM_FAIL_MAX = 3
DEFAULT_COMM_RETRY_DELAY = 0.1
```

**설명**:
- `DEFAULT_HOST`: 호스트 주소 (0.0.0.0, 모든 인터페이스)
- `DEFAULT_PORT`: 포트 번호 (8888)
- `DEFAULT_RECV_TIMEOUT`: 수신 타임아웃 (0.5초)
- `DEFAULT_RECV_LOOP_TIMEOUT`: 수신 루프 타임아웃 (0.05초)
- `DEFAULT_COMM_FAIL_MAX`: 통신 실패 최대 횟수 (3)
- `DEFAULT_COMM_RETRY_DELAY`: 재시도 지연 (0.1초)

#### 4.5 학습 설정

```python
DEFAULT_MAX_REWARDS_HISTORY = 1000
DEFAULT_REPLAY_BUFFER_SIZE = 10000
MIN_BUFFER_FOR_LEARNING = 32
MIN_BATCH_SIZE = 32
```

**설명**:
- `DEFAULT_MAX_REWARDS_HISTORY`: 최대 보상 히스토리 (1000개)
- `DEFAULT_REPLAY_BUFFER_SIZE`: 리플레이 버퍼 크기 (10000개, 세그먼트 분할 대응)
- `MIN_BUFFER_FOR_LEARNING`: 학습 시작 최소 버퍼 크기 (32개)
- `MIN_BATCH_SIZE`: 최소 배치 크기 (32개)

#### 4.6 저장 경로

```python
DEFAULT_MODEL_SAVE_DIR = "/home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl/saved_agents"
DEFAULT_LOG_DIR = "/home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl/experiment_logs"
```

**설명**:
- `DEFAULT_MODEL_SAVE_DIR`: 모델 저장 디렉토리
- `DEFAULT_LOG_DIR`: 로그 디렉토리

#### 4.7 세그먼트 분할 설정

```python
NUM_SEGMENTS = 5
SEGMENT_LENGTH_S = 2.0
```

**설명**:
- `NUM_SEGMENTS`: 에피소드를 나눌 세그먼트 수 (5개)
- `SEGMENT_LENGTH_S`: 각 세그먼트 길이 (2.0초)
- 10초 에피소드 → 5개 세그먼트 (2초씩)

#### 4.8 Warm-start 설정

```python
WARM_START_ENABLED = True
WARM_START_NUM_SAMPLES = 50
```

**설명**:
- `WARM_START_ENABLED`: Warm-start 활성화 여부 (True)
- `WARM_START_NUM_SAMPLES`: LHS 샘플 수 (50개, 30~50 범위)

#### 4.9 세그먼트용 보상 파라미터 조정

```python
SHAPING_WARMUP_S = 0.2  # 0.5 → 0.2 (세그먼트용)
SETTLING_HOLD_TIME_S = 0.5  # 1.0 → 0.5 (세그먼트용)
```

**설명**:
- `SHAPING_WARMUP_S`: PBRS warmup 시간 (0.2초, 세그먼트용으로 축소)
- `SETTLING_HOLD_TIME_S`: 정착 시간 hold 시간 (0.5초, 세그먼트용으로 축소)

#### 4.10 제어 물리 상수

```python
BAND_TOLERANCE_N = 1.5
SETTLING_BAND_TOLERANCE = 1.0
SAFETY_FORCE_LIMIT = 100.0
SAFETY_FORCE_PENALTY = -1.0
PI_OUTPUT_MAX = 0.4
PI_OUTPUT_SAT_THRESHOLD = 0.95
```

**설명**:
- `BAND_TOLERANCE_N`: 밴드 허용 오차 (±1.5N)
- `SETTLING_BAND_TOLERANCE`: 정착 시간 밴드 허용 오차 (±1.0N)
- `SAFETY_FORCE_LIMIT`: 안전 힘 한계 (±100N)
- `SAFETY_FORCE_PENALTY`: 안전 위반 패널티 (-1.0)
- `PI_OUTPUT_MAX`: PI 출력 최대값 (0.4)
- `PI_OUTPUT_SAT_THRESHOLD`: PI 출력 포화 임계값 (0.95)

#### 4.11 스코어화 기반 보상 시스템

```python
SCORE_TAU_TS = 5.0
SCORE_TAU_MP_PERCENT = 8.0  # 12.0 → 8.0 (오버슈트 더 민감하게)
SCORE_TAU_ESS_N = 1.0
SCORE_TAU_U = 0.5

SCORE_W_TS   = 0.30
SCORE_W_MP   = 0.35  # 0.25 → 0.35 (오버슈트 가중치 강화)
SCORE_W_ESS  = 0.20
SCORE_W_BAND = 0.15
SCORE_W_U    = 0.05
SCORE_W_FAIL = 0.15
SCORE_W_PBRS = 0.10
```

**설명**:
- **TAU 값들**: 지수 스코어 변환 시 시상수
  - `SCORE_TAU_TS`: 정착시간 시상수 (5.0초)
  - `SCORE_TAU_MP_PERCENT`: 오버슈트 시상수 (8.0%, 더 민감하게)
  - `SCORE_TAU_ESS_N`: 정상상태 오차 시상수 (1.0N)
  - `SCORE_TAU_U`: 제어 노력 시상수 (0.5)
- **W 값들**: 보상 가중치
  - `SCORE_W_TS`: 정착시간 가중치 (0.30)
  - `SCORE_W_MP`: 오버슈트 가중치 (0.35, 강화됨)
  - `SCORE_W_ESS`: 정상상태 오차 가중치 (0.20)
  - `SCORE_W_BAND`: 밴드 유지 가중치 (0.15)
  - `SCORE_W_U`: 제어 노력 가중치 (0.05)
  - `SCORE_W_FAIL`: 추종 실패 패널티 가중치 (0.15)
  - `SCORE_W_PBRS`: PBRS 가중치 (0.10)

#### 4.12 초기 구간 피크 패널티 설정

```python
EARLY_PEAK_TIME_WINDOW = 0.5  # 0~0.5초 구간
EARLY_PEAK_PENALTY_SCALE = 0.2  # 패널티 스케일
EARLY_PEAK_PENALTY_MAX = 0.2  # 상한 (0.15~0.2)
```

**설명**:
- `EARLY_PEAK_TIME_WINDOW`: 초기 구간 시간 창 (0.5초)
- `EARLY_PEAK_PENALTY_SCALE`: 패널티 스케일 (0.2)
- `EARLY_PEAK_PENALTY_MAX`: 패널티 상한 (0.2)

#### 4.13 추종 실패 임계값

```python
TRACKING_FAIL_RMSE_THRESHOLD = 5.0
TRACKING_FAIL_BAND_RATIO = 0.3
```

**설명**:
- `TRACKING_FAIL_RMSE_THRESHOLD`: RMSE 임계값 (5.0N)
- `TRACKING_FAIL_BAND_RATIO`: 밴드 비율 임계값 (0.3)

#### 4.14 오버슈트 임계값

```python
OVERSHOOT_THRESHOLD_MILD = 5.0
OVERSHOOT_THRESHOLD_MODERATE = 15.0
OVERSHOOT_THRESHOLD_SEVERE = 30.0
```

**설명**:
- 경미/중간/심각 오버슈트 구분 (레거시, 현재 미사용)

#### 4.15 탐색/탐욕 비율 설정

```python
ACTOR_LOG_STD_MAX = -0.3  # -1.0 → -0.3 (더 큰 탐색 범위)
ACTOR_LOG_STD_MIN = -2.5  # 하한 보장
ACTOR_INITIAL_ALPHA = 0.1  # 0.02 → 0.1 (더 큰 초기값)
ACTOR_WEIGHT_GAIN = 0.05  # 0.5 → 0.05 (안정화)
```

**설명**:
- `ACTOR_LOG_STD_MAX`: Actor log_std 상한 (-0.3, σ ≤ 0.74)
- `ACTOR_LOG_STD_MIN`: Actor log_std 하한 (-2.5)
- `ACTOR_INITIAL_ALPHA`: 초기 entropy 계수 (0.1)
- `ACTOR_WEIGHT_GAIN`: 가중치 초기화 gain (0.05, Fine-tuning용)

#### 4.16 표준편차 Annealing 설정

```python
STD_ANNEAL_START_EPISODE = 0
STD_ANNEAL_END_EPISODE = 150  # 200 → 150 (더 빠르게)
STD_ANNEAL_INITIAL = 1.0
STD_ANNEAL_FINAL = 0.5  # 0.3 → 0.5 (덜 축소, 탐색 유지)
```

**설명**:
- `STD_ANNEAL_START_EPISODE`: Annealing 시작 에피소드 (0)
- `STD_ANNEAL_END_EPISODE`: Annealing 종료 에피소드 (150)
- `STD_ANNEAL_INITIAL`: 초기 std_scale (1.0)
- `STD_ANNEAL_FINAL`: 최종 std_scale (0.5)
- 선형 감소: 1.0 → 0.5 (0~150 에피소드)

#### 4.17 Target Entropy 동적 조정 설정

```python
TARGET_ENTROPY_INITIAL_FACTOR = -1.2  # 초기 100ep: 더 공격적 탐색
TARGET_ENTROPY_FINAL_FACTOR = -1.0    # 이후: 표준
TARGET_ENTROPY_TRANSITION_EPISODES = 100  # 전환 에피소드 수
```

**설명**:
- `TARGET_ENTROPY_INITIAL_FACTOR`: 초기 target_entropy 계수 (-1.2)
- `TARGET_ENTROPY_FINAL_FACTOR`: 최종 target_entropy 계수 (-1.0)
- `TARGET_ENTROPY_TRANSITION_EPISODES`: 전환 에피소드 수 (100)
- 초기 100ep: -1.2×action_dim, 이후: -1.0×action_dim

#### 4.18 레거시 보상 함수 파라미터 (미사용)

```python
TAU_RMSE = 2.5
TAU_SETTLE = 5.0
TAU_VAR = 0.15
TAU_U = 1.0
TAU_DU = 1.0
POTENTIAL_GAMMA = 0.99
REWARD_WEIGHT_PROGRESS = 0.05
REWARD_MIN = -100.0
REWARD_MAX = 50.0
```

**설명**: 레거시 코드 호환성 유지용 (현재 미사용)

---

## 📋 요약

### 실행 모듈
- **`__main__.py`**: 모듈 실행 엔트리 포인트, 기본 설정 사용
- **`main.py`**: 명령줄 인터페이스, 유연한 설정 지원

### 설정 모듈
- **`config.py`**: 타입 안전한 설정 관리 (dataclass)
- **`constants.py`**: 모든 상수 중앙 집중 관리

### 핵심 특징
1. **재현성 보장**: 시드 고정 (42)
2. **안전 종료**: 시그널 핸들러 지원
3. **유연한 설정**: 코드 수정 또는 명령줄 인자
4. **타입 안전성**: dataclass 사용
5. **중앙 집중 관리**: 모든 상수를 constants.py에 모음

---

**다음**: CODE_FUNCTIONALITY_2.md - 핵심 RL 모듈 (agent, env, comm, monitor)

