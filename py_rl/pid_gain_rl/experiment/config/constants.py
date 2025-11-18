"""
상수 정의 - 물리/표현 상수만
"""


class Constants:
    # ===== 상태 공간 차원 설정 =====
    STATE_BASE_DIM = 6  # 기본 상태 차원 (현재 힘/목표 힘 등 실시간 관측)
    STATE_TRAJECTORY_DIM = 0  # 궤적 요약 차원 제거
    STATE_DIM = STATE_BASE_DIM + STATE_TRAJECTORY_DIM  # 총 상태 차원 (6)
    
    # 신경망 기본 설정 (fine-tuning 기본값)
    DEFAULT_HIDDEN_DIM = 128
    DEFAULT_LR = 1e-4
    DEFAULT_LR_ACTOR = 1e-4
    DEFAULT_LR_CRITIC = 2e-4
    DEFAULT_GAMMA = 0.98
    DEFAULT_TAU = 0.01
    
    # PID 범위 (새 범위: 오버슈트 억제 및 댐핑 효과 확보)
    DEFAULT_PID_RANGE = {
        "Kp": (5.0, 80.0),
        "Ki": (10.0, 100.0),
        "Kd": (0.0, 1e-2),
    }
    
    # 통신 기본 설정
    DEFAULT_RECV_FREQ = 1000
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_EPISODES = 500
    MIN_EPISODES_FOR_SAVING = 50  # 최고 성능 모델 저장 시작 에피소드 (초반 lucky reward 방지)
    DEFAULT_EPISODE_SECONDS = 10.0
    DEFAULT_TARGET_FORCE = -60.0
    INITIAL_CONTACT_FORCE = -45.0
    INITIAL_PI_OUTPUT = 0.05
    DEFAULT_UPDATES_PER_EPISODE = 35  # 10 → 35 (세그먼트 분할 대응, 30~40 범위)
    
    # 통신 설정
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = 8888
    DEFAULT_RECV_TIMEOUT = 0.5
    DEFAULT_RECV_LOOP_TIMEOUT = 0.05
    DEFAULT_COMM_FAIL_MAX = 3
    DEFAULT_COMM_RETRY_DELAY = 0.1
    
    # 학습 설정
    DEFAULT_MAX_REWARDS_HISTORY = 1000
    DEFAULT_REPLAY_BUFFER_SIZE = 10000  # 🔥 2000 → 10000 (5배 유입 대응)
    MIN_BUFFER_FOR_LEARNING = 32
    MIN_BATCH_SIZE = 32
    
    # 저장 경로
    DEFAULT_MODEL_SAVE_DIR = (
        "/home/katech/Robot-Polishing-RL-system/"
        "py_rl/pid_gain_rl/saved_agents"
    )
    DEFAULT_LOG_DIR = (
        "/home/katech/Robot-Polishing-RL-system/"
        "py_rl/pid_gain_rl/experiment_logs"
    )
    
    # ===== 🔥 세그먼트 분할 설정 =====
    NUM_SEGMENTS = 5  # 에피소드를 5개 세그먼트로 분할 (2초씩)
    SEGMENT_LENGTH_S = 2.0  # 각 세그먼트 길이 (초)
    
    # Warm-start 설정
    WARM_START_ENABLED = True  # Warm-start 활성화 여부
    WARM_START_NUM_SAMPLES = 50  # LHS 샘플 수 (30~50 범위)
    
    SETTLING_HOLD_TIME_S = 0.5  # 정착 판정을 위한 밴드 유지 시간 (초)
    REWARD_MIN = -1.0
    REWARD_MAX = 1.0
    REWARD_ERROR_REF_PERCENT = 40.0  # 평균 %오차가 이 값일 때 보상 -1 (선형 스케일)
    
    # 제어 물리 상수
    BAND_TOLERANCE_N = 1.5
    SAFETY_FORCE_LIMIT = 100.0
    SAFETY_FORCE_PENALTY = -1.0
    BAND_RATIO_TOLERANCE = 0.05
    
    # ===== 🔥 탐색/탐욕 비율 설정 (새 PID 범위 탐색 강화) =====
    ACTOR_LOG_STD_MAX = -0.3  # -1.0 → -0.3 (더 큰 탐색 범위, σ ≤ e^{-0.3} ≈ 0.74)
    ACTOR_LOG_STD_MIN = -2.5  # 하한 보장
    ACTOR_INITIAL_ALPHA = 0.1  # 0.02 → 0.1 (더 큰 초기값, 자동 튜닝 활성화 시)
    
    # ===== 🔥 표준편차 Annealing 설정 =====
    STD_ANNEAL_START_EPISODE = 0
    STD_ANNEAL_END_EPISODE = 150  # 200 → 150 (더 빠르게)
    STD_ANNEAL_INITIAL = 1.2  # 초반 더 넓은 탐색
    STD_ANNEAL_FINAL = 0.4  # 후반 더 탐욕적
    
    # ===== Target Entropy 동적 조정 설정 =====
    TARGET_ENTROPY_INITIAL_FACTOR = -1.2  # 초기 100ep: 더 공격적 탐색
    TARGET_ENTROPY_FINAL_FACTOR = -0.9    # 이후: 더 탐욕적
    TARGET_ENTROPY_TRANSITION_EPISODES = 80  # 전환 에피소드 수 단축

    # ===== 초기 강제 탐색 설정 =====
    FORCED_RANDOM_EPISODES = 80  # 초기 80 에피소드: PID 랜덤 샘플로 탐색 공간 골고루 채우기
