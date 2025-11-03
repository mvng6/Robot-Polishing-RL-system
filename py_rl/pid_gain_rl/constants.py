"""
상수 정의 - 물리/표현 상수만
"""


class Constants:
    # 신경망 기본 설정 (fine-tuning 기본값)
    DEFAULT_HIDDEN_DIM = 128
    DEFAULT_LR = 1e-4
    DEFAULT_LR_ACTOR = 1e-4
    DEFAULT_LR_CRITIC = 2e-4
    DEFAULT_GAMMA = 0.99
    DEFAULT_TAU = 0.01
    
    # PID 범위 (fine-tuning 국소 탐색)
    DEFAULT_PID_RANGE = {
        "Kp": (35.0, 45.0),
        "Ki": (45.0, 55.0),
        "Kd": (1e-6, 1e-3),
    }
    
    # 통신 기본 설정
    DEFAULT_RECV_FREQ = 1000
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_EPISODES = 500
    DEFAULT_EPISODE_SECONDS = 10.0
    DEFAULT_TARGET_FORCE = -40.0
    DEFAULT_UPDATES_PER_EPISODE = 10  # 🔥 8 → 10 (세그먼트 분할 대응)
    
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
    
    WAIT_MESSAGE_INTERVAL = 1.0
    DEFAULT_FORCE_VALUE = -30.0
    
    # ===== 🔥 세그먼트 분할 설정 =====
    NUM_SEGMENTS = 5  # 에피소드를 5개 세그먼트로 분할 (2초씩)
    SEGMENT_LENGTH_S = 2.0  # 각 세그먼트 길이 (초)
    
    # ===== 세그먼트용 보상 파라미터 조정 =====
    SHAPING_WARMUP_S = 0.2  # 🔥 0.5 → 0.2 (세그먼트용)
    SETTLING_HOLD_TIME_S = 0.5  # 🔥 1.0 → 0.5 (세그먼트용)
    
    # (레거시) 보상 함수 스케일 파라미터
    TAU_RMSE = 2.5
    TAU_SETTLE = 5.0
    TAU_VAR = 0.15
    TAU_U = 1.0
    TAU_DU = 1.0
    POTENTIAL_GAMMA = 0.99
    REWARD_WEIGHT_PROGRESS = 0.05
    REWARD_MIN = -100.0
    REWARD_MAX = 50.0
    
    # 제어 물리 상수
    BAND_TOLERANCE_N = 1.5
    SETTLING_BAND_TOLERANCE = 1.0  # 0.5 → 1.0 (그래프 기준 실제 진동 고려)
    SAFETY_FORCE_LIMIT = 100.0
    SAFETY_FORCE_PENALTY = -1.0
    PI_OUTPUT_MAX = 0.4
    PI_OUTPUT_SAT_THRESHOLD = 0.95
    
    # 보상 함수 가중치 (레거시)
    REWARD_WEIGHT_ACCURACY = 0.25
    REWARD_WEIGHT_BAND_QUALITY = 0.20
    REWARD_WEIGHT_FAST_SETTLE = 0.15
    REWARD_WEIGHT_STABILITY = 0.15
    REWARD_WEIGHT_EFFICIENCY = 0.05
    REWARD_WEIGHT_SMOOTHNESS = 0.05
    REWARD_WEIGHT_NO_SAT = 0.05
    REWARD_PENALTY_OVERSHOOT = 0.30
    REWARD_PENALTY_TRACKING_FAIL = 0.20
    REWARD_SUCCESS_BONUS_MAX = 0.20

    # ===== 스코어화 기반 보상 시스템 =====
    SCORE_TAU_TS = 5.0
    SCORE_TAU_MP_PERCENT = 12.0
    SCORE_TAU_ESS_N = 1.0
    SCORE_TAU_U = 0.5
    
    SCORE_W_TS   = 0.30
    SCORE_W_MP   = 0.25
    SCORE_W_ESS  = 0.20
    SCORE_W_BAND = 0.15
    SCORE_W_U    = 0.05
    SCORE_W_FAIL = 0.15
    SCORE_W_PBRS = 0.10
    
    TRACKING_FAIL_RMSE_THRESHOLD = 5.0
    TRACKING_FAIL_BAND_RATIO = 0.3
    
    OVERSHOOT_THRESHOLD_MILD = 5.0
    OVERSHOOT_THRESHOLD_MODERATE = 15.0
    OVERSHOOT_THRESHOLD_SEVERE = 30.0
    
    BAND_RATIO_TOLERANCE = 0.05
    BAND_RATIO_TOLERANCE_STRICT = 0.02
    
    # ===== 🔥 Fine-tuning 설정 =====
    ACTOR_LOG_STD_MAX = -1.0  # 🔥 0.5 → -1.0 (탐험 축소)
    ACTOR_INITIAL_ALPHA = 0.02  # 🔥 0.2 → 0.02 (탐험 축소)
    ACTOR_WEIGHT_GAIN = 0.05  # 🔥 0.5 → 0.05 (안정화)
    
    # ===== 🔥 표준편차 Annealing 설정 =====
    STD_ANNEAL_START_EPISODE = 0
    STD_ANNEAL_END_EPISODE = 200
    STD_ANNEAL_INITIAL = 1.0
    STD_ANNEAL_FINAL = 0.3

