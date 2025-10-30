"""
상수 정의 - 물리/표현 상수만
"""


class Constants:
    # 신경망 기본 설정 (fine-tuning 기본값)
    DEFAULT_HIDDEN_DIM = 128
    DEFAULT_LR = 1e-4  # 학습률 추가 감소: 3e-4 → 1e-4 (초기 안정성 강화, NaN 방지)
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
    DEFAULT_EPISODE_SECONDS = 10.0  # 에피소드 길이: PID 과도응답은 5~10초 내 완료
    DEFAULT_TARGET_FORCE = -40.0  # FT 센서 좌표계: 압축력이 음수
    DEFAULT_UPDATES_PER_EPISODE = 8  # 업데이트 횟수 감소: 16 → 8 (과적합 방지)
    
    # 통신 설정
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = 8888
    DEFAULT_RECV_TIMEOUT = 0.5
    DEFAULT_RECV_LOOP_TIMEOUT = 0.05
    DEFAULT_COMM_FAIL_MAX = 3
    DEFAULT_COMM_RETRY_DELAY = 0.1
    
    # 학습 설정
    DEFAULT_MAX_REWARDS_HISTORY = 1000
    DEFAULT_REPLAY_BUFFER_SIZE = 2000  # 절충안: 2000 (메모리 사용량 적당)
    MIN_BUFFER_FOR_LEARNING = 32  # 학습 시작: 안정적 학습 보장 (최소 32개 필요)
    MIN_BATCH_SIZE = 32  # 최소 배치 크기: 초기 학습 안정성 (최소 32개 필요)
    
    # 저장 경로 (실험 경로는 그대로 유지)
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
    
    # (레거시) 보상 함수 스케일 파라미터 - 일부 지표 참고용
    TAU_RMSE = 2.5      # RMSE 스케일 (정확도 기준 강화)
    TAU_SETTLE = 5.0    # 정착시간 스케일
    TAU_VAR = 0.15      # 분산 스케일 (안정성 기준 완화)
    TAU_U = 1.0
    TAU_DU = 1.0
    POTENTIAL_GAMMA = 0.99
    SHAPING_WARMUP_S = 0.5
    REWARD_WEIGHT_PROGRESS = 0.05
    REWARD_MIN = -100.0
    REWARD_MAX = 50.0
    
    # 제어 물리 상수
    BAND_TOLERANCE_N = 1.5
    SETTLING_BAND_TOLERANCE = 0.5
    SETTLING_HOLD_TIME_S = 1.0
    SAFETY_FORCE_LIMIT = 100.0
    SAFETY_FORCE_PENALTY = -1.0  # 안전 위반 시 최대 패널티 ([-1,1] 스케일)
    PI_OUTPUT_MAX = 0.4
    PI_OUTPUT_SAT_THRESHOLD = 0.95
    
    # 보상 함수 가중치
    REWARD_WEIGHT_ACCURACY = 0.25      # RMSE (정확도) ↑
    REWARD_WEIGHT_BAND_QUALITY = 0.20  # 밴드 내 비율 ↑
    REWARD_WEIGHT_FAST_SETTLE = 0.15   # 정착 속도
    REWARD_WEIGHT_STABILITY = 0.15     # 안정성 (진동 억제)
    REWARD_WEIGHT_EFFICIENCY = 0.05    # 효율성
    REWARD_WEIGHT_SMOOTHNESS = 0.05    # 부드러움
    REWARD_WEIGHT_NO_SAT = 0.05        # 포화 방지
    REWARD_PENALTY_OVERSHOOT = 0.30    # 오버슈트
    REWARD_PENALTY_TRACKING_FAIL = 0.20  # 추종 실패
    REWARD_SUCCESS_BONUS_MAX = 0.20    # 성공 보너스

    # === 스코어화 기반 보상 가중치/스케일 (fine-tuning; 최종 보상 [-1,1]) ===
    SCORE_TAU_TS = 5.0            # s, 정착시간 스케일
    SCORE_TAU_MP_PERCENT = 10.0   # %, 오버슈트 스케일 (10%에서 1/e)
    SCORE_TAU_ESS_N = 1.0         # N, 정상상태 오차 스케일 (1N에서 1/e)
    SCORE_TAU_U = 0.5             # 제어입력 RMS 스케일
    SCORE_W_TS = 0.30
    SCORE_W_MP = 0.25
    SCORE_W_ESS = 0.20
    SCORE_W_BAND = 0.15
    SCORE_W_U = 0.05
    SCORE_W_FAIL = 0.15
    SCORE_W_PBRS = 0.10
    
    # 오버슈트 페널티 임계값
    OVERSHOOT_THRESHOLD_MILD = 5.0     # 5% 이하: 경미
    OVERSHOOT_THRESHOLD_MODERATE = 15.0  # 5~15%: 보통
    OVERSHOOT_THRESHOLD_SEVERE = 30.0  # 15% 이상: 심각
    
    # 추종 실패 기준
    TRACKING_FAIL_RMSE_THRESHOLD = 5.0  # RMSE > 5N이면 추종 실패
    TRACKING_FAIL_BAND_RATIO = 0.3      # 밴드 내 체류 < 30%면 추종 실패
    
    # 목표 범위 기준
    BAND_RATIO_TOLERANCE = 0.05  # ±5% 목표 범위
    BAND_RATIO_TOLERANCE_STRICT = 0.02  # ±2% 엄격한 범위

