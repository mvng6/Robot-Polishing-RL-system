# 📋 코드 기능 상세 분석 - Part 1: 실행 및 설정 모듈

## 📁 모듈 목록 (experiment/config)
1. `__main__.py` - 모듈 실행 엔트리
2. `main.py` - CLI 실행 스크립트
3. `config.py` - dataclass 기반 설정 관리
4. `constants.py` - 학습/통신/보상 상수 집합

---

## 1. `__main__.py` - 모듈 실행 엔트리 포인트
**경로**: `py_rl/pid_gain_rl/experiment/config/__main__.py`  
**역할**: `python -m pid_gain_rl.experiment.config` 실행 시의 엔트리

- 기본값 설정 후 `create_config()`로 설정 생성 (`RECV_FREQUENCY_HZ=1000`, `EPISODE_LENGTH_SECONDS=10.0`).
- 시드 고정(42) 후 `PIDGainEnvironment` 생성, 시그널 핸들러 설치, 학습 루프 실행.
- KeyboardInterrupt/예외 발생 시 PID 중단 신호 전송, 로깅 및 `DataSaver.save_all_data()` 호출 후 통신 종료.
- 전역 `_global_env`를 시그널 핸들러에서 참조.

실행 예:
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl
python3 -m pid_gain_rl.experiment.config
```

---

## 2. `main.py` - 명령줄 인터페이스
**경로**: `py_rl/pid_gain_rl/experiment/config/main.py`  
**역할**: argparse 기반 실행 스크립트 (모델 로드/경로 지정 등 실험 편의성 제공)

- 인자: `--episodes`, `--batch-size`, `--lr`, `--lr-actor`, `--lr-critic`, `--target-force`, `--episode-seconds`, `--recv-freq`, `--load-model`, `--log-dir`, `--model-dir`.
- `create_config()`로 기본 설정 후 인자로 덮어쓰기, 시작 정보 출력(STATE_DIM, 세그먼트 분할, 탐색 설정 등).
- 로그/모델 디렉토리 생성, 환경/시그널 핸들러 준비, 선택적으로 모델 로드.
- 학습 루프/예외 처리 플로우는 `__main__.py`와 동일.

실행 예:
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl/experiment/config
python3 main.py --episodes 300 --target-force -60 --recv-freq 800
```

---

## 3. `config.py` - 설정 관리
**경로**: `py_rl/pid_gain_rl/experiment/config/config.py`  
**역할**: 타입 안전한 설정 dataclass + 헬퍼 함수

### `Config` 클래스
- 기본 상태/액션: `state_dim=Constants.STATE_DIM`(10), `action_dim=4`(precharge+PID).
- 학습률: `lr=1e-4`, `lr_actor=1e-4`, `lr_critic=2e-4`, `gamma=0.98`, `tau=0.01`, `auto_entropy=True`.
- PID/공압 범위: `pid_range`(Kp 1~70, Ki 0~70, Kd 0~0.03), `precharge_range=(0.01,0.03)`.
- 에피소드/통신: `episode_seconds=10.0`, `target_force=-60.0`, `updates_per_episode=35`, `episodes=500`, `recv_freq_hz=1000`, `batch_size=64`, `host=0.0.0.0`, `port=8888`, `recv_timeout_sec=0.5`, `recv_loop_timeout_sec=0.05`.
- 실패/경로: `comm_fail_max=3`, `comm_retry_delay=0.1`, `model_save_dir`, `log_dir`.
- 학습: `max_episode_rewards_history=1000`, `replay_buffer_size=10000`.
- `__post_init__`에서 `recv_interval_sec` 계산. `to_dict()`는 기존 코드 호환을 위한 대문자 키 사용. `from_dict()`는 누락 키에 기본값 적용(액션 차원 기본 3으로 로드 호환).

### 헬퍼 함수
- `create_config(recv_freq_hz=None, episode_seconds=None)`: 값 검증 후 Config 생성, 수신 주파수/에피소드 길이 오버라이드.
- `change_episode_length(config, new_length_seconds)`: 길이 검증 후 갱신, 데이터 개수 안내 출력.

---

## 4. `constants.py` - 상수 정의
**경로**: `py_rl/pid_gain_rl/experiment/config/constants.py`  
**역할**: 학습/보상/통신/탐색 관련 모든 하드코딩 값

- 상태/행동: `STATE_DIM=10`(실시간 6 + 준비 컨텍스트 4), `STATE_BASE_DIM=10`, `ACTION_DIM`은 Config에서 4로 사용.
- 학습 기본: `DEFAULT_HIDDEN_DIM=128`, `DEFAULT_LR=1e-4`, `DEFAULT_LR_ACTOR=1e-4`, `DEFAULT_LR_CRITIC=2e-4`, `DEFAULT_GAMMA=0.98`, `DEFAULT_TAU=0.01`.
- PID/공압 범위: `DEFAULT_PID_RANGE` Kp(1~70), Ki(0~70), Kd(0~0.03); `DEFAULT_PRECHARGE_RANGE=(0.01,0.03)`.
- 통신/에피소드: `DEFAULT_RECV_FREQ=1000`, `DEFAULT_BATCH_SIZE=64`, `DEFAULT_EPISODES=500`, `MIN_EPISODES_FOR_SAVING=50`, `DEFAULT_EPISODE_SECONDS=10.0`, `DEFAULT_TARGET_FORCE=-60.0`, `DEFAULT_UPDATES_PER_EPISODE=35`, 타임아웃 0.5/0.05, 재시도 최대 3회(지연 0.1s), host/port 0.0.0.0:8888.
- 저장 경로: 모델 `/home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl/saved_agents`, 로그 `/home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl/experiment_logs`.
- 세그먼트/워밍업: `NUM_SEGMENTS=5`, `SEGMENT_LENGTH_S=2.0`, `WARM_START_ENABLED=True`, `WARM_START_NUM_SAMPLES=50`.
- 보상/안전: `SETTLING_HOLD_TIME_S=0.5`, `REWARD_MIN/MAX=-1/1`, `REWARD_ERROR_REF_PERCENT=30.0`, `BAND_TOLERANCE_N=1.5`, `BAND_RATIO_TOLERANCE=0.05`, `SAFETY_FORCE_LIMIT=100.0`, `SAFETY_FORCE_PENALTY=-1.0`, `TARGET_REACHED_TOLERANCE_N=0.5`.
- 탐색: `ACTOR_LOG_STD_MAX=-0.3`, `ACTOR_LOG_STD_MIN=-2.5`, `ACTOR_INITIAL_ALPHA=0.1`.
- 표준편차 Annealing: start 0, end 180, scale 1.2 → 0.5.
- Target Entropy 전환: 초기 계수 -1.2 → 최종 -0.9, 120 에피소드에 걸쳐 선형 전환.
- 기타: `DEFAULT_MAX_REWARDS_HISTORY=1000`, `DEFAULT_REPLAY_BUFFER_SIZE=10000`, 강제 랜덤 탐색 `FORCED_RANDOM_EPISODES=80`, 초기 상태 상수 `INITIAL_CONTACT_FORCE=-45.0`, `INITIAL_PI_OUTPUT=0.05`, 프리차지/보상 워밍업 스킵 0초.

---

**다음**: CODE_FUNCTIONALITY_2.md - 핵심 RL 모듈 (agent, env, comm, monitor)
