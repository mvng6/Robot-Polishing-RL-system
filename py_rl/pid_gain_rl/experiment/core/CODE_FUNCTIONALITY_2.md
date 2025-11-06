# 📋 코드 기능 상세 분석 - Part 2: 핵심 RL 모듈

## 📁 모듈 목록

1. `agent.py` - SAC 에이전트 (Actor, Critic, ReplayBuffer)
2. `env.py` - PID Gain 강화학습 환경
3. `comm.py` - TCP/IP 통신 래퍼
4. `monitor.py` - 실시간 모니터링 GUI

---

## 1. `agent.py` - SAC 에이전트

**파일 경로**: `py_rl/pid_gain_rl/agent.py`  
**줄 수**: ~565줄  
**역할**: Soft Actor-Critic (SAC) 알고리즘 구현

### 주요 클래스

#### 1.1 `Actor` 클래스 (nn.Module)

**역할**: 정책 네트워크 (Policy Network)

**구조**:
- 입력: `state_dim` 차원 (20차원)
- 은닉층: 2층 MLP (128-128)
- 출력:
  - `mean`: 액션 평균 (3차원: Kp, Ki, Kd)
  - `log_std`: 액션 로그 표준편차 (3차원)

**주요 메서드**:

1. **`__init__()`**
   - `log_std_min=-2.5`, `log_std_max=-0.3` 설정
   - 가중치 초기화: `ACTOR_WEIGHT_GAIN=0.05` (Fine-tuning용)

2. **`forward(state)`**
   - 입력: 상태 벡터
   - 출력: `mean`, `log_std`
   - `mean`: [-10.0, 10.0]로 클리핑
   - `log_std`: [log_std_min, log_std_max]로 클리핑

3. **`sample(state, std_scale=1.0)`**
   - **표준편차 스케일링 지원** (annealing용)
   - `std_scale`: 0.3~1.0 범위
   - 정규분포에서 샘플링 → tanh 적용
   - 로그 확률 계산 (reparameterization trick)

#### 1.2 `Critic` 클래스 (nn.Module)

**역할**: Q-값 네트워크 (Twin Q-Networks)

**구조**:
- 입력: `state_dim + action_dim` (20 + 3 = 23차원)
- 은닉층: 2층 MLP (128-128)
- 출력: Q1, Q2 (각각 1차원)

**주요 메서드**:

1. **`forward(state, action)`**
   - 상태와 액션을 concat하여 입력
   - Q1, Q2 반환 (Twin Q-Networks)

#### 1.3 `ReplayBuffer` 클래스

**역할**: 경험 리플레이 버퍼

**주요 메서드**:

1. **`__init__(capacity=None)`**
   - `deque` 사용 (최대 크기 제한)
   - 기본 크기: `DEFAULT_REPLAY_BUFFER_SIZE` (10000)

2. **`push(state, action, reward, next_state, done)`**
   - transition 저장

3. **`sample(batch_size)`**
   - 랜덤 샘플링
   - 배치 형태로 반환 (numpy array)

#### 1.4 `PIDGainSACAgent` 클래스

**역할**: SAC 알고리즘 통합 에이전트

**주요 속성**:

1. **네트워크**:
   - `actor`: Actor 네트워크
   - `critic`: Critic 네트워크 (Q1, Q2)
   - `critic_target`: Target Critic 네트워크

2. **옵티마이저**:
   - `actor_opt`: Actor 옵티마이저 (LR: 1e-4)
   - `critic_opt`: Critic 옵티마이저 (LR: 2e-4)
   - `alpha_opt`: Entropy 계수 옵티마이저 (자동 튜닝 시)

3. **하이퍼파라미터**:
   - `gamma`: 할인율 (0.99)
   - `tau`: Soft update 계수 (0.01)
   - `alpha`: Entropy 계수 (초기: 0.1, 자동 튜닝)

4. **탐색 관련**:
   - `std_scale`: 표준편차 스케일 (annealing용, 1.0 → 0.5)
   - `target_entropy`: 목표 엔트로피 (동적 조정: -3.6 → -3.0)
   - `recent_actions`: 최근 액션 히스토리 (탐색 메트릭용)

**주요 메서드**:

1. **`__init__(cfg)`**
   - 네트워크 초기화
   - 옵티마이저 설정
   - 리플레이 버퍼 생성
   - Target entropy 동적 조정 준비

2. **`update_std_scale(episode_num)`**
   - **표준편차 Annealing**
   - 선형 감소: 1.0 → 0.5 (0~150 에피소드)
   - 탐색 강도 점진적 감소

3. **`update_target_entropy(episode_num)`**
   - **Target Entropy 동적 조정**
   - 초기 100ep: -1.2×action_dim (공격적 탐색)
   - 이후: -1.0×action_dim (표준)
   - 점진적 전환

4. **`select_action(state, evaluate=False)`**
   - 액션 선택
   - `evaluate=False`: 탐색 (std_scale 적용)
   - `evaluate=True`: 평가 (평균값만 사용)
   - PID gain으로 변환하여 반환
   - 탐색 메트릭 추적 (최근 액션 저장)

5. **`select_action_random()`**
   - 안전 위반 시 랜덤 PID 선택

6. **`store_transition(state, action, reward, next_state, done)`**
   - Transition 저장
   - NaN/Inf 검증
   - 보상 클리핑 (-100.0 ~ 50.0)
   - PID gain 정규화 ([-1, 1] 범위)

7. **`update_parameters_one_step(batch_size=None, num_updates=128)`**
   - **SAC 업데이트**
   - Critic 업데이트 (Q1, Q2)
   - Actor 업데이트 (정책)
   - Entropy 자동 튜닝 (선택적)
   - Target Critic Soft Update (tau=0.01)
   - Gradient clipping (2.0)

8. **`warm_start_buffer(num_samples=None)`**
   - **Warm-start 버퍼 초기화**
   - Latin Hypercube Sampling (LHS) 사용
   - Kp, Ki: 선형 샘플링
   - Kd: 로그 스케일 샘플링
   - 더미 transition 저장 (50개 샘플)

9. **`log_exploration_metrics(episode_num)`**
   - **탐색 메트릭 계산**
   - Action std/range 비율 계산
   - Kd decade 커버리지 계산
   - 최근 20개 액션 기준

10. **`save_model(path)`**
    - 모델 저장 (actor, critic, critic_target, optimizer, cfg)

11. **`load_model(path, strict=True)`**
    - 모델 로드
    - 옵티마이저 상태 복원
    - 보상 히스토리 복원

12. **`transfer_learning_setup(source_model_path, learning_rate_scale=0.1)`**
    - 전이학습 설정
    - 학습률 스케일링 (기본 0.1배)

---

## 2. `env.py` - PID Gain 강화학습 환경

**파일 경로**: `py_rl/pid_gain_rl/env.py`  
**줄 수**: ~1472줄  
**역할**: 강화학습 환경, 에피소드 관리, 보상 계산, 상태 생성

### 주요 클래스: `PIDGainEnvironment`

#### 2.1 초기화 (`__init__`)

**주요 속성**:

1. **설정**:
   - `cfg`: 설정 딕셔너리
   - `STATE_DIM`: 20차원 강제 설정 (12 + 8)

2. **컴포넌트**:
   - `agent`: SAC 에이전트
   - `comm`: 통신 객체
   - `cplogger`: 제어 성능 로거
   - `rlogger`: 보상 분석 로거
   - `ldlogger`: 학습 완료 로거

3. **에피소드 데이터**:
   - `episode_force_data`: 힘 데이터 수집
   - `episode_pi_output_data`: PI 출력 데이터 수집
   - `current_pid_gains`: 현재 PID gain

4. **히스토리**:
   - `previous_pid_gains`: 이전 에피소드 PID
   - `pid_gains_next`: 다음 에피소드 PID (미리 전송)
   - `episode_history`: 최근 5개 에피소드 기록
   - `historical_errors`: 최근 10개 에피소드 에러

#### 2.2 보상 계산 (`calculate_episode_reward`)

**역할**: 복합 보상 함수 계산

**단계**:

1. **입력 검증 및 기본값**
   - `force_data`, `target_force`, `episode_len_s` 검증

2. **핵심 지표 계산**:
   - **RMSE**: 전체 추종 오차
   - **초기 구간 피크 패널티**: 0~0.5초 구간 전용
   - **Overshoot**: % 단위로 일관화
   - **밴드 유지율**: ±1.5N 범위 내 유지 비율
   - **정착시간**: 연속 유지 기준 (2초)
   - **정상상태 오차 (ESS)**: 마지막 10% 구간 평균
   - **제어 입력 품질**: u_rms 정규화

3. **PBRS (Potential-Based Reward Shaping)**:
   - Potential 함수: `-abs(errors) / T_abs`
   - 개선분: `gamma * phi[1:] - phi[:-1]`
   - 양수만 사용 (개선 신호만)

4. **추종 실패 패널티**:
   - RMSE 기준: `(rmse - threshold) / threshold`
   - 밴드 비율 기준: `(threshold - band_ratio) / threshold`
   - 최대값 사용

5. **연속형 스코어 (0~1)**:
   - `S_ts = exp(-settling_time / tau_settle)`
   - `S_mp = exp(-overshoot_pct / tau_mp)`
   - `S_ess = exp(-abs(ess) / tau_ess)`
   - `S_band = band_ratio`
   - `S_u = exp(-u_rms_n / tau_u)`

6. **기준선 대비 개선량 (EWMA)**:
   - 기준선: `_baseline` (ts, mp, ess_abs)
   - 개선량: `(baseline - current) / tau`
   - EWMA 업데이트: `α=0.1` (최근 10개 에피소드)

7. **통합 보상 스코어 (0~1)**:
   ```
   reward_score = 0.30*S_ts + 0.35*S_mp + 0.20*S_ess + 0.15*S_band
                  + 0.05*S_u - 0.15*P_fail + 0.10*progress
                  + 0.10*I_improve - early_peak_penalty
   ```

8. **중심화 + tanh 소프트클립**:
   - 기준선: `_rew_baseline` (EWMA, β=0.99)
   - 중심화: `r_centered = reward_score - baseline`
   - tanh 클리핑: `reward = tanh(2.0 * r_centered)`
   - 최종 범위: [-1.0, 1.0]

9. **안전 위반 하드 패널티**:
   - 힘 > ±100N → `reward = -1.0`

10. **메트릭 반환**:
    - 기본 지표 (rmse, overshoot, settling_time 등)
    - 디버깅 정보 (reward_score, r_centered, r_baseline 등)

#### 2.3 초기 구간 피크 패널티 (`_calculate_early_peak_penalty`)

**역할**: 초기 구간 (0~0.5초) 피크 감지 및 패널티

**로직**:
1. 0~0.5초 구간 추출
2. 목표값보다 더 나쁜 방향으로의 최대 편차 계산
3. 패널티: `min(penalty_scale * peak_ratio, max_penalty)`
4. 상한: 0.2

#### 2.4 세그먼트 상태 생성 (`_build_segment_state`)

**역할**: 세그먼트별 20차원 상태 벡터 생성

**구성**:

1. **기존 12차원**:
   - 로봇PC 6차원: Kp, Ki, Kd, prev_reward, target_force, segment_len_s
   - 강화학습PC 6차원: force 평균, std, min, max, error 평균, std

2. **궤적 요약 8차원**:
   - `overshoot`: 오버슈트 (%)
   - `settling_time`: 정착시간 (초)
   - `rmse`: RMSE
   - `band_ratio`: 밴드 유지 비율
   - `oscillation_freq`: 진동 주파수 (FFT)
   - `oscillation_amp`: 진동 진폭 (FFT)
   - `rise_time`: 상승 시간 (10% → 90%)
   - `steady_state_error`: 정상상태 오차

**NaN/Inf 가드**: 모든 값이 NaN/Inf가 아니도록 보장

#### 2.5 메인 학습 루프 (`run_pid_optimization_training`)

**역할**: 전체 강화학습 루프 실행

**단계**:

1. **연결 설정**:
   - TCP/IP 연결 대기
   - RL 활성화 대기 (sander_active=True)

2. **모니터 시작**:
   - 실시간 모니터링 GUI 시작 (10 Hz)

3. **Warm-start**:
   - 첫 에피소드 전에만 버퍼 초기화 (LHS 샘플링)

4. **에피소드 루프**:
   - **초기 상태 생성**: 이전 PID 또는 기준 PID 사용
   - **PID gain 선택**: 첫 에피소드는 기준 PID (40, 50, 0)
   - **에피소드 시작 신호 전송**: episode_done=False
   - **데이터 수집**: 1kHz로 10초간 수집
   - **안전 위반 체크**: 힘 > ±100N → 조기 종료
   - **세그먼트 분할**: 5개 세그먼트 (2초씩)
   - **세그먼트별 보상 계산**: 각 세그먼트 보상 계산
   - **세그먼트별 상태 생성**: 20차원 상태 벡터
   - **리플레이 버퍼 저장**: 세그먼트별 transition 저장
   - **표준편차 Annealing 업데이트**
   - **Target Entropy 동적 조정**
   - **학습**: 버퍼 크기 >= 32일 때 업데이트 (35회)
   - **최고 성능 모델 저장**: 50 에피소드 이후 최고 보상만 저장
   - **다음 PID 계산**: 다음 에피소드 PID 미리 계산
   - **에피소드 완료 신호 전송**: episode_done=True + 다음 PID
   - **로봇 리셋 대기**: 2초 대기 (모니터링 지속)

5. **최종 처리**:
   - 성능 요약 저장
   - 그래프 생성
   - 학습 완료 신호 전송
   - 모니터 종료

#### 2.6 기타 메서드

1. **`is_episode_done(force_data, target_force)`**
   - 에피소드 종료 조건 확인 (시간 기반)

2. **`reset_episode()`**
   - 에피소드 리셋 (데이터 초기화)

3. **`generate_episode_reward_graph()`**
   - 에피소드별 보상 그래프 생성

---

## 3. `comm.py` - TCP/IP 통신 래퍼

**파일 경로**: `py_rl/pid_gain_rl/comm.py`  
**줄 수**: ~357줄  
**역할**: 로봇 제어 PC와의 TCP/IP 통신 관리

### 주요 클래스: `PIDGainCommunicator`

#### 3.1 초기화 (`__init__`)

**주요 속성**:
- `host`, `port`: 서버 주소
- `socket`, `conn`: 소켓 및 연결 객체
- `connected`: 연결 상태 플래그
- `receive_thread`: 수신 스레드
- `latest_state`: 최신 상태 데이터
- `latest_sander_active`: 최신 RL 활성화 플래그

**패킷 포맷**:
- **수신 (C++ → Python)**: `>HffffffBH` (29 bytes)
  - SOF, current_force, target_force, force_error, force_error_dot, force_error_int, pi_output, sander_active, checksum
- **전송 (Python → C++)**: `>HfffBBBH` (19 bytes)
  - SOF, Kp, Ki, Kd, timing_accurate, episode_done, learning_done, checksum

#### 3.2 주요 메서드

1. **`connect()`**
   - TCP/IP 서버 시작 (bind, listen)
   - 클라이언트 연결 대기
   - 연결 성공 시 수신 스레드 시작

2. **`start_receiving()`**
   - 수신 스레드 시작 (daemon thread)

3. **`_receive_loop()`**
   - **1kHz 정확 수신 루프**
   - 주기 고정 방식 (next_receive_time 기반)
   - 타임아웃 처리
   - 패킷 처리 및 상태 업데이트

4. **`_recv_exact(nbytes)`**
   - 정확한 바이트 수 수신 (청크 처리)

5. **`_process_packet(data)`**
   - 패킷 언팩
   - SOF 검증
   - 체크섬 검증 (CRC16)
   - 상태 배열 생성 (6차원)
   - 통계 업데이트

6. **`calculate_crc16(data)`**
   - CRC16 체크섬 계산

7. **`get_latest_state()`**
   - 최신 상태 반환 (스레드 안전)
   - 오래된 데이터 경고 (2초 이상)
   - sander_active 변경 감지

8. **`send_pid_once(kp, ki, kd, timing_accurate, episode_done, learning_done)`**
   - PID gain 전송
   - 패킷 생성 및 체크섬 추가
   - 전송 및 통계 업데이트

9. **`send_reset()`**
   - 리셋 신호 전송 (미사용)

10. **`get_communication_stats()`**
    - 통신 통계 반환 (uptime, packets_received, packets_sent, rates)

11. **`print_communication_stats()`**
    - 통신 통계 출력

12. **`close()`**
    - 연결 종료
    - 수신 스레드 종료
    - 소켓 닫기

---

## 4. `monitor.py` - 실시간 모니터링 GUI

**파일 경로**: `py_rl/pid_gain_rl/monitor.py`  
**줄 수**: ~188줄  
**역할**: 실시간 모니터링 GUI (matplotlib)

### 주요 클래스: `RLRealtimeMonitor`

#### 4.1 초기화 (`__init__`)

**매개변수**:
- `title`: 창 제목 (기본: "PID Gain RL Monitor")
- `hz`: 업데이트 주파수 (기본: 10 Hz)
- `rolling_window`: 롤링 윈도우 시간 (기본: 30.0초)

**속성**:
- `q`: Queue (프로세스 간 통신)
- `proc`: 프로세스 객체

#### 4.2 주요 메서드

1. **`start()`**
   - 별도 프로세스로 모니터 시작 (multiprocessing.Process)

2. **`stop(timeout=2.0)`**
   - 종료 신호 전송
   - 프로세스 종료 대기

3. **`reset_force_buffers()`**
   - 힘 데이터 버퍼 리셋

4. **`post_force(t_sec, current_f, desired_f)`**
   - 힘 데이터 전송 (Queue 사용)

5. **`post_reward(episode, reward)`**
   - 보상 데이터 전송 (Queue 사용)

6. **`_run()`**
   - **실제 모니터링 GUI 실행**
   - 백엔드 설정 (TkAgg 또는 Agg)
   - matplotlib FuncAnimation 사용
   - 2개 서브플롯:
     - **Force subplot**: 현재 힘 vs 목표 힘
     - **Reward subplot**: 에피소드별 보상
   - 롤링 윈도우 처리 (30초)
   - 10 Hz 업데이트

**특징**:
- **Headless 모드 지원**: TkAgg 없을 때 Agg 사용
- **롤링 윈도우**: 최근 30초 데이터만 표시
- **비동기 처리**: 별도 프로세스로 실행 (메인 루프 블로킹 방지)

---

## 📋 요약

### Agent 모듈
- **Actor**: 정책 네트워크 (128-128 MLP, std_scale 지원)
- **Critic**: Twin Q-Networks (128-128 MLP)
- **ReplayBuffer**: 경험 리플레이 버퍼 (10000 크기)
- **SAC 알고리즘**: 표준편차 Annealing, Target Entropy 동적 조정, Warm-start

### Environment 모듈
- **복합 보상 함수**: 스코어화 + PBRS + 기준선 중심화 + tanh 클리핑
- **세그먼트 분할**: 5개 세그먼트 (2초씩)
- **20차원 상태**: 기존 12차원 + 궤적 요약 8차원
- **안전 위반 처리**: 힘 > ±100N → 조기 종료

### Communication 모듈
- **1kHz 정확 수신**: 주기 고정 방식
- **스레드 안전**: Lock 사용
- **CRC16 체크섬**: 패킷 검증

### Monitor 모듈
- **실시간 GUI**: matplotlib 기반
- **롤링 윈도우**: 최근 30초 데이터
- **비동기 처리**: 별도 프로세스

---

**다음**: CODE_FUNCTIONALITY_3.md - 유틸리티 및 로거 모듈

