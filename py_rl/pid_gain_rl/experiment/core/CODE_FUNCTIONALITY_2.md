# 📋 코드 기능 상세 분석 - Part 2: 핵심 RL 모듈

## 📁 모듈 목록 (experiment/core)
1. `agent.py` - SAC 에이전트 (Actor/Critic/ReplayBuffer)
2. `env.py` - PID Gain RL 환경 및 학습 루프
3. `comm.py` - 로봇/센서 TCP 래퍼
4. `monitor.py` - 실시간 모니터 GUI 프로세스

---
## 🆕 최근 주요 포인트
- 상태 10차원(힘/오차 6 + 준비 컨텍스트 4), 액션 4차원(precharge + PID).
- 세그먼트 학습: 1 에피소드 = 5개 세그먼트, 세그먼트별 보상/transition 저장.
- 탐색 강화: log_std 상한 -0.3, std annealing(1.2→0.5, ep 0~180), target entropy 동적 전환(-1.2→-0.9, 120ep).
- 안전 로직: 힘 한계(±100N) 위반 시 패널티 -1, 랜덤 PID 전송 후 에피소드 스킵.

---

## 1. `agent.py` - SAC 에이전트
**경로**: `py_rl/pid_gain_rl/experiment/core/agent.py`  
**역할**: SAC 네트워크/버퍼/탐색 관리, LR·엔트로피 스케줄, warm-start

- **Actor/Critic**: 128-128 MLP 2층. Actor log_std ∈ [-2.5,-0.3], tanh 후 precharge+PID로 스케일링.
- **탐색 스케줄**:
  - `update_std_scale(ep)`: std_scale 1.2 → 0.5 선형(0~180 ep).
  - `update_target_entropy(ep)`: -1.2×action_dim → -0.9×action_dim을 120 ep에 걸쳐 전환.
  - `recent_actions` 기반 탐색 메트릭(표준편차 비율, Kd 커버리지) 20개 이상일 때 계산.
- **리플레이/Warm-start**: `ReplayBuffer` maxlen 기본 10000. `warm_start_buffer()`가 LHS(또는 랜덤)로 PID 샘플 50개를 더미 transition으로 채움.
- **액션 처리**:
  - `select_action(state, evaluate=False)`: std_scale 적용 샘플 → `scale_action_to_control`로 (precharge, PID) 반환.
  - `select_action_random()`: 안전 위반 시 사용할 랜덤 precharge+PID 생성.
  - `store_transition(...)`: NaN/Inf 가드, 보상 클리핑([-1,1]), precharge/PID를 [-1,1]로 정규화해 저장(Kd는 로그 스케일 지원).
- **학습**: `update_parameters_one_step(batch_size, num_updates=128)`에서 SAC 업데이트, gradient clip 2.0, 자동 엔트로피 튜닝 지원. 에피소드 150 이후 학습률을 0.7배로 한 번만 축소.
- **모델 입출력**: `save_model`, `load_model`, `transfer_learning_setup`(전이학습 시 학습률 스케일).

---

## 2. `env.py` - PID Gain 강화학습 환경
**경로**: `py_rl/pid_gain_rl/experiment/core/env.py`  
**역할**: 통신 연결, 에피소드 실행/보상 계산/세그먼트 전처리, 로깅·모니터링

- **초기화**: `cfg["STATE_DIM"]`을 Constants 값(10)으로 강제, 로그 폴더 구조 구성, `PIDGainCommunicator`, `PIDGainSACAgent`, 실시간 모니터 준비.
- **보상 계산**: `calculate_episode_reward(force_data, pi_output_data, target_force, episode_len_s)`  
  - 평균 절대 오차%를 `REWARD_ERROR_REF_PERCENT(30%)` 기준으로 선형 스케일해 [-1,1] 클리핑.  
  - rmse/overshoot/settling_time(0.5s 연속 유지), band_ratio(±1.5N) 등 지표 반환.
- **메인 루프 `run_pid_optimization_training(episodes)`**:
  - 통신 서버 bind/accept 후 RL 활성화(sander_active True)까지 대기.
  - Warm-start 버퍼 1회 수행(옵션).  
  - Ep1은 고정 PID(40,50,0) + precharge 중간값으로 시작, 이후 에피소드는 이전에 미리 전송한 PID/프리차지 사용.
  - sander_active 상승 에지 감지 후 1kHz 데이터 수집(기본 10s, 워밍업 스킵 0s). 통신 반복값 감지 시 에피소드 무효 처리 후 재시도.
  - 안전 위반(|force|>100N) 시 즉시 패널티 -1, 랜덤 PID를 다음 에피소드로 전송(episode_done=True) 후 짧은 리셋 대기.
  - 수집 완료 시 5개 세그먼트로 분할 → 각 세그먼트 보상/지표 계산 → `store_transition()` 저장 → `update_parameters_one_step()`로 SAC 업데이트.
  - Reward/Force/PI를 실시간 모니터에 10Hz로 전송, ControlPerformance/RewardBreakdown 로거에 기록. 모델/보상 그래프 저장은 로거·DataSaver를 통해 처리.

---

## 3. `comm.py` - 로봇 통신 래퍼
**경로**: `py_rl/pid_gain_rl/experiment/core/comm.py`  
**역할**: TCP 서버 소켓으로 로봇 제어 PC와 통신, 상태 수신(기본 1kHz)·PID 전송

- **수신 패킷(>HffffffBfffBH, 46B)**  
  SOF 0xAAAA, current_force, target_force, force_error, force_error_dot, force_error_int, pi_output, sander_active(1B), precharge_applied, j3_prep, prep_force_avg, prep_flag(1B), checksum(CRC16). 준비 구간에서 prep_flag ON 동안 force를 적산해 prep_force_avg로 고정 후 상승 에지에서 유지.
- **전송 패킷(>HffffBBBH, 23B)**  
  SOF 0xBBBB, precharge(소수 4자리로 라운드), Kp, Ki, Kd, timing_accurate, episode_done, learning_done, checksum(CRC16).
- 수신 스레드가 `RECV_INTERVAL_SEC` 주기로 패킷 파싱. 2초 이상 데이터 지연 시 1회 경고.
- `send_pid_once(...)`: PID/프리차지 전송 및 로깅. episode_done=True일 때 다음 에피소드용 precharge/PID 모두 포함. `send_reset()`은 간단한 리셋 패킷 전송. `get_communication_stats/print_communication_stats`로 통계 제공.

---

## 4. `monitor.py` - 실시간 모니터
**경로**: `py_rl/pid_gain_rl/experiment/core/monitor.py`  
**역할**: 별도 프로세스에서 Force/Reward/PID/PI를 시각화

- TkAgg 사용 가능 시 GUI 창, 불가 시 Agg 백엔드로 headless 로그만 유지.  
- `post_force`, `post_reward`, `post_pid`, `post_pi_output`로 메시지 전송, `reset_force_buffers`로 버퍼 초기화, `stop()`은 종료 메시지 후 조인.
   - **세그먼트별 상태 생성**: 10차원 상태 벡터 (통신 패킷 전체)
   - **리플레이 버퍼 저장**: 세그먼트별 transition 저장
   - **10-episode 로그**: 매 10번째 에피소드마다 `control_log/`에 힘 궤적 그래프(PNG)와 raw CSV 저장
   - **표준편차 Annealing 업데이트**
   - **Target Entropy 동적 조정**
   - **학습**: 버퍼 크기 >= 32일 때 업데이트 (35회)
   - **최고 성능 모델 저장**: 50 에피소드 이후 최고 보상만 저장
   - **다음 PID 계산**: 다음 에피소드 PID 미리 계산
   - **에피소드 완료 신호 전송**: episode_done=True + 다음 PID/프리차지(precharge, Kp, Ki, Kd 모두 전송)
   - **로봇 리셋 대기**: 2초 대기 (모니터링 지속)

5. **최종 처리**:
   - 성능 요약 저장
   - 그래프 생성
   - 학습 완료 신호 전송
   - 모니터 종료

#### 2.5 기타 메서드

1. **`is_episode_done(force_data, target_force)`**
   - 에피소드 종료 조건 확인 (시간 기반)

2. **`reset_episode()`**
   - 에피소드 리셋 (데이터 초기화)

3. **`generate_episode_reward_graph()`**
   - 에피소드별 보상 그래프 생성

4. **`_export_control_trace(episode_num, force_series, state_history)`**
   - 10 에피소드마다 현재 힘/목표 힘 데이터를 0~10초 범위로 시각화
   - `learning_done_*/control_log/`에 그래프(PNG)와 동일 구간 CSV(`time_s`, `current_force_N`, `target_force_N`) 저장
   - 현재 힘은 검은 실선, 목표 힘은 붉은 점선으로 표시하며 모든 레이블은 영문

---

## 3. `comm.py` - TCP/IP 통신 래퍼

**파일 경로**: `py_rl/pid_gain_rl/experiment/core/comm.py`  
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
- **수신 (C++ → Python)**: `>HffffffBfffBH` (46 bytes)  
  SOF 0xAAAA, current_force, target_force, force_error, force_error_dot, force_error_int, pi_output, sander_active(1B), precharge_applied, j3_prep, prep_force_avg, prep_flag(1B), checksum(CRC16)
- **전송 (Python → C++)**: `>HffffBBBH` (23 bytes)  
  SOF 0xBBBB, precharge, Kp, Ki, Kd, timing_accurate, episode_done, learning_done, checksum(CRC16)

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
   - 상태 배열 생성 (10차원: 통신 패킷 전체)
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

**파일 경로**: `py_rl/pid_gain_rl/experiment/core/monitor.py`  
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

6. **`post_pi_output(t_sec, pi_output)`**
   - 실시간 압력(PI 출력) 데이터 전송

7. **`_run()`**
   - **실제 모니터링 GUI 실행**
   - 백엔드 설정 (TkAgg 또는 Agg)
   - matplotlib FuncAnimation 사용
   - Force/Reward 2개 서브플롯 + Force 영역 안쪽 텍스트
     - **Force subplot**: 현재 힘 vs 목표 힘
     - **Reward subplot**: 에피소드별 보상
   - Force subplot 왼쪽 상단에 `Pressure: {value} MPa` 텍스트 업데이트
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
- **ReplayBuffer**: 경험 리플레이 버퍼 (기본 10000)
- **SAC 알고리즘**: 표준편차 Annealing, Target Entropy 동적 조정, Warm-start
- **PID 스케일링**: Kd 로그 스케일 일관화, PID 게인은 0.01 단위 양자화(소수 둘째 자리), 보상은 `[-1, 1]` 범위로 클리핑

### Environment 모듈
- **보상 함수**: 평균 힘 오차(%) 기반 단순 선형 보상 (0%→1, 100%→-1)
- **세그먼트 분할**: 5개 세그먼트 (2초씩, 총 10초)
- **10차원 상태**: 통신 패킷 전체 사용(힘/오차/PI 출력 + prep 컨텍스트 4채널)
- **안전 위반 처리**: 힘 > ±100N → 즉시 종료 및 패널티
- **로그 관리**: 매 10번째 에피소드마다 `control_log/`에 힘 궤적 그래프·CSV 저장

### Communication 모듈
- **1kHz 정확 수신**: 주기 고정 방식
- **스레드 안전**: Lock 사용
- **CRC16 체크섬**: 패킷 검증

### Monitor 모듈
- **실시간 GUI**: matplotlib 기반 (Force / Episode Reward / Pressure 3중 표시)
- **롤링 윈도우**: 최근 30초 데이터
- **비동기 처리**: 별도 프로세스

---

**다음**: CODE_FUNCTIONALITY_3.md - 유틸리티 및 로거 모듈
