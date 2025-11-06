# 📋 코드 기능 상세 분석 - Part 3: 유틸리티 및 로거 모듈

## 📁 모듈 목록

### 유틸리티 모듈 (`utils/`)
1. `math_utils.py` - 수학 유틸리티 (PID 액션 스케일링, 상태 생성)
2. `data_saver.py` - 데이터 저장 유틸리티
3. `signals.py` - 시그널 처리 (안전한 종료)

### 로거 모듈 (`loggers/`)
4. `base_logger.py` - 기본 로거 (AppLogger)
5. `control_performance.py` - 제어 성능 로거
6. `reward_breakdown.py` - 보상 분석 로거
7. `learning_done.py` - 학습 완료 로거

---

## 1. `utils/math_utils.py` - 수학 유틸리티

**파일 경로**: `py_rl/pid_gain_rl/utils/math_utils.py`  
**줄 수**: ~120줄  
**역할**: PID 액션 스케일링 및 초기 상태 생성

### 주요 함수

#### 1.1 `scale_action_to_pid(action, pid_range)`

**역할**: 액션을 PID gain으로 스케일링

**매개변수**:
- `action`: 정규화된 액션 배열 ([-1, 1] 범위, 3차원)
- `pid_range`: PID 범위 딕셔너리

**로직**:
1. **Kp, Ki**: 선형 매핑
   ```python
   kp = kp_lo + (a[0] + 1.0) * 0.5 * (kp_hi - kp_lo)
   ki = ki_lo + (a[1] + 1.0) * 0.5 * (ki_hi - ki_lo)
   ```
   - 소수점 2자리 반올림

2. **Kd**: 로그 스케일 매핑
   ```python
   kd_log = loL + (a[2] + 1.0) * 0.5 * (hiL - loL)
   kd = 10 ** kd_log
   ```
   - 극소 범위 해상도 확보 (예: [1e-4, 5e-2])
   - 소수점 6자리 유지

**반환값**: `np.array([kp, ki, kd], dtype=np.float32)`

#### 1.2 `create_initial_state(force_data, target_force, prev_pid_gains=None, episode_history=None, dt_sec=0.001)`

**역할**: 초기 상태 벡터 생성 (20차원)

**매개변수**:
- `force_data`: 힘 데이터 리스트
- `target_force`: 목표 힘
- `prev_pid_gains`: 이전 PID gain [Kp, Ki, Kd]
- `episode_history`: 에피소드 히스토리 (미사용)
- `dt_sec`: 샘플링 시간 (미사용)

**로직**:

1. **기존 12차원**:
   - 로봇PC 6차원:
     - Kp, Ki, Kd (이전 PID 또는 기본값)
     - prev_reward (0.0)
     - target_force
     - episode_seconds (10.0)
   - 강화학습PC 6차원:
     - force 평균, std, min, max
     - error 평균, std

2. **궤적 요약 8차원** (초기값은 모두 0):
   - overshoot, settling_time, rmse, band_ratio
   - oscillation_freq, oscillation_amp
   - rise_time, steady_state_error

**NaN/Inf 가드**: 모든 값이 NaN/Inf가 아니도록 보장

**반환값**: `np.array` (20차원, dtype=np.float32)

---

## 2. `utils/data_saver.py` - 데이터 저장 유틸리티

**파일 경로**: `py_rl/pid_gain_rl/utils/data_saver.py`  
**줄 수**: ~62줄  
**역할**: 모든 데이터 저장 통합 관리

### 주요 클래스

#### 2.1 `Logger` 클래스

**역할**: 간단한 로거 (타임스탬프 + 아이콘)

**메서드**:
- `log(level, message)`: 로그 출력
  - 레벨: INFO, SUCCESS, WARNING, ERROR, DEBUG
  - 아이콘 자동 추가

#### 2.2 `DataSaver` 클래스

**역할**: 모든 데이터 저장 통합 관리

**메서드**:

1. **`save_all_data(env, current_episode=None, force=True)`**
   - **보상 분석 데이터 저장**:
     ```python
     env.rlogger.flush_if_needed(..., force=True, ...)
     ```
   - **제어 성능 지표 저장**:
     ```python
     env.cplogger.save_performance_summary()
     env.cplogger.generate_plots()
     ```
   - 예외 처리 포함

---

## 3. `utils/signals.py` - 시그널 처리

**파일 경로**: `py_rl/pid_gain_rl/utils/signals.py`  
**줄 수**: ~76줄  
**역할**: 안전한 종료 처리 (SIGINT, SIGTERM)

### 주요 함수

#### 3.1 `signal_handler(signum, frame, env=None)`

**역할**: 시그널 핸들러 (SIGINT, SIGTERM)

**로직**:
1. **학습 종료 신호 전송**:
   ```python
   env.comm.send_pid_once(0.0, 0.0, 0.0, True, False, True)
   ```
   - `learning_done=True` 전송

2. **데이터 저장**:
   - 보상 분석 데이터 저장
   - 제어 성능 지표 저장

3. **프로그램 종료**: `sys.exit(0)`

#### 3.2 `install_signal_handlers()`

**역할**: 시그널 핸들러 설치

**로직**:
- `__main__` 모듈에서 `_global_env` 참조
- SIGINT, SIGTERM 핸들러 등록

**사용**: `__main__.py`, `main.py`에서 호출

---

## 4. `loggers/base_logger.py` - 기본 로거

**파일 경로**: `py_rl/pid_gain_rl/loggers/base_logger.py`  
**줄 수**: ~28줄  
**역할**: 애플리케이션 로거 (Python 표준 logging 래핑)

### 주요 클래스: `AppLogger`

**역할**: 타임스탬프 + 아이콘 로거

**메서드**:

1. **`log(level, message)`** (정적 메서드)
   - 타임스탬프 생성 (`HH:MM:SS.mmm`)
   - 레벨별 아이콘 매핑:
     - INFO: ℹ️
     - SUCCESS: ✅
     - WARNING: ⚠️
     - ERROR: ❌
     - DEBUG: 🔍
   - 콘솔 출력

**사용**: 모든 모듈에서 `AppLogger.log()` 호출

---

## 5. `loggers/control_performance.py` - 제어 성능 로거

**파일 경로**: `py_rl/pid_gain_rl/loggers/control_performance.py`  
**줄 수**: ~913줄  
**역할**: 제어공학 지표 계산 및 시각화

### 주요 클래스: `ControlPerformanceLogger`

#### 5.1 초기화 (`__init__`)

**속성**:
- `log_dir`: 로그 디렉토리
- `control_perf_dir`: 제어 성능 서브폴더
- `time_data`, `force_data`, `target_data`, `error_data`: 데이터 버퍼
- `pi_output_data`, `input_data`: 제어 입력 데이터
- `episode_metrics`: 에피소드별 지표 리스트

#### 5.2 데이터 수집

1. **`add_data_point(time, force, target, control_effort, pi_output, pid_gains)`**
   - 실시간 데이터 포인트 추가
   - 버퍼에 저장 (리스트)

#### 5.3 제어공학 지표 계산 (`calculate_episode_metrics`)

**논문용 10개 핵심 지표**:

1. **기본 성능 지표 (6개)**:
   - **RMSE**: `_calculate_rmse()`
   - **Steady-State Error**: `_calculate_steady_state_error()` (마지막 10% 구간 평균)
   - **Rise Time**: `_calculate_rise_time()` (목표값 ±5% 밴드 내 최초 진입 시간)
   - **Settling Time**: `_calculate_settling_time()` (연속 유지 기준, 2초 연속)
   - **Overshoot**: `_calculate_overshoot()` (목표값을 넘어선 최대 편차, %)
   - **IAE**: `_calculate_iae()` (Integral Absolute Error)

2. **제어 노력 지표 (2개)**:
   - **Input RMS**: `_calculate_input_rms()` (PID gain 합의 RMS)
   - **Total Variation**: `_calculate_total_variation()` (제어 출력 변화 총량)

3. **안정성 지표 (2개)**:
   - **Band Ratio**: `_calculate_success_rate()` (목표 범위 내 유지 비율, ±2% 오차)
   - **Error Variance**: `_calculate_error_variance()` (오차 분산)

#### 5.4 지표 계산 상세

1. **`_calculate_rmse()`**
   - Root Mean Square Error
   - `sqrt(mean(errors^2))`

2. **`_calculate_steady_state_error()`**
   - 마지막 10% 구간의 평균 절대 오차

3. **`_calculate_rise_time()`**
   - 목표값 ±5% 밴드 내 최초 진입 시간

4. **`_calculate_settling_time()`**
   - 연속 유지 기준 (2초 연속)
   - 에피소드 보상과 동일한 기준

5. **`_calculate_overshoot()`**
   - 목표값보다 더 나쁜 방향으로의 최대 편차
   - 음수 목표값: `(target - extreme) / |target| * 100`
   - 양수 목표값: `(extreme - target) / |target| * 100`

6. **`_calculate_iae()`**
   - Integral Absolute Error
   - `sum(abs(errors)) * dt`

7. **`_calculate_input_rms()`**
   - PID gain 합의 RMS 값

8. **`_calculate_total_variation()`**
   - 제어 출력 변화 총량 (밸브 마모와 직결)
   - `sum(abs(diff(pi_output)))`

9. **`_calculate_success_rate()`**
   - 목표 범위 내 유지 비율
   - ±2% 오차 범위 (±0.8N)

10. **`_calculate_error_variance()`**
    - 오차 분산 (안정성 지표)

#### 5.5 데이터 저장

1. **`save_episode_metrics(episode_num)`**
   - 에피소드별 지표를 CSV로 저장
   - 개별 지표별 CSV 파일 생성
   - 예: `rmse.csv`, `overshoot.csv`, `settling_time.csv` 등

2. **`save_performance_summary()`**
   - 전체 성능 요약 저장 (논문용 10개 핵심 지표)
   - `performance_summary.csv` 생성
   - Mean, Std, Min, Max, Unit, Description 포함

#### 5.6 시각화

1. **`generate_plots()`**
   - 모든 지표 그래프 생성
   - 개별 지표 그래프 (PNG)
   - 종합 대시보드 (`comprehensive_dashboard.png`)
   - 스텝 기반 그래프 (`step_dashboard.png`)

2. **`_plot_metric(metric_name)`**
   - 개별 지표 그래프 생성
   - 에피소드별 값 플롯
   - 단위 및 설명 포함

3. **`_generate_comprehensive_dashboard()`**
   - 종합 대시보드 생성
   - 10개 지표를 한 화면에 표시
   - 서브플롯 레이아웃

4. **`_generate_step_based_plots()`**
   - 스텝 기반 그래프 생성
   - 힘 추종 곡선, 오차 시계열, 제어 입력 시계열, 보상 분석

5. **`_plot_force_tracking_curve()`**
   - 힘 추종 곡선 플롯
   - 현재 힘 vs 목표 힘

6. **`_plot_error_time_series()`**
   - 오차 시계열 플롯

7. **`_plot_control_input_series()`**
   - 제어 입력 시계열 플롯 (PI 출력)

8. **`_plot_reward_breakdown()`**
   - 보상 분석 플롯 (RewardBreakdownLogger와 연동)

9. **`_generate_step_dashboard()`**
   - 스텝 기반 종합 대시보드

#### 5.7 유틸리티 메서드

1. **`_get_metric_unit(metric_name)`**
   - 지표별 단위 반환

2. **`_get_metric_description(metric_name)`**
   - 지표별 설명 반환

3. **`reset_episode_data()`**
   - 에피소드 데이터 초기화

---

## 6. `loggers/reward_breakdown.py` - 보상 분석 로거

**파일 경로**: `py_rl/pid_gain_rl/loggers/reward_breakdown.py`  
**줄 수**: ~581줄  
**역할**: 보상 구성 요소 분석 및 시각화

### 주요 클래스: `RewardBreakdownLogger`

#### 6.1 초기화 (`__init__`)

**속성**:
- `log_dir`: 로그 디렉토리
- `reward_breakdown_dir`: 보상 분석 서브폴더
- `rows`: 스텝별 보상 데이터 리스트
- `episode_components`: 에피소드별 보상 구성 요소 리스트

#### 6.2 스텝별 로깅

1. **`log_step(episode, step, prog, in_band, edot_abs, du_abs, reward, done)`**
   - 스텝별 보상 구성 요소 로깅
   - `prog`: 진행도 (exp(-error / 5.0))
   - `in_band`: 밴드 내 여부
   - `edot_abs`: 오차 변화율 절대값
   - `du_abs`: 제어 입력 변화 절대값
   - 버퍼에 저장 (메모리)

#### 6.3 에피소드별 로깅

1. **`log_episode_components(episode, reward_score, r_centered, r_baseline, overshoot, settling_time, band_ratio, reward)`**
   - 에피소드별 보상 구성 요소 로깅
   - `reward_score`: 원시 보상 스코어 (0~1)
   - `r_centered`: 중심화된 보상
   - `r_baseline`: 보상 기준선 (EWMA)
   - `overshoot`: 오버슈트 (%)
   - `settling_time`: 정착시간 (초)
   - `band_ratio`: 밴드 유지 비율
   - `reward`: 최종 보상 (tanh 적용 후)

#### 6.4 데이터 저장

1. **`save_episode_rewards(episode_rewards)`**
   - 에피소드별 보상 CSV 저장
   - `episode_rewards.csv` 생성

2. **`save_reward_breakdown_csv()`**
   - 스텝별 보상 분석 CSV 저장
   - `reward_breakdown.csv` 생성

3. **`save_episode_components_csv()`**
   - 에피소드별 보상 구성 요소 CSV 저장
   - `episode_reward_components.csv` 생성

#### 6.5 시각화

1. **`generate_episode_reward_graph(episode_rewards)`**
   - 에피소드별 보상 그래프 생성
   - `episode_rewards.png` 생성
   - 평균 보상 선 표시

2. **`generate_reward_components_graph()`**
   - 보상 구성 요소 시각화 그래프 생성
   - `episode_reward_components.png` 생성
   - 2개 서브플롯:
     - **보상 구성 요소** (왼쪽 Y축): reward_score, r_centered, r_baseline
     - **제어 성능 지표** (오른쪽 Y축): overshoot, settling_time, band_ratio
   - Moving average (10-ep) 적용

3. **`_plot_png(start_ep, end_ep)`**
   - 스텝별 보상 분석 PNG 생성
   - 여러 서브플롯:
     - `reward_breakdown_reward_ep{start}-{end}.png`: 보상
     - `reward_breakdown_prog_ep{start}-{end}.png`: 진행도
     - `reward_breakdown_inband_ep{start}-{end}.png`: 밴드 내 여부
     - `reward_breakdown_edot_ep{start}-{end}.png`: 오차 변화율
     - `reward_breakdown_du_ep{start}-{end}.png`: 제어 입력 변화

#### 6.6 유틸리티 메서드

1. **`flush_if_needed(current_episode, force=False, episode_rewards=None)`**
   - CSV 저장 + PNG 시각화 수행
   - `force=True`: PNG 생성, `force=False`: CSV만 저장
   - 메모리 절약을 위해 `force=False`일 때는 CSV만 저장

2. **`_moving_average(data, window=10)`**
   - Moving average 계산

3. **`_write_csv_append()`**
   - CSV 파일에 추가 쓰기 (미사용)

---

## 7. `loggers/learning_done.py` - 학습 완료 로거

**파일 경로**: `py_rl/pid_gain_rl/loggers/learning_done.py`  
**줄 수**: ~22줄  
**역할**: 학습 완료 시 전체 로깅을 관리하는 클래스

### 주요 클래스: `LearningDoneLogger`

#### 7.1 초기화 (`__init__`)

**역할**: 타임스탬프 기반 learning_done 폴더 생성

**로직**:
1. 타임스탬프 생성: `learning_done_{timestamp}` (예: `learning_done_251028_17h22m`)
2. 디렉토리 생성
3. 다른 로거들이 이 폴더 안에 서브폴더 생성
   - `control_performance/`
   - `reward_breakdown/`

**사용**: `env.py`에서 한 번만 생성, 다른 로거들이 참조

---

## 📋 요약

### 유틸리티 모듈
- **`math_utils.py`**: PID 액션 스케일링 (선형/로그), 초기 상태 생성 (20차원)
- **`data_saver.py`**: 모든 데이터 저장 통합 관리
- **`signals.py`**: 안전한 종료 처리 (SIGINT, SIGTERM)

### 로거 모듈
- **`base_logger.py`**: 타임스탬프 + 아이콘 로거 (AppLogger)
- **`control_performance.py`**: 제어공학 지표 계산 및 시각화 (10개 핵심 지표)
- **`reward_breakdown.py`**: 보상 구성 요소 분석 및 시각화
- **`learning_done.py`**: 학습 완료 로그 폴더 관리

### 핵심 특징
1. **타임스탬프 기반 폴더 구조**: 학습 완료 시 한 번만 생성
2. **CSV + PNG 저장**: 데이터와 시각화 모두 저장
3. **메모리 효율**: `force=False`일 때 CSV만 저장
4. **논문용 지표**: 10개 핵심 제어공학 지표 계산
5. **보상 분석**: 스텝별 및 에피소드별 보상 구성 요소 분석

---

**완료**: 모든 모듈 기능 분석 완료! (총 16개 모듈)

