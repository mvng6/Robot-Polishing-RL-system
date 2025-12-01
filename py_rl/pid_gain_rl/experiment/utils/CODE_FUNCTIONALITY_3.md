# 📋 코드 기능 상세 분석 - Part 3: 유틸리티 및 로거 모듈

## 📁 모듈 목록 (experiment/utils)
- `utils/math_utils.py` → `experiment/utils/utils/math_utils.py` (액션 스케일링·초기 상태 생성)
- `utils/data_saver.py` → `experiment/utils/utils/data_saver.py` (학습 중 데이터 저장)
- `utils/signals.py` → `experiment/utils/utils/signals.py` (SIGINT/SIGTERM 안전 종료)
- `loggers/base_logger.py` → `experiment/utils/loggers/base_logger.py`
- `loggers/control_performance.py` → `experiment/utils/loggers/control_performance.py`
- `loggers/reward_breakdown.py` → `experiment/utils/loggers/reward_breakdown.py`
- `loggers/learning_done.py` → `experiment/utils/loggers/learning_done.py`

---
## 🆕 최근 포인트
- 모든 경로는 `py_rl/pid_gain_rl/experiment/utils/...`.
- STATE_DIM=10, ACTION_DIM=4에 맞춰 스케일링·초기 상태 생성 로직 정리.
- ControlPerformance/RewardBreakdown가 learning_done 하위에 CSV+PNG를 생성해 DataSaver/신호 처리 시 자동 저장.

---

## 1. `utils/math_utils.py` - 수학 유틸
**경로**: `py_rl/pid_gain_rl/experiment/utils/utils/math_utils.py`  
**역할**: [-1,1] 내부 액션을 실제 precharge/PID로 매핑하고 초기 상태를 구성.

- `scale_action_to_control(action, pid_range, precharge_range)`:  
  precharge 0.01~0.03 선형 매핑(클리핑), Kp/Ki 선형+0.01 step 양자화, Kd는 작은 범위(≤0.03)면 선형·그 외 로그 매핑 후 0.001/0.01 step 양자화(소수 3자리).  
- `scale_action_to_pid(action, pid_range)`: precharge 없이 PID만 동일 규칙으로 변환/양자화.
- `create_initial_state(force_data, target_force, ...)`: 힘 데이터가 없으면 `INITIAL_CONTACT_FORCE=-45.0`/`INITIAL_PI_OUTPUT=0.05`을 사용해 [힘, 목표, 오차, 오차미분, 오차적분, PI, prep 4채널 0]의 10차원 상태를 생성(NaN/Inf 가드 포함).

---

## 2. `utils/data_saver.py` - 데이터 저장
**경로**: `py_rl/pid_gain_rl/experiment/utils/utils/data_saver.py`  
**역할**: 학습 중간/종료 시 모든 로그를 한번에 저장.

- `Logger.log(level, message)`: 타임스탬프+아이콘 콘솔 로깅.
- `DataSaver.save_all_data(env, current_episode=None, force=True)`:  
  `RewardBreakdownLogger.flush_if_needed` 호출로 보상 CSV/PNG 저장, `ControlPerformanceLogger.save_performance_summary()/generate_plots()` 호출로 제어 지표 CSV/그래프 생성. 오류는 로깅하고 학습은 유지.

---

## 3. `utils/signals.py` - 시그널 처리
**경로**: `py_rl/pid_gain_rl/experiment/utils/utils/signals.py`  
**역할**: SIGINT/SIGTERM 시 안전 종료.

- `signal_handler(signum, frame, env=None)`: `learning_done=True` PID 패킷 전송 후 RewardBreakdown/ControlPerformance 저장, 종료.
- `install_signal_handlers()`: SIGINT/SIGTERM을 래핑해 `__main__` 또는 `pid_gain_rl.experiment.config.__main__`에 있는 `_global_env`를 찾아 handler에 전달.

---

## 4. 로그 모듈

- **`loggers/base_logger.py` (AppLogger)**: `[HH:MM:SS.mmm]` 타임스탬프 + 레벨별 아이콘(ℹ️/✅/⚠️/❌/🔍)으로 콘솔 출력.

- **`loggers/control_performance.py` (ControlPerformanceLogger)**:  
  Force/Target/PI/PID 시계열을 적산해 에피소드별 지표(RMSE, Steady-State Error, Rise/Settling Time, Overshoot, IAE, Input RMS, Total Variation, Band Ratio=Success Rate, Error Variance)를 계산/CSV 저장하고 matplotlib로 그래프 생성(기본 폰트 Times New Roman, 크기 상향). 디렉터리는 learning_done 하위 `control_performance`.

- **`loggers/reward_breakdown.py` (RewardBreakdownLogger)**:  
  스텝별 `prog/in_band_now/edot_abs/du_abs/reward`를 버퍼에 저장, episode_rewards CSV와 이동평균(기본 50) 컬럼, 에피소드 보상 PNG 생성. 학습 종료 시 `flush_if_needed`로 자동 저장. 디렉터리는 learning_done 하위 `reward_breakdown`.

- **`loggers/learning_done.py` (LearningDoneLogger)**:  
  실행 시점에 `learning_done_YYMMDD_HHhMMm` 폴더를 한 번 생성해 모든 로그·그래프의 베이스 경로로 사용.

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

3. **안정성 지표 (2개 + Success Rate 별도 노출)**:
   - **Band Ratio** / **Success Rate**: `_calculate_success_rate()` (목표 범위 내 유지 비율, ±2% 오차, `success_rate` 컬럼으로 별도 저장/그래프화)
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
   - 예: `rmse.csv`, `success_rate.csv`, `overshoot.csv`, `settling_time.csv` 등

2. **`save_performance_summary()`**
   - 전체 성능 요약 저장 (논문용 10개 핵심 지표)
   - `performance_summary.csv` 생성
   - Mean, Std, Min, Max, Unit, Description 포함

#### 5.6 시각화

1. **`generate_plots()`**
   - 모든 지표 그래프 생성
   - 개별 지표 그래프 (PNG, `success_rate` 포함)
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

1. **`save_episode_rewards(episode_rewards, ma_window=50)`**
   - 에피소드별 보상 + 이동평균(기본 50 ep) CSV 저장
   - `episode_rewards.csv` 생성 (`episode`, `reward`, `reward_ma_50`)

2. **`save_reward_breakdown_csv()`**
   - 스텝별 보상 분석 CSV 저장
   - `reward_breakdown.csv` 생성

3. **`save_episode_components_csv()`**
   - 에피소드별 보상 구성 요소 CSV 저장
   - `episode_reward_components.csv` 생성

#### 6.5 시각화

1. **`generate_episode_reward_graph(episode_rewards, ma_window=50)`**
   - 에피소드별 보상 그래프 생성 (이동평균 + 평균선 포함)
   - `episode_rewards.png` 생성 (`Episode Reward`, `Moving Avg`, `Mean`)

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

2. **`_compute_moving_average(values, window)`**
   - 이동평균 계산 (Reward 그래프용)

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
- **`math_utils.py`**: PID 액션 스케일링 (선형/로그, 0.1 양자화), 초기 상태 생성 (6차원)
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
