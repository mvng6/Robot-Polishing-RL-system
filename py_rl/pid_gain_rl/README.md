# PID Gain RL - 모듈화된 강화학습 시스템

## 📋 개요

공압 폴리싱 로봇의 PID 게인을 SAC 강화학습으로 최적화하는 시스템입니다.
**3876줄 → 16개 모듈**로 리팩토링하여 유지보수성과 확장성을 향상했습니다.

### ⚠️ 최근 변경사항
- 보상 스케일: 최종 보상 범위 [-1, 1] (지수 스코어 + PBRS)
  - S_ts(정착시간), S_mp(오버슈트%), S_ess(정상상태오차), S_band(밴드유지율), S_u(입력RMS)
  - reward = 2*(Σw_i S_i - w_fail P_fail + w_pbrs progress) - 1
  - 안전 위반: -1.0
- PID 범위(국소탐색): Kp(35–45), Ki(45–55), Kd(1e-6–1e-3)
  - Kd 매핑: 로그 스케일
- 탐색 제어: Actor log_std ∈ [-3, 0.5], ε-greedy(10ep마다 p=0.05)
- 네트워크: Actor/Critic 128–128 2층 MLP 경량화
- 학습률 분리: LR_ACTOR=1e-4, LR_CRITIC=2e-4 (cfg 주입 지원)
- 안전 위반 시에도 최소 transition 저장 + 단발 학습 수행
- `logging/` → `loggers/`로 디렉토리 이름 변경 (과거 변경 유지)

---

## 🚀 빠른 시작

### 방법 1: 모듈 방식 (권장)
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl
python3 -m pid_gain_rl
```

### 방법 2: 직접 실행
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl
python3 main.py
```

### 방법 3: 기존 방식
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl/experiment
python3 JY_PID_Gain_SAC_MDP_monitor_3_reset.py
```

> **⚠️ 중요**: 반드시 위 경로에서 실행하세요. 잘못된 경로에서 실행하면 import 오류가 발생합니다.

---

## 📁 파일 구조

```
pid_gain_rl/
├── __init__.py                     패키지 초기화
├── __main__.py                     엔트리 포인트 (python -m pid_gain_rl)
├── main.py                         간단한 실행 스크립트
│
├── constants.py                     상수 정의
│   └── Constants 클래스: PID 범위, 보상 가중치, 물리 상수 등
│
├── config.py                        설정 관리 (6.8KB, 165줄)
│   ├── Config (dataclass): 타입 안전한 설정
│   ├── create_config(): 설정 생성
│   └── change_episode_length(): 에피소드 길이 변경
│
├── monitor.py                       실시간 GUI (6.3KB, 181줄)
│   └── RLRealtimeMonitor: Force 추적 + 보상 그래프 (10Hz, 30초 윈도우)
│
├── comm.py                          TCP 통신 (14KB, 352줄)
│   └── PIDGainCommunicator: 1kHz 데이터 수신, CRC16 체크섬
│
├── agent.py                         SAC 에이전트
│   ├── Actor: 128–128 MLP, log_std∈[-3,0.5]
│   ├── Critic: 128–128 MLP (Dual Q)
│   ├── ReplayBuffer: 경험 버퍼
│   └── PIDGainSACAgent: 통합 에이전트
│
├── env.py                           학습 환경 ⭐️ 가장 큰 파일
│   └── PIDGainOptimizationEnvironment
│      - 보상: 지수 스코어 + PBRS → 최종 [-1,1]
│      - 안전 위반 시에도 최소 학습 수행
│
├── utils/                           유틸리티 (3개 파일)
│   ├── math_utils.py               수학 함수: scale_action_to_pid, create_initial_state
│   ├── data_saver.py               데이터 저장: DataSaver, Logger
│   └── signals.py                  시그널 처리: Ctrl+C 안전 종료
│
├── loggers/                         로깅 시스템 (4개 파일) ⚠️ logging→loggers로 변경
│   ├── base_logger.py             AppLogger: 기본 로거 (이모지 지원)
│   ├── control_performance.py     제어 성능 로깅 (897줄) ⭐️ 큰 파일
│   │   └── 10개 핵심 지표: RMSE, Overshoot, Settling Time, IAE 등
│   ├── reward_breakdown.py        보상 분석 (258줄)
│   └── learning_done.py           학습 완료 관리
│
├── experiment/                      기존 코드 (레거시 호환)
│   └── JY_PID_Gain_SAC_MDP_monitor_3_reset.py (3876줄)
│
├── experiment_logs/                 로그 저장 (자동 생성)
│   └── learning_done_YYMMDD_HHhMMm/
│       ├── control_performance/   제어 성능 지표 (CSV, PNG)
│       └── reward_breakdown/       보상 분석 그래프
│
└── saved_agents/                    모델 저장 (자동 생성)
    └── best_pid_agent_episode_*.pth
```

---

## 📦 각 파일의 역할

### 🔵 핵심 모듈

**constants.py**
- PID 범위(Kp,Ki 선형/ Kd 로그 스케일), 물리 상수
- 보상 스코어 가중치/τ/W_PBRS

**utils/** (3개 파일)
- **math_utils.py**: 수학 함수 (scale_action_to_pid, create_initial_state)
- **data_saver.py**: 데이터 저장 (DataSaver, Logger)
- **signals.py**: 시그널 처리 (Ctrl+C 안전 종료)

**loggers/** (4개 파일) ⚠️ **logging → loggers로 변경** (Python 내장과 충돌 방지)
- **base_logger.py**: AppLogger - 이모지 로거
- **control_performance.py** (897줄): 제어 성능 로깅 - 10개 핵심 지표
- **reward_breakdown.py** (258줄): 보상 분석
- **learning_done.py**: 학습 완료 관리

**config.py** (165줄)
- `Config` 데이터클래스로 설정 관리
- `create_config()`: 실행 시 설정 생성
- 딕셔너리 호환 (`to_dict()`, `from_dict()`)

**monitor.py** (181줄)
- 실시간 Force/보상 그래프 표시
- TkAgg 백엔드 (헤드리스 감지 → Agg 자동 전환)

**comm.py** (352줄)
- TCP/IP 통신 (1kHz 수신)
- CRC16 체크섬 검증
- 재연결 처리

**agent.py**
- Actor/Critic SAC 구현, LR 분리, log_std 범위 축소
- PID 게인 액션 선택

**env.py** ⭐️
- 메인 학습 루프, 보상(지수 스코어+PBRS, 최종 [-1,1])
- 10개 제어공학 지표 계산
- ε-greedy(10ep마다 p=0.05)

---

## ⚙️ 설정 변경

### __main__.py에서 설정 변경
```python
def main():
    global _global_env
    
    # ========== 설정 변경 포인트 ==========
    RECV_FREQUENCY_HZ = 1000           # 수신 주파수 (Hz)
    EPISODE_LENGTH_SECONDS = 10.0      # 에피소드 길이 (초)
    # ====================================
    ...
```

### constants.py에서 상수 변경 (예시)
```python
class Constants:
    DEFAULT_TARGET_FORCE = -40.0
    DEFAULT_EPISODES = 500
    DEFAULT_LR = 1e-4
    DEFAULT_LR_ACTOR = 1e-4
    DEFAULT_LR_CRITIC = 2e-4
    DEFAULT_PID_RANGE = {"Kp": (35.0,45.0), "Ki": (45.0,55.0), "Kd": (1e-6,1e-3)}
    SCORE_W_TS=0.30; SCORE_W_MP=0.25; SCORE_W_ESS=0.20; SCORE_W_BAND=0.15; SCORE_W_U=0.05; SCORE_W_FAIL=0.15; SCORE_W_PBRS=0.10
```

---

## 📊 출력 파일

### experiment_logs/learning_done_YYMMDD_HHhMMm/

**control_performance/** - 제어 성능 지표
- `rmse.csv`, `rmse.png` - RMSE 추이
- `overshoot.csv`, `overshoot.png` - 오버슈트 추이
- `settling_time.csv`, `settling_time.png` - 정착시간 추이
- `comprehensive_dashboard.png` - 종합 대시보드

**reward_breakdown/** - 보상 분석
- `episode_rewards.csv`, `episode_rewards.png` - 에피소드별 보상
- `reward_breakdown.csv` - 스텝별 보상 구성
- `reward_breakdown_*.png` - 5개 세부 그래프

### saved_agents/
- `best_pid_agent_episode_*_reward_*.pth` - 최고 성능 에이전트

---

## 🔍 주요 개선사항

### 리팩토링 전 ❌
- 3876줄 단일 파일
- 코드 찾기 어려움
- 유지보수 어려움
- 재사용/테스트 불가

### 리팩토링 후 ✅
- 16개 모듈로 분리
- 명확한 구조
- 독립 테스트/재사용 가능
- 기존 코드 호환

---

## 🐛 문제 해결

### ImportError: No module named 'pid_gain_rl'
**원인**: 잘못된 경로에서 실행  
**해결**:
```bash
# 올바른 경로로 이동
cd /home/katech/Robot-Polishing-RL-system/py_rl

# 방법 1 (권장)
python3 -m pid_gain_rl

# 방법 2
cd pid_gain_rl
python3 main.py
```

### AttributeError: partially initialized module 'logging'
**원인**: `logging` 디렉토리가 Python 내장 모듈과 충돌  
**해결**: ⚠️ **이미 해결됨** - `logging/` → `loggers/`로 이름 변경 완료

### ModuleNotFoundError
**원인**: 절대 경로가 sys.path에 없음  
**해결**: 반드시 위 경로에서 실행하거나 `PYTHONPATH` 설정
```bash
export PYTHONPATH=/home/katech/Robot-Polishing-RL-system/py_rl:$PYTHONPATH
```

### 연결 실패
- 로봇 PC 실행 확인
- 포트 8888 확인
- 네트워크 연결 확인

### GUI 미표시
- 헤드리스 모드 자동 감지 (정상)
- PNG 파일로 저장됨

---

## 📝 참고

- 원본: JY_PID_Gain_SAC_MDP_monitor_3_reset.py (3876줄)
- 리팩토링: 2024-10-30 (지속 업데이트)
- 구조 유지: experiment/, experiment_logs/, saved_agents/

---

**리팩토링 완료!** 🎉  
3876줄 → 16개 모듈로 분리 완료
