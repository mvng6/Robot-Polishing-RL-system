# 📁 Phase별 버전 관리 전략

## 🎯 목표

- Phase 1-5를 독립적으로 관리
- 각 Phase의 설정, 코드, 모델을 명확히 분리
- 전이학습 시 이전 Phase 모델 로드 용이
- 버전 충돌 방지 및 명확한 구조

---

## 🏗️ 추천 방법 1: Phase별 디렉토리 구조 (권장)

### 구조

```
py_rl/pid_gain_rl/
├── phases/                    # Phase별 코드 관리
│   ├── __init__.py
│   ├── base/                  # 공통 베이스 클래스
│   │   ├── __init__.py
│   │   ├── base_env.py        # 기본 Environment 클래스
│   │   ├── base_config.py     # 기본 Config 클래스
│   │   └── base_agent.py      # 기본 Agent 클래스
│   │
│   ├── phase1/                # Phase 1: 고정 목표
│   │   ├── __init__.py
│   │   ├── config.py          # Phase 1 전용 설정
│   │   ├── env.py             # Phase 1 전용 환경 (base_env 상속)
│   │   ├── main.py            # Phase 1 실행 파일
│   │   └── README.md          # Phase 1 문서
│   │
│   ├── phase2/                # Phase 2: 무작위 목표
│   │   ├── __init__.py
│   │   ├── config.py          # Phase 2 전용 설정
│   │   ├── env.py             # Phase 2 전용 환경
│   │   ├── target_manager.py  # 동적 목표 관리
│   │   ├── main.py
│   │   └── README.md
│   │
│   ├── phase3/                # Phase 3: 1회 변경
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── env.py
│   │   ├── target_manager.py
│   │   ├── main.py
│   │   └── README.md
│   │
│   ├── phase4/                # Phase 4: 2회 변경
│   │   └── ...
│   │
│   └── phase5/                # Phase 5: 3회 변경 (최종)
│       └── ...
│
├── common/                    # 공통 모듈 (모든 Phase에서 사용)
│   ├── __init__.py
│   ├── agent.py               # SAC Agent (공통)
│   ├── comm.py                # 통신 모듈 (공통)
│   ├── monitor.py             # 모니터 (공통)
│   └── utils/                 # 유틸리티 (공통)
│
├── saved_agents/              # Phase별 모델 저장
│   ├── phase1/
│   │   ├── best_model.pth
│   │   └── checkpoint_ep100.pth
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   └── phase5/
│
└── experiment_logs/            # Phase별 로그
    ├── phase1/
    ├── phase2/
    └── ...
```

### 장점

✅ **명확한 분리**: 각 Phase가 독립적인 디렉토리  
✅ **코드 중복 최소화**: 공통 모듈은 `common/`에 유지  
✅ **전이학습 용이**: `saved_agents/phase1/`에서 모델 로드  
✅ **버전 관리 쉬움**: Git에서 Phase별 커밋 분리 가능  
✅ **문서화 용이**: 각 Phase별 README 작성 가능  

### 단점

⚠️ **코드 중복 가능성**: 각 Phase에서 비슷한 코드 반복  
⚠️ **공통 수정 시**: 여러 Phase에 동시 적용 필요  

---

## 🔄 추천 방법 2: 팩토리 패턴 + 설정 파일 (유연함)

### 구조

```
py_rl/pid_gain_rl/
├── phases/
│   ├── __init__.py
│   ├── phase_factory.py       # Phase별 객체 생성 팩토리
│   ├── phase1/
│   │   ├── config.yaml        # Phase 1 설정 파일
│   │   └── env_override.py    # Phase 1 전용 오버라이드
│   ├── phase2/
│   │   ├── config.yaml
│   │   └── env_override.py
│   └── ...
│
├── configs/                   # Phase별 설정 파일
│   ├── phase1.yaml
│   ├── phase2.yaml
│   ├── phase3.yaml
│   ├── phase4.yaml
│   └── phase5.yaml
│
└── (기존 구조 유지)
```

### 구현 예시

```python
# phases/phase_factory.py
class PhaseFactory:
    @staticmethod
    def create_environment(phase: int, config_path: str = None):
        if phase == 1:
            from phases.phase1.env_override import Phase1Environment
            return Phase1Environment(config_path)
        elif phase == 2:
            from phases.phase2.env_override import Phase2Environment
            return Phase2Environment(config_path)
        # ...
    
    @staticmethod
    def create_config(phase: int):
        config_path = f"configs/phase{phase}.yaml"
        return load_config(config_path)

# 사용법
env = PhaseFactory.create_environment(phase=2)
```

### 장점

✅ **하나의 코드베이스**: 공통 코드는 유지  
✅ **설정 파일로 분리**: Phase별 설정만 YAML로 관리  
✅ **오버라이드 가능**: 필요한 부분만 오버라이드  

### 단점

⚠️ **복잡도 증가**: 팩토리 패턴 이해 필요  
⚠️ **디버깅 어려움**: 동적 생성으로 추적 어려울 수 있음  

---

## 📊 추천 방법 3: Git 브랜치 전략 (고급)

### 구조

```
main (기본)
├── phase1 (브랜치)
├── phase2 (브랜치, phase1에서 분기)
├── phase3 (브랜치, phase2에서 분기)
├── phase4 (브랜치, phase3에서 분기)
└── phase5 (브랜치, phase4에서 분기)
```

### 사용법

```bash
# Phase 1 작업
git checkout -b phase1
# 코드 작성 및 커밋
git checkout main
git merge phase1

# Phase 2 시작 (Phase 1 기반)
git checkout -b phase2 phase1
# Phase 2 코드 작성
```

### 장점

✅ **명확한 히스토리**: 각 Phase의 개발 과정 추적  
✅ **병렬 개발**: 여러 Phase 동시 작업 가능  

### 단점

⚠️ **복잡도**: Git 브랜치 관리 필요  
⚠️ **병합 충돌**: 공통 코드 수정 시 충돌 가능  

---

## 🎯 최종 권장안: 하이브리드 접근법

### 구조 (방법 1 + 방법 2 결합)

```
py_rl/pid_gain_rl/
├── phases/
│   ├── base/                  # 베이스 클래스
│   │   ├── base_env.py
│   │   └── base_config.py
│   │
│   ├── phase1/
│   │   ├── config.py          # Phase 1 설정
│   │   ├── env.py             # base_env 상속
│   │   └── main.py            # 실행 파일
│   │
│   ├── phase2/
│   │   ├── config.py
│   │   ├── env.py             # base_env 상속 + 오버라이드
│   │   ├── target_manager.py  # Phase 2 전용
│   │   └── main.py
│   │
│   └── ...
│
├── common/                    # 공통 모듈
│   ├── agent.py
│   ├── comm.py
│   └── monitor.py
│
├── saved_agents/
│   └── phase{1..5}/          # Phase별 모델
│
└── experiment_logs/
    └── phase{1..5}/           # Phase별 로그
```

### 실행 방법

```bash
# Phase 1 실행
python3 -m phases.phase1.main

# Phase 2 실행 (Phase 1 모델 로드)
python3 -m phases.phase2.main --load-model saved_agents/phase1/best_model.pth

# Phase 3 실행
python3 -m phases.phase3.main --load-model saved_agents/phase2/best_model.pth
```

---

## 🛠️ 구현 단계

### Step 1: 베이스 클래스 생성

```python
# phases/base/base_env.py
from common.agent import PIDGainSACAgent
from common.comm import PIDGainCommunicator

class BaseEnvironment:
    """모든 Phase의 공통 기능"""
    def __init__(self, config):
        self.config = config
        self.agent = PIDGainSACAgent(config)
        self.comm = PIDGainCommunicator(...)
    
    def run_training(self):
        """공통 학습 루프"""
        # 기본 구현
        pass

# phases/base/base_config.py
class BaseConfig:
    """공통 설정"""
    def __init__(self, phase: int):
        self.phase = phase
        # 공통 설정
```

### Step 2: Phase 1 구현

```python
# phases/phase1/env.py
from phases.base.base_env import BaseEnvironment

class Phase1Environment(BaseEnvironment):
    """Phase 1: 고정 목표"""
    def __init__(self, config):
        super().__init__(config)
        # Phase 1 전용 초기화
    
    def get_target_force(self, episode_time: float) -> float:
        """고정 목표 반환"""
        return self.config.target_force
    
    def run_training(self):
        """Phase 1 전용 학습 루프"""
        # 오버라이드하여 Phase 1 로직 구현
        super().run_training()
```

### Step 3: Phase 2 구현

```python
# phases/phase2/env.py
from phases.base.base_env import BaseEnvironment
from phases.phase2.target_manager import DynamicTargetManager

class Phase2Environment(BaseEnvironment):
    """Phase 2: 무작위 목표"""
    def __init__(self, config):
        super().__init__(config)
        self.target_manager = DynamicTargetManager(
            phase=2,
            target_min=config.target_force_min,
            target_max=config.target_force_max,
        )
    
    def get_target_force(self, episode_time: float) -> float:
        """에피소드별 무작위 목표"""
        return self.target_manager.get_current_target(episode_time)
```

---

## 📝 Phase별 설정 관리

### 설정 파일 예시 (YAML)

```yaml
# configs/phase1.yaml
phase: 1
name: "Fixed Target Force"

episode:
  length_seconds: 10.0
  target_force: -40.0
  target_force_min: null
  target_force_max: null

network:
  state_dim: 20
  action_dim: 3
  hidden_dim: 128

training:
  episodes: 500
  batch_size: 64
  lr_actor: 1e-4
  lr_critic: 2e-4

model:
  save_dir: "saved_agents/phase1"
  load_from: null  # 처음부터 학습

logs:
  dir: "experiment_logs/phase1"
```

```yaml
# configs/phase2.yaml
phase: 2
name: "Random Target Per Episode"

episode:
  length_seconds: 10.0
  target_force: null  # 무작위 사용
  target_force_min: -50.0
  target_force_max: -35.0

network:
  state_dim: 22  # +2 (목표 정보)
  action_dim: 3
  hidden_dim: 128

training:
  episodes: 500
  batch_size: 64

model:
  save_dir: "saved_agents/phase2"
  load_from: "saved_agents/phase1/best_model.pth"  # 전이학습

logs:
  dir: "experiment_logs/phase2"
```

---

## 🔄 전이학습 통합

### Phase 전환 시 자동 모델 로드

```python
# phases/phase_factory.py
import os
from phases.phase2.config import Phase2Config
from phases.phase2.env import Phase2Environment

def create_phase2_environment(config_path: str = None):
    config = Phase2Config(config_path)
    
    # 전이학습: Phase 1 모델 로드
    if config.model.load_from and os.path.exists(config.model.load_from):
        print(f"🔄 Phase 1 모델 로드: {config.model.load_from}")
        env = Phase2Environment(config)
        env.agent.load_checkpoint(config.model.load_from)
        return env
    else:
        print("⚠️ Phase 1 모델 없음, 처음부터 학습")
        return Phase2Environment(config)
```

---

## 📊 Phase별 독립 실행 스크립트

### 각 Phase별 main.py

```python
# phases/phase1/main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phases.phase1.config import Phase1Config
from phases.phase1.env import Phase1Environment
from phases.base.base_config import create_config

def main():
    # Phase 1 설정 로드
    config = Phase1Config()
    config_dict = config.to_dict()
    
    # 환경 생성
    env = Phase1Environment(config_dict)
    
    # 학습 실행
    env.run_training()
    
if __name__ == "__main__":
    main()
```

```python
# phases/phase2/main.py
from phases.phase2.config import Phase2Config
from phases.phase2.env import Phase2Environment

def main():
    # Phase 2 설정 (Phase 1 모델 자동 로드)
    config = Phase2Config()
    env = Phase2Environment(config.to_dict())
    
    # Phase 1 모델 로드 (설정 파일에 정의됨)
    if config.model.load_from:
        env.agent.load_checkpoint(config.model.load_from)
        print(f"✅ Phase 1 모델 로드 완료")
    
    env.run_training()

if __name__ == "__main__":
    main()
```

---

## 🚀 실행 방법

### Phase 1 실행

```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl
python3 -m pid_gain_rl.phases.phase1.main
```

### Phase 2 실행 (Phase 1 모델 로드)

```bash
python3 -m pid_gain_rl.phases.phase2.main
# 자동으로 saved_agents/phase1/best_model.pth 로드
```

### 명령줄 옵션

```bash
# Phase 2 직접 지정
python3 -m pid_gain_rl.phases.phase2.main \
    --load-model saved_agents/phase1/best_model.pth \
    --episodes 500

# 설정 파일 지정
python3 -m pid_gain_rl.phases.phase2.main \
    --config configs/phase2_custom.yaml
```

---

## 📁 디렉토리 생성 스크립트

### 초기 구조 생성

```bash
# create_phase_structure.sh
#!/bin/bash

cd /home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl

# Phase 디렉토리 생성
mkdir -p phases/{base,phase1,phase2,phase3,phase4,phase5}
mkdir -p phases/base
mkdir -p configs
mkdir -p saved_agents/{phase1,phase2,phase3,phase4,phase5}
mkdir -p experiment_logs/{phase1,phase2,phase3,phase4,phase5}

# __init__.py 생성
touch phases/__init__.py
touch phases/base/__init__.py
for i in {1..5}; do
    touch phases/phase$i/__init__.py
done

echo "✅ Phase 구조 생성 완료!"
```

---

## ✅ 최종 권장사항

**추천 방법: 하이브리드 접근법 (방법 1 + 설정 파일)**

1. ✅ **명확한 분리**: 각 Phase별 디렉토리
2. ✅ **코드 재사용**: 베이스 클래스 상속
3. ✅ **설정 파일**: YAML로 Phase별 설정 관리
4. ✅ **전이학습**: 설정 파일에 이전 Phase 모델 경로
5. ✅ **독립 실행**: 각 Phase별 main.py

이 구조로 Phase 1-5를 깔끔하게 관리할 수 있습니다!



