# ✅ 실행 전 체크리스트

## 📋 코드 수정 완료 확인

### ✅ 완료된 수정 사항
- [x] `constants.py`: PID 범위, 보상 가중치, 탐색 설정 변경
- [x] `agent.py`: target_entropy 동적 조정, warm_start_buffer, log_exploration_metrics 추가
- [x] `env.py`: 초기 구간 피크 패널티 추가 및 메서드 호출 통합
- [x] `loggers/reward_breakdown.py`: 보상 구성 요소 시각화 추가
- [x] Linter 오류 없음 확인

---

## 🔍 실행 전 확인 사항

### 1. 의존성 확인
- [ ] `numpy`, `torch`, `matplotlib` 설치 확인
- [ ] `scipy` (선택적): Warm-start LHS 샘플링용
  - 없어도 실행 가능 (랜덤 샘플링으로 대체)
  - 설치 권장: `pip install scipy`

### 2. 설정 확인
- [ ] `config.py` 또는 실행 시 설정값 확인:
  - `EPISODE_SECONDS`: 에피소드 길이 (기본 10초)
  - `TARGET_FORCE`: 목표 힘 (기본 -40N)
  - `EPISODES`: 학습 에피소드 수
  - `PID_RANGE`: PID 범위 (constants.py 기본값 사용)

### 3. 하드웨어 연결 확인
- [ ] 로봇 제어 PC와 통신 연결 확인
- [ ] `comm.py`의 호스트/포트 설정 확인

---

## 🚀 실행 방법

### 기본 실행
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl/pid_gain_rl
python -m py_rl.pid_gain_rl
```

또는

```bash
python main.py
```

---

## 📊 실행 후 확인 사항

### 1. 초기 로그 확인
- [ ] Warm-start 버퍼 초기화 로그 확인:
  ```
  🔥 Warm-start: 50개 샘플로 버퍼 초기화 중...
  ✅ Warm-start 완료: 50개 transition
  ```

### 2. 학습 진행 확인
- [ ] Target Entropy 동적 조정 로그 (20 에피소드마다):
  ```
  📊 탐색 메트릭 [Ep 20]: std_ratio=XX.X%, Kd_coverage=XX.X%
  ```
- [ ] 초기 구간 피크 패널티 계산 확인 (로그에서 확인)

### 3. 출력 파일 확인 (학습 종료 후)
- [ ] `reward_breakdown/episode_reward_components.png`: 보상 구성 요소 그래프
- [ ] `reward_breakdown/episode_reward_comparison.png`: 보상 비교 그래프
- [ ] `reward_breakdown/episode_reward_components.csv`: 보상 구성 요소 데이터

---

## ⚠️ 주의 사항

### 1. Warm-start (scipy 없을 때)
- scipy가 없으면 랜덤 샘플링으로 대체됨
- 경고 메시지: `⚠️ scipy 없음, 랜덤 샘플링 사용`
- 기능은 정상 작동하지만 LHS가 더 효율적

### 2. 첫 에피소드
- 첫 에피소드는 `r_baseline` 초기화로 `r_centered`가 0에 가까울 수 있음
- 정상 동작 (EWMA 기준선이 천천히 추적)

### 3. 초기 탐색 단계
- 최소 버퍼 크기(32개) 도달 전까지는 학습하지 않음
- 정상 동작 (데이터 수집 단계)

---

## 🐛 문제 발생 시

### 1. Import 오류
- `ModuleNotFoundError`: 필요한 패키지 설치
  ```bash
  pip install numpy torch matplotlib
  ```

### 2. 통신 오류
- `comm.py`의 호스트/포트 확인
- 로봇 제어 PC 연결 상태 확인

### 3. 메모리 오류
- `REPLAY_BUFFER_SIZE` 줄이기 (constants.py)
- `BATCH_SIZE` 줄이기 (config.py)

---

## 📈 성공적인 실행 확인

### 학습이 잘 진행되고 있다면:
1. ✅ 에피소드별 보상이 점진적으로 개선 (노이즈 있음)
2. ✅ 탐색 메트릭이 적절한 범위 (std_ratio 10~30%)
3. ✅ 오버슈트가 점진적으로 감소
4. ✅ 보상 구성 요소 그래프에서 `reward_score` 상승 추세

### 학습이 안 되고 있다면:
1. ❌ `reward_score`가 하락하거나 변동 없음
2. ❌ `r_baseline`이 `reward_score`를 빠르게 추적 (중심화 부작용)
3. ❌ 오버슈트가 개선되지 않음
4. → 보상 구성 요소 그래프로 원인 분석 가능!

---

**모든 준비가 완료되었습니다! 🚀**

