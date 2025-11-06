"""
SAC Agent - Actor, Critic, ReplayBuffer, PIDGainSACAgent
- Actor/Critic: 128–128 2층 MLP, Actor log_std ∈ [-2.5, -0.3] (새 PID 범위 탐색 강화)
- 학습률 분리: LR_ACTOR(1e-4), LR_CRITIC(2e-4) 지원
- 🆕 세그먼트 분할 학습 지원
- 🆕 표준편차 Annealing 지원
- 🆕 Target Entropy 동적 조정 (초기 100ep 공격적 탐색)
- 🆕 Warm-start 버퍼 초기화 (LHS 샘플링)
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from ..config.constants import Constants
from ..utils.utils.math_utils import scale_action_to_pid

class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        log_std_min=-2.5,  # 하한 보장
        log_std_max=-0.3,  # -1.0 → -0.3 (새 PID 범위 탐색 강화)
    ):
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        """Fine-tuning용 안정적 가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 🔥 0.5 → 0.05 (Fine-tuning용)
                nn.init.orthogonal_(m.weight, gain=Constants.ACTOR_WEIGHT_GAIN)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        mean = torch.clamp(mean, -10.0, 10.0)
        log_std = torch.clamp(
            self.log_std_head(x), self.log_std_min, self.log_std_max
        )
        return mean, log_std

    def sample(self, state, std_scale=1.0):
        """
        🆕 표준편차 스케일링 추가 (annealing용)
        Args:
            state: 상태 벡터
            std_scale: 표준편차 스케일 (0.3~1.0)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp() * std_scale  # 🔥 스케일 적용
        
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print(f"⚠️ [경고] Actor 출력에 NaN 감지: mean={mean}, std={std}")
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std) * 0.5
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=Constants.ACTOR_WEIGHT_GAIN)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        return q1, q2

class ReplayBuffer:
    def __init__(self, capacity=None):
        self.buffer = deque(
            maxlen=capacity or Constants.DEFAULT_REPLAY_BUFFER_SIZE
        )

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            state,
            action.reshape(-1, 1) if action.ndim == 1 else action,
            reward,
            next_state,
            done,
        )

    def __len__(self):
        return len(self.buffer)

class PIDGainSACAgent:
    def __init__(self, cfg=None):
        if cfg is None:
            raise ValueError("cfg 파라미터가 필요합니다")
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        s_dim, a_dim, hidden = (
            cfg["STATE_DIM"],
            cfg["ACTION_DIM"],
            cfg["HIDDEN"],
        )
        self.gamma, self.tau = cfg["GAMMA"], cfg["TAU"]
        self.alpha = Constants.ACTOR_INITIAL_ALPHA  # 0.1 (업데이트됨)
        self.auto_entropy_tuning = cfg["AUTO_ENTROPY"]
        
        # 🆕 표준편차 Annealing 상태
        self.current_episode = 0
        self.std_scale = Constants.STD_ANNEAL_INITIAL  # 1.0
        
        # 🆕 Target Entropy 동적 조정 준비
        self.action_dim_for_entropy = a_dim
        
        self.actor = Actor(
            s_dim, a_dim, hidden, 
            log_std_max=Constants.ACTOR_LOG_STD_MAX,
            log_std_min=Constants.ACTOR_LOG_STD_MIN
        ).to(self.device)
        self.critic = Critic(s_dim, a_dim, hidden).to(self.device)
        self.critic_target = Critic(s_dim, a_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        lr_actor = cfg.get("LR_ACTOR", cfg["LR"])
        lr_critic = cfg.get("LR_CRITIC", cfg["LR"])
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        if self.auto_entropy_tuning:
            # 🆕 Target Entropy 동적 조정: 초기 100ep는 더 공격적 탐색
            self.target_entropy_initial = Constants.TARGET_ENTROPY_INITIAL_FACTOR * a_dim  # -3.6 (3차원)
            self.target_entropy_final = Constants.TARGET_ENTROPY_FINAL_FACTOR * a_dim    # -3.0 (3차원)
            self.target_entropy_transition_episodes = Constants.TARGET_ENTROPY_TRANSITION_EPISODES
            self.target_entropy = self.target_entropy_initial  # 초기값: 더 공격적
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device
            )
            self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg["LR"])
        
        self.replay = ReplayBuffer(
            cfg.get("REPLAY_BUFFER_SIZE", Constants.DEFAULT_REPLAY_BUFFER_SIZE)
        )
        self.total_steps = 0
        self.episode_rewards = []
        self.max_rewards_history = cfg.get(
            "MAX_EPISODE_REWARDS_HISTORY",
            Constants.DEFAULT_MAX_REWARDS_HISTORY,
        )
        
        # 🆕 탐색 메트릭 추적
        self.recent_actions = []  # 최근 액션 히스토리 (탐색 메트릭용)
        self.max_recent_actions = 100  # 최대 100개 유지

    def update_std_scale(self, episode_num):
        """
        🆕 표준편차 Annealing 업데이트
        에피소드 진행에 따라 탐험 강도 감소
        """
        self.current_episode = episode_num
        
        if episode_num < Constants.STD_ANNEAL_START_EPISODE:
            self.std_scale = Constants.STD_ANNEAL_INITIAL
        elif episode_num >= Constants.STD_ANNEAL_END_EPISODE:
            self.std_scale = Constants.STD_ANNEAL_FINAL
        else:
            # 선형 감소
            progress = (episode_num - Constants.STD_ANNEAL_START_EPISODE) / \
                      (Constants.STD_ANNEAL_END_EPISODE - Constants.STD_ANNEAL_START_EPISODE)
            self.std_scale = Constants.STD_ANNEAL_INITIAL + \
                            progress * (Constants.STD_ANNEAL_FINAL - Constants.STD_ANNEAL_INITIAL)
    
    def update_target_entropy(self, episode_num):
        """
        🆕 Target Entropy 동적 조정
        초기 100ep 동안 더 공격적 탐색을 위해 target_entropy를 점진적으로 조정
        """
        if not self.auto_entropy_tuning:
            return
        
        if episode_num < self.target_entropy_transition_episodes:
            # 초기 100ep: 점진적으로 전환
            progress = episode_num / self.target_entropy_transition_episodes
            self.target_entropy = (
                self.target_entropy_initial + 
                progress * (self.target_entropy_final - self.target_entropy_initial)
            )
        else:
            # 이후: 최종값 유지
            self.target_entropy = self.target_entropy_final

    def select_action(self, state, evaluate=False):
        """
        🔄 수정: 표준편차 스케일 적용
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
                log_prob = None
            else:
                # 🔥 std_scale 전달
                action, log_prob = self.actor.sample(state, std_scale=self.std_scale)

        action_np = action.cpu().numpy().flatten()
        pid_gains = scale_action_to_pid(action_np, self.cfg["PID_RANGE"])
        
        # 🆕 탐색 메트릭 추적 (최근 액션 저장)
        if not evaluate:
            self.recent_actions.append(action_np.copy())
            if len(self.recent_actions) > self.max_recent_actions:
                self.recent_actions.pop(0)  # 오래된 것 제거

        if log_prob is not None:
            log_prob = log_prob.cpu().numpy()
            return pid_gains, log_prob
        else:
            return pid_gains, None

    def select_action_random(self):
        """안전 위반 시 사용할 랜덤 PID gain 선택"""
        action_np = np.random.uniform(-1.0, 1.0, size=3)
        pid_gains = scale_action_to_pid(action_np, self.cfg["PID_RANGE"])
        return pid_gains

    def store_transition(self, state, action, reward, next_state, done):
        """
        🔄 수정: 세그먼트별 transition 저장
        Args:
            state: 현재 상태 (STATE_DIM 차원) [0-(STATE_BASE_DIM-1): 기존, STATE_BASE_DIM-(STATE_DIM-1): 궤적 요약]
            action: PID gain 액션 [Kp, Ki, Kd]
            reward: 세그먼트 보상
            next_state: 다음 상태 (STATE_DIM 차원)
            done: 세그먼트 종료 여부
        """
        state_arr = np.array(state, dtype=np.float32)
        next_state_arr = np.array(next_state, dtype=np.float32)
        action_arr = np.array(action, dtype=np.float32)
        
        if (np.isnan(state_arr).any() or np.isinf(state_arr).any() or
            np.isnan(next_state_arr).any() or np.isinf(next_state_arr).any() or
            np.isnan(action_arr).any() or np.isinf(action_arr).any() or
            np.isnan(reward) or np.isinf(reward)):
            print(f"❌ [오류] 리플레이 버퍼 저장 실패 - NaN/Inf 검출!")
            return
        
        if reward < -100.0 or reward > 50.0:
            print(f"⚠️ [경고] 비정상 보상 클리핑: {reward:.2f} → ", end="")
            reward = np.clip(reward, -100.0, 50.0)
            print(f"{reward:.2f}")
        
        norm_action = self._normalize_pid_action(action)
        
        if np.isnan(norm_action).any() or np.isinf(norm_action).any():
            print(f"❌ [오류] 정규화된 액션에 NaN/Inf 검출!")
            return
        
        self.replay.push(state_arr, norm_action, reward, next_state_arr, done)

    def _normalize_pid_action(self, pid_action):
        """PID gain을 [-1, 1] 범위로 정규화"""
        def normalize_single(v, lo, hi):
            if abs(hi - lo) < 1e-9:
                return 0.0
            v = np.clip(v, lo, hi)
            normalized = 2.0 * (v - lo) / (hi - lo) - 1.0
            if np.isnan(normalized) or np.isinf(normalized):
                print(f"❌ [오류] 정규화 결과 비정상: v={v}, lo={lo}, hi={hi}")
                return 0.0
            return normalized

        return np.array(
            [
                normalize_single(pid_action[0], *self.cfg["PID_RANGE"]["Kp"]),
                normalize_single(pid_action[1], *self.cfg["PID_RANGE"]["Ki"]),
                normalize_single(pid_action[2], *self.cfg["PID_RANGE"]["Kd"]),
            ],
            dtype=np.float32,
        )

    def update_parameters_one_step(self, batch_size=None, num_updates=128):
        """한 스텝 MDP에 최적화된 SAC 업데이트"""
        bs = min(batch_size or self.cfg["BATCH_SIZE"], len(self.replay))

        if len(self.replay) < 2:
            return

        for _ in range(num_updates):
            s, a, r, ns, d = self.replay.sample(bs)
            s = torch.FloatTensor(s).to(self.device)
            a = torch.FloatTensor(a).to(self.device)
            r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
            ns = torch.FloatTensor(ns).to(self.device)
            d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
            
            if (torch.isnan(s).any() or torch.isnan(a).any() or 
                torch.isnan(r).any() or torch.isinf(r).any()):
                print(f"⚠️ [경고] 배치 데이터에 NaN/Inf 발견 - 업데이트 건너뜀")
                continue
            
            r = torch.clamp(r, -100.0, 50.0)

            with torch.no_grad():
                y = r

            q1, q2 = self.critic(s, a)
            
            if torch.isnan(q1).any() or torch.isnan(q2).any():
                print(f"⚠️ [경고] Q 값에 NaN/Inf 발견 - Critic 업데이트 건너뜀")
                continue
            
            q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            
            if torch.isnan(q_loss) or torch.isinf(q_loss):
                print(f"⚠️ [경고] Critic loss가 비정상입니다: {q_loss.item()}")
                continue
            
            self.critic_opt.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 2.0)
            self.critic_opt.step()

            pi, logp = self.actor.sample(s, std_scale=self.std_scale)
            q1_pi, q2_pi = self.critic(s, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            pi_loss = ((self.alpha * logp) - min_q_pi).mean()
            
            if torch.isnan(pi_loss) or torch.isinf(pi_loss):
                print(f"⚠️ [경고] Actor loss가 비정상입니다: {pi_loss.item()}")
                continue
            
            self.actor_opt.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 2.0)
            self.actor_opt.step()

            if self.auto_entropy_tuning:
                logp_entropy = logp.squeeze(1)
                a_loss = -(
                    self.log_alpha
                    * (logp_entropy + self.target_entropy).detach()
                ).mean()
                self.alpha_opt.zero_grad()
                a_loss.backward()
                self.alpha_opt.step()
                self.alpha = self.log_alpha.exp()

            with torch.no_grad():
                for tp, lp in zip(
                    self.critic_target.parameters(), self.critic.parameters()
                ):
                    tp.data.copy_(
                        self.tau * lp.data + (1 - self.tau) * tp.data
                    )

    def warm_start_buffer(self, num_samples=None):
        """
        🆕 라틴 하이퍼큐브 샘플링으로 버퍼 초기화
        새 PID 범위에서 골고루 샘플링하여 버퍼 품질 확보
        """
        if num_samples is None:
            num_samples = Constants.WARM_START_NUM_SAMPLES
        
        print(f"🔥 Warm-start: {num_samples}개 샘플로 버퍼 초기화 중...")
        
        try:
            from scipy.stats import qmc
            use_lhs = True
        except ImportError:
            print("⚠️ scipy 없음, 랜덤 샘플링 사용")
            use_lhs = False
        
        pid_range = self.cfg["PID_RANGE"]
        
        if use_lhs:
            # LHS 샘플링
            sampler = qmc.LatinHypercube(d=3)
            samples = sampler.random(n=num_samples)
            
            # Kp, Ki: 선형, Kd: 로그
            kp_samples = samples[:, 0] * (pid_range["Kp"][1] - pid_range["Kp"][0]) + pid_range["Kp"][0]
            ki_samples = samples[:, 1] * (pid_range["Ki"][1] - pid_range["Ki"][0]) + pid_range["Ki"][0]
            kd_log_samples = samples[:, 2] * (np.log10(pid_range["Kd"][1]) - np.log10(pid_range["Kd"][0])) + np.log10(pid_range["Kd"][0])
            kd_samples = 10 ** kd_log_samples
        else:
            # 랜덤 샘플링 (scipy 없을 때)
            kp_samples = np.random.uniform(pid_range["Kp"][0], pid_range["Kp"][1], num_samples)
            ki_samples = np.random.uniform(pid_range["Ki"][0], pid_range["Ki"][1], num_samples)
            kd_log_samples = np.random.uniform(
                np.log10(pid_range["Kd"][0]), 
                np.log10(pid_range["Kd"][1]), 
                num_samples
            )
            kd_samples = 10 ** kd_log_samples
        
        # 각 샘플에 대해 더미 transition 저장
        for kp, ki, kd in zip(kp_samples, ki_samples, kd_samples):
            # 더미 상태 생성
            dummy_state = np.zeros(self.cfg["STATE_DIM"], dtype=np.float32)
            dummy_next_state = np.zeros(self.cfg["STATE_DIM"], dtype=np.float32)
            
            # PID를 액션으로 변환 (scale_action_to_pid의 역함수)
            kp_norm = 2.0 * (kp - pid_range["Kp"][0]) / (pid_range["Kp"][1] - pid_range["Kp"][0]) - 1.0
            ki_norm = 2.0 * (ki - pid_range["Ki"][0]) / (pid_range["Ki"][1] - pid_range["Ki"][0]) - 1.0
            kd_log = np.log10(max(kd, 1e-12))
            kd_log_min = np.log10(pid_range["Kd"][0])
            kd_log_max = np.log10(pid_range["Kd"][1])
            kd_norm = 2.0 * (kd_log - kd_log_min) / (kd_log_max - kd_log_min) - 1.0
            
            dummy_action = np.array([kp_norm, ki_norm, kd_norm], dtype=np.float32)
            dummy_reward = 0.0
            dummy_done = False
            
            self.replay.push(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
        
        print(f"✅ Warm-start 완료: {len(self.replay)}개 transition (목표: {num_samples})")
    
    def log_exploration_metrics(self, episode_num):
        """
        🆕 탐색 효과 모니터링
        action std/range 비율 및 Kd decade 커버리지 계산
        """
        if len(self.recent_actions) < 20:
            return None
        
        recent_actions = np.array(self.recent_actions[-20:])  # 최근 20개
        pid_range = self.cfg["PID_RANGE"]
        
        # Action std/range 비율 계산
        action_std = np.std(recent_actions, axis=0)
        action_range = np.array([2.0, 2.0, 2.0])  # [-1, 1] 범위
        std_ratio_pct = (action_std / action_range * 100).mean()  # 평균
        
        # Kd decade 커버리지
        # PID로 변환하여 Kd 값 추출
        kd_values = []
        for action in recent_actions:
            pid_gains = scale_action_to_pid(action, pid_range)
            kd_values.append(pid_gains[2])
        
        kd_log = np.log10(np.array(kd_values))
        kd_log_range = np.log10(pid_range["Kd"][1]) - np.log10(pid_range["Kd"][0])
        num_bins = 7  # 6개 구간
        kd_bins = np.linspace(np.log10(pid_range["Kd"][0]), np.log10(pid_range["Kd"][1]), num_bins + 1)
        bin_indices = np.digitize(kd_log, kd_bins)
        unique_bins = len(np.unique(bin_indices))
        coverage_pct = (unique_bins / num_bins) * 100
        
        # Target 값
        target_std_ratio = 3.0 if episode_num > 100 else 15.0
        
        metrics = {
            "episode": episode_num,
            "action_std_ratio_pct": float(std_ratio_pct),
            "kd_coverage_pct": float(coverage_pct),
            "target_std_ratio": target_std_ratio,
            "num_recent_actions": len(recent_actions),
        }
        
        return metrics

    def save_model(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "total_steps": self.total_steps,
                "episode_rewards": self.episode_rewards,
                "cfg": self.cfg,
            },
            path,
        )
        print(f"💾 Saved: {path}")

    def load_model(self, path, strict=True):
        if not os.path.exists(path):
            print(f"⚠️ 모델 파일이 존재하지 않음: {path}")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"], strict=strict)
            self.critic.load_state_dict(checkpoint["critic"], strict=strict)
            self.critic_target.load_state_dict(checkpoint["critic_target"], strict=strict)
            
            if "actor_opt" in checkpoint:
                self.actor_opt.load_state_dict(checkpoint["actor_opt"])
            if "critic_opt" in checkpoint:
                self.critic_opt.load_state_dict(checkpoint["critic_opt"])
                
            if "total_steps" in checkpoint:
                self.total_steps = checkpoint["total_steps"]
            if "episode_rewards" in checkpoint:
                self.episode_rewards = checkpoint["episode_rewards"]
                
            print(f"✅ 모델 로드 완료: {path}")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False

    def transfer_learning_setup(self, source_model_path, learning_rate_scale=0.1):
        """
        전이학습 설정
        Args:
            source_model_path: 소스 모델 경로
            learning_rate_scale: 학습률 스케일링 (0.1 = 기존의 10%)
        """
        if not self.load_model(source_model_path, strict=False):
            print("⚠️ 전이학습 실패: 소스 모델 로드 불가")
            return False
            
        # 학습률 조정 (전이학습 시 더 낮은 학습률 사용)
        original_lr = self.cfg["LR"]
        new_lr = original_lr * learning_rate_scale
        
        for param_group in self.actor_opt.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.critic_opt.param_groups:
            param_group['lr'] = new_lr
            
        print(f"🔄 전이학습 설정 완료")
        print(f"📈 학습률 조정: {original_lr:.2e} → {new_lr:.2e} (x{learning_rate_scale})")
        return True