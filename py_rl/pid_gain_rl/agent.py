"""
SAC Agent - Actor, Critic, ReplayBuffer, PIDGainSACAgent
- Actor/Critic: 128–128 2층 MLP, Actor log_std ∈ [-3, 0.5]
- 학습률 분리: LR_ACTOR(1e-4), LR_CRITIC(2e-4) 지원 (cfg에서 주입)
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from .constants import Constants
from .utils.math_utils import scale_action_to_pid

class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        log_std_min=-3,
        log_std_max=0.5,
    ):  # fine-tuning 단계용 경량/저탐색 설정
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        """균형잡힌 가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal 초기화 (안정적 + 적절한 탐색)
                nn.init.orthogonal_(m.weight, gain=0.5)  # 0.01 → 0.5 (적절한 탐색 허용)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        # Mean 제한 (안정성 강화: 극단값 방지)
        mean = torch.clamp(mean, -10.0, 10.0)
        log_std = torch.clamp(
            self.log_std_head(x), self.log_std_min, self.log_std_max
        )
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # NaN 체크
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print(f"⚠️ [경고] Actor 출력에 NaN 감지: mean={mean}, std={std}")
            # 안전한 기본값 반환
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
                # Orthogonal 초기화 (안정적 + 적절한 탐색)
                nn.init.orthogonal_(m.weight, gain=0.5)  # 0.01 → 0.5 (적절한 탐색 허용)
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

# ========== Replay Buffer ==========
# 경험 버퍼: 현재 상태: St, 액션: 선택한 PID 게인, 보상: 스텝에서 얻은 보상, next_state: St+1, done: 에피소드 종료 여부
# 이 정보들이 학습 시 무작위로 샘플링되어 gradient update에 사용됨
class ReplayBuffer:
    def __init__(self, capacity=None):
        self.buffer = deque(
            maxlen=capacity or Constants.DEFAULT_REPLAY_BUFFER_SIZE
        )

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
# 버퍼에서 중복 없이 무작위로 batch_size 개의 샘플 추출
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

# ========== SAC Agent ==========

class PIDGainSACAgent:
    def __init__(self, cfg=None):
        if cfg is None:
            raise ValueError("cfg 파라미터가 필요합니다")
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        s_dim, a_dim, hidden = (
            cfg["STATE_DIM"],
            cfg["ACTION_DIM"],
            cfg["HIDDEN"],
        )
        self.gamma, self.tau = cfg["GAMMA"], cfg["TAU"]
        self.alpha = 0.2  # 탐색 증가: 0.05 → 0.2 (극단적 행동 완화)
        self.auto_entropy_tuning = cfg["AUTO_ENTROPY"]
        self.actor = Actor(s_dim, a_dim, hidden).to(self.device)
        self.critic = Critic(s_dim, a_dim, hidden).to(self.device)
        self.critic_target = Critic(s_dim, a_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # LR 분리 지원 (없으면 기존 LR 사용)
        lr_actor = cfg.get("LR_ACTOR", cfg["LR"])
        lr_critic = cfg.get("LR_CRITIC", cfg["LR"]) 
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.tensor([a_dim], device=self.device)
            ).item()
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

    def select_action(self, state, evaluate=False):
        """
        PID gain 액션 선택 (에피소드당 한 번)
        Args:
            state: 12차원 상태 벡터 [0-5: 로봇제어PC 전송, 6-11: 강화학습PC 계산]
            evaluate: 평가 모드 여부
        Returns:
            pid_gains: [Kp, Ki, Kd] 실제 PID gain 값들
            log_prob: 로그 확률 (학습용)
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
                log_prob = None
            else:
                action, log_prob = self.actor.sample(state)

        action_np = action.cpu().numpy().flatten()
        pid_gains = scale_action_to_pid(action_np, self.cfg["PID_RANGE"])

        if log_prob is not None:
            log_prob = log_prob.cpu().numpy()
            return pid_gains, log_prob
        else:
            return pid_gains, None

    def select_action_random(self):
        """
        안전 위반 시 사용할 랜덤 PID gain 선택
        Returns:
            pid_gains: [Kp, Ki, Kd] 랜덤 PID gain 값들
        """
        action_np = np.random.uniform(-1.0, 1.0, size=3)
        pid_gains = scale_action_to_pid(action_np, self.cfg["PID_RANGE"])
        return pid_gains

    def store_transition(self, state, action, reward, next_state, done):
        """
        PID gain transition 저장 (에피소드당 한 개)
        Args:
            state: 초기 상태 (12차원) [0-5: 로봇제어PC 전송, 6-11: 강화학습PC 계산]
            action: PID gain 액션 [Kp, Ki, Kd] (3차원 그대로 사용)
            reward: 에피소드 총보상
            next_state: 최종 상태 (요약 또는 zero)
            done: 에피소드 종료 여부 (항상 True)
        """
        # 🔍 저장 전 데이터 검증 (NaN/Inf 체크)
        state_arr = np.array(state, dtype=np.float32)
        next_state_arr = np.array(next_state, dtype=np.float32)
        action_arr = np.array(action, dtype=np.float32)
        
        # NaN/Inf 검출
        if (np.isnan(state_arr).any() or np.isinf(state_arr).any() or
            np.isnan(next_state_arr).any() or np.isinf(next_state_arr).any() or
            np.isnan(action_arr).any() or np.isinf(action_arr).any() or
            np.isnan(reward) or np.isinf(reward)):
            print(f"❌ [오류] 리플레이 버퍼 저장 실패 - NaN/Inf 검출!")
            print(f"   state: {state_arr}")
            print(f"   action: {action_arr}")
            print(f"   reward: {reward}")
            print(f"   next_state: {next_state_arr}")
            return  # 저장하지 않음
        
        # 보상 범위 검증 및 클리핑
        if reward < -100.0 or reward > 50.0:
            print(f"⚠️ [경고] 비정상 보상 클리핑: {reward:.2f} → ", end="")
            reward = np.clip(reward, -100.0, 50.0)
            print(f"{reward:.2f}")
        
        # PID gain을 [Kp, Ki, Kd] 모두 사용 (3차원)
        norm_action = self._normalize_pid_action(action)
        
        # 정규화된 액션도 검증
        if np.isnan(norm_action).any() or np.isinf(norm_action).any():
            print(f"❌ [오류] 정규화된 액션에 NaN/Inf 검출! 원본 action: {action}")
            return  # 저장하지 않음
        
        self.replay.push(state_arr, norm_action, reward, next_state_arr, done)
        print(f"✅ [저장] 리플레이 버퍼에 데이터 저장 완료 (보상: {reward:.2f})")

    def _normalize_pid_action(self, pid_action):
        """PID gain을 [-1, 1] 범위로 정규화 (Kp, Ki, Kd 모두)"""

        def normalize_single(v, lo, hi):
            # 범위 체크: lo == hi인 경우 (D gain처럼 고정값)
            if abs(hi - lo) < 1e-9:
                # 고정값이므로 정규화 불필요, 0.0 반환
                return 0.0
            
            # 값 클리핑 (범위 내로 제한)
            v = np.clip(v, lo, hi)
            
            # 정규화
            normalized = 2.0 * (v - lo) / (hi - lo) - 1.0
            
            # 최종 안전 체크
            if np.isnan(normalized) or np.isinf(normalized):
                print(f"❌ [오류] 정규화 결과 비정상: v={v}, lo={lo}, hi={hi} → {normalized}")
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
        """
        한 스텝 MDP에 최적화된 SAC 업데이트
        Args:
            batch_size: 배치 크기 (None이면 replay buffer 크기에 맞춰 동적 조정)
            num_updates: 업데이트 횟수
        """
        # 동적 배치 크기: min(설정 배치 크기, 현재 버퍼 크기)
        bs = min(batch_size or self.cfg["BATCH_SIZE"], len(self.replay))

        # 최소 2개 이상 있어야 학습 가능
        if len(self.replay) < 2:
            return

        for _ in range(num_updates):
            s, a, r, ns, d = self.replay.sample(bs)
            s = torch.FloatTensor(s).to(self.device)
            a = torch.FloatTensor(a).to(self.device)
            r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
            ns = torch.FloatTensor(ns).to(self.device)
            d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
            
            # 입력 데이터 검증 (NaN/Inf 체크)
            if (torch.isnan(s).any() or torch.isnan(a).any() or 
                torch.isnan(r).any() or torch.isinf(r).any()):
                print(f"⚠️ [경고] 배치 데이터에 NaN/Inf 발견 - 업데이트 건너뜀")
                continue
            
            # 보상 정규화 (극단값 방지)
            r = torch.clamp(r, -100.0, 50.0)  # 보상 범위 제한

            # 한 스텝 MDP이므로 y = r (부트스트랩 없음)
            with torch.no_grad():
                y = r

            # Critic 업데이트
            q1, q2 = self.critic(s, a)
            
            # Q 값 검증 (NaN/Inf 체크)
            if torch.isnan(q1).any() or torch.isnan(q2).any() or torch.isinf(q1).any() or torch.isinf(q2).any():
                print(f"⚠️ [경고] Q 값에 NaN/Inf 발견 - Critic 업데이트 건너뜀")
                continue
            
            q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            
            # Loss 검증 (NaN 체크)
            if torch.isnan(q_loss) or torch.isinf(q_loss):
                print(f"⚠️ [경고] Critic loss가 비정상입니다: {q_loss.item()}")
                continue  # 이 업데이트 건너뛰기
            
            self.critic_opt.zero_grad()
            q_loss.backward()
            # Gradient clipping: 1.0 → 2.0 (균형잡힌 학습)
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), 2.0
            )
            self.critic_opt.step()

            # Actor 업데이트
            pi, logp = self.actor.sample(s)
            q1_pi, q2_pi = self.critic(s, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            pi_loss = ((self.alpha * logp) - min_q_pi).mean()
            
            # Loss 검증 (NaN 체크)
            if torch.isnan(pi_loss) or torch.isinf(pi_loss):
                print(f"⚠️ [경고] Actor loss가 비정상입니다: {pi_loss.item()}")
                continue  # 이 업데이트 건너뛰기
            
            self.actor_opt.zero_grad()
            pi_loss.backward()
            # Gradient clipping: 1.0 → 2.0 (균형잡힌 학습)
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), 2.0
            )
            self.actor_opt.step()

            # 엔트로피 자동 조절
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

            # 타겟 네트워크 소프트 업데이트
            with torch.no_grad():
                for tp, lp in zip(
                    self.critic_target.parameters(), self.critic.parameters()
                ):
                    tp.data.copy_(
                        self.tau * lp.data + (1 - self.tau) * tp.data
                    )

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
                "cfg": self.cfg,  # 설정 정보도 저장
            },
            path,
        )
        print(f"💾 Saved: {path}")

    def load_model(self, path, strict=True):
        """
        모델 로드 및 전이학습 설정
        Args:
            path: 모델 파일 경로
            strict: True면 완전 일치 필요, False면 부분 로드 허용
        """
        if not os.path.exists(path):
            print(f"⚠️ 모델 파일이 존재하지 않음: {path}")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Actor 로드
            self.actor.load_state_dict(checkpoint["actor"], strict=strict)
            
            # Critic 로드
            self.critic.load_state_dict(checkpoint["critic"], strict=strict)
            self.critic_target.load_state_dict(checkpoint["critic_target"], strict=strict)
            
            # Optimizer 로드 (선택적)
            if "actor_opt" in checkpoint:
                self.actor_opt.load_state_dict(checkpoint["actor_opt"])
            if "critic_opt" in checkpoint:
                self.critic_opt.load_state_dict(checkpoint["critic_opt"])
                
            # 학습 진행 상황 로드 (선택적)
            if "total_steps" in checkpoint:
                self.total_steps = checkpoint["total_steps"]
            if "episode_rewards" in checkpoint:
                self.episode_rewards = checkpoint["episode_rewards"]
                
            print(f"✅ 모델 로드 완료: {path}")
            print(f"📊 로드된 정보: total_steps={self.total_steps}, episodes={len(self.episode_rewards)}")
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