"""
설정 관리 - Config 클래스
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple

from .constants import Constants


@dataclass
class Config:
    """강화학습 설정을 관리하는 데이터클래스"""
    # 신경망 설정
    state_dim: int = 12  # PID 정보를 압축하여 12D 유지
    action_dim: int = 3
    hidden_dim: int = Constants.DEFAULT_HIDDEN_DIM
    lr: float = Constants.DEFAULT_LR
    lr_actor: float = Constants.DEFAULT_LR_ACTOR
    lr_critic: float = Constants.DEFAULT_LR_CRITIC
    gamma: float = Constants.DEFAULT_GAMMA
    tau: float = Constants.DEFAULT_TAU
    auto_entropy: bool = True
    
    # PID 설정
    pid_range: Dict[str, Tuple[float, float]] = field(default_factory=lambda: Constants.DEFAULT_PID_RANGE)
    
    # 에피소드 설정
    episode_seconds: float = Constants.DEFAULT_EPISODE_SECONDS
    target_force: float = Constants.DEFAULT_TARGET_FORCE
    updates_per_episode: int = Constants.DEFAULT_UPDATES_PER_EPISODE
    episodes: int = Constants.DEFAULT_EPISODES
    
    # 통신 설정
    recv_freq_hz: float = Constants.DEFAULT_RECV_FREQ
    recv_interval_sec: float = field(init=False)
    batch_size: int = Constants.DEFAULT_BATCH_SIZE
    host: str = Constants.DEFAULT_HOST
    port: int = Constants.DEFAULT_PORT
    recv_timeout_sec: float = Constants.DEFAULT_RECV_TIMEOUT
    recv_loop_timeout_sec: float = Constants.DEFAULT_RECV_LOOP_TIMEOUT
    
    # 실패 처리 설정
    comm_fail_max: int = Constants.DEFAULT_COMM_FAIL_MAX
    comm_retry_delay: float = Constants.DEFAULT_COMM_RETRY_DELAY
    
    # 저장 설정
    model_save_dir: str = Constants.DEFAULT_MODEL_SAVE_DIR
    log_dir: str = Constants.DEFAULT_LOG_DIR
    
    # 학습 설정
    max_episode_rewards_history: int = Constants.DEFAULT_MAX_REWARDS_HISTORY
    replay_buffer_size: int = Constants.DEFAULT_REPLAY_BUFFER_SIZE
    
    def __post_init__(self):
        """초기화 후 처리"""
        self.recv_interval_sec = 1.0 / self.recv_freq_hz
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환 (호환성)"""
        return {
            "STATE_DIM": self.state_dim,
            "ACTION_DIM": self.action_dim,
            "HIDDEN": self.hidden_dim,
            "LR": self.lr,
            "LR_ACTOR": self.lr_actor,
            "LR_CRITIC": self.lr_critic,
            "GAMMA": self.gamma,
            "TAU": self.tau,
            "AUTO_ENTROPY": self.auto_entropy,
            "PID_RANGE": self.pid_range,
            "EPISODE_SECONDS": self.episode_seconds,
            "TARGET_FORCE": self.target_force,
            "UPDATES_PER_EPISODE": self.updates_per_episode,
            "RECV_FREQ_HZ": self.recv_freq_hz,
            "RECV_INTERVAL_SEC": self.recv_interval_sec,
            "BATCH_SIZE": self.batch_size,
            "HOST": self.host,
            "PORT": self.port,
            "RECV_TIMEOUT_SEC": self.recv_timeout_sec,
            "RECV_LOOP_TIMEOUT_SEC": self.recv_loop_timeout_sec,
            "COMM_FAIL_MAX": self.comm_fail_max,
            "COMM_RETRY_DELAY": self.comm_retry_delay,
            "EPISODES": self.episodes,
            "MODEL_SAVE_DIR": self.model_save_dir,
            "LOG_DIR": self.log_dir,
            "MAX_EPISODE_REWARDS_HISTORY": self.max_episode_rewards_history,
            "REPLAY_BUFFER_SIZE": self.replay_buffer_size,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """딕셔너리로부터 생성"""
        return cls(
            state_dim=config_dict.get("STATE_DIM", 12),
            action_dim=config_dict.get("ACTION_DIM", 3),
            hidden_dim=config_dict.get("HIDDEN", Constants.DEFAULT_HIDDEN_DIM),
            lr=config_dict.get("LR", Constants.DEFAULT_LR),
            lr_actor=config_dict.get("LR_ACTOR", Constants.DEFAULT_LR_ACTOR),
            lr_critic=config_dict.get("LR_CRITIC", Constants.DEFAULT_LR_CRITIC),
            gamma=config_dict.get("GAMMA", Constants.DEFAULT_GAMMA),
            tau=config_dict.get("TAU", Constants.DEFAULT_TAU),
            auto_entropy=config_dict.get("AUTO_ENTROPY", True),
            pid_range=config_dict.get("PID_RANGE", Constants.DEFAULT_PID_RANGE),
            episode_seconds=config_dict.get("EPISODE_SECONDS", Constants.DEFAULT_EPISODE_SECONDS),
            target_force=config_dict.get("TARGET_FORCE", Constants.DEFAULT_TARGET_FORCE),
            updates_per_episode=config_dict.get("UPDATES_PER_EPISODE", Constants.DEFAULT_UPDATES_PER_EPISODE),
            episodes=config_dict.get("EPISODES", Constants.DEFAULT_EPISODES),
            recv_freq_hz=config_dict.get("RECV_FREQ_HZ", Constants.DEFAULT_RECV_FREQ),
            batch_size=config_dict.get("BATCH_SIZE", Constants.DEFAULT_BATCH_SIZE),
            host=config_dict.get("HOST", Constants.DEFAULT_HOST),
            port=config_dict.get("PORT", Constants.DEFAULT_PORT),
            recv_timeout_sec=config_dict.get("RECV_TIMEOUT_SEC", Constants.DEFAULT_RECV_TIMEOUT),
            recv_loop_timeout_sec=config_dict.get("RECV_LOOP_TIMEOUT_SEC", Constants.DEFAULT_RECV_LOOP_TIMEOUT),
            comm_fail_max=config_dict.get("COMM_FAIL_MAX", Constants.DEFAULT_COMM_FAIL_MAX),
            comm_retry_delay=config_dict.get("COMM_RETRY_DELAY", Constants.DEFAULT_COMM_RETRY_DELAY),
            model_save_dir=config_dict.get("MODEL_SAVE_DIR", Constants.DEFAULT_MODEL_SAVE_DIR),
            log_dir=config_dict.get("LOG_DIR", Constants.DEFAULT_LOG_DIR),
            max_episode_rewards_history=config_dict.get("MAX_EPISODE_REWARDS_HISTORY", Constants.DEFAULT_MAX_REWARDS_HISTORY),
            replay_buffer_size=config_dict.get("REPLAY_BUFFER_SIZE", Constants.DEFAULT_REPLAY_BUFFER_SIZE),
        )


def create_config(recv_freq_hz=None, episode_seconds=None) -> Config:
    """
    설정 생성 함수
    Args:
        recv_freq_hz: 수신 주파수 (Hz)
        episode_seconds: 에피소드 길이 (초)
    Returns:
        Config 객체
    """
    config = Config()
    
    if recv_freq_hz is not None:
        if recv_freq_hz <= 0 or recv_freq_hz > 10000:
            raise ValueError(
                f"수신 주파수는 0과 10000 사이여야 합니다: {recv_freq_hz}"
            )
        config.recv_freq_hz = recv_freq_hz
        config.__post_init__()  # recv_interval_sec 재계산
    
    if episode_seconds is not None:
        if episode_seconds <= 0:
            raise ValueError(
                f"에피소드 길이는 0보다 커야 합니다: {episode_seconds}"
            )
        config.episode_seconds = episode_seconds
    
    return config


def change_episode_length(config: Config, new_length_seconds: float) -> Config:
    """
    에피소드 길이를 동적으로 변경하는 함수
    Args:
        config: 현재 설정
        new_length_seconds: 새로운 에피소드 길이 (초)
    Returns:
        업데이트된 설정
    """
    if new_length_seconds <= 0:
        raise ValueError(
            f"에피소드 길이는 0보다 커야 합니다: {new_length_seconds}"
        )
    
    config.episode_seconds = new_length_seconds
    print(f"🔄 에피소드 길이 변경: {new_length_seconds}초")
    print(f"📊 목표 데이터 개수: {int(new_length_seconds * config.recv_freq_hz)}개")
    return config

