"""
Core RL modules - Agent, Environment, Communication, Monitor
"""
# Lazy imports to avoid circular dependencies
__all__ = [
    "PIDGainSACAgent",
    "Actor",
    "Critic",
    "ReplayBuffer",
    "PIDGainEnvironment",
    "PIDGainCommunicator",
    "RLRealtimeMonitor",
]

def __getattr__(name):
    """Lazy import to avoid circular dependencies"""
    if name == "PIDGainSACAgent" or name == "Actor" or name == "Critic" or name == "ReplayBuffer":
        from .agent import PIDGainSACAgent, Actor, Critic, ReplayBuffer
        if name == "PIDGainSACAgent":
            return PIDGainSACAgent
        elif name == "Actor":
            return Actor
        elif name == "Critic":
            return Critic
        elif name == "ReplayBuffer":
            return ReplayBuffer
    elif name == "PIDGainEnvironment":
        from .env import PIDGainEnvironment
        return PIDGainEnvironment
    elif name == "PIDGainCommunicator":
        from .comm import PIDGainCommunicator
        return PIDGainCommunicator
    elif name == "RLRealtimeMonitor":
        from .monitor import RLRealtimeMonitor
        return RLRealtimeMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

