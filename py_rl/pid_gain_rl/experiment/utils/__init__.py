"""
Utilities and loggers modules
"""
# Lazy imports to avoid circular dependencies
__all__ = [
    "scale_action_to_pid",
    "create_initial_state",
    "DataSaver",
    "install_signal_handlers",
    "AppLogger",
    "ControlPerformanceLogger",
    "RewardBreakdownLogger",
    "LearningDoneLogger",
]

def __getattr__(name):
    """Lazy import to avoid circular dependencies"""
    if name in ["scale_action_to_pid", "create_initial_state", "DataSaver", "install_signal_handlers"]:
        from .utils import scale_action_to_pid, create_initial_state, DataSaver, install_signal_handlers
        return locals()[name]
    elif name in ["AppLogger", "ControlPerformanceLogger", "RewardBreakdownLogger", "LearningDoneLogger"]:
        from .loggers import AppLogger, ControlPerformanceLogger, RewardBreakdownLogger, LearningDoneLogger
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

