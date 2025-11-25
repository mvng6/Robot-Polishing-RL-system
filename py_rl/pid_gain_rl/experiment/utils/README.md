# 📁 Utilities & Loggers Modules

This folder contains **utility and logger modules** for the PID Gain RL system.

## 📋 Modules in This Folder

### Utility Modules (`utils/`)
- **`math_utils.py`** - Math utilities (PID action scaling, state creation)
- **`data_saver.py`** - Data saving utilities (unified data storage)
- **`signals.py`** - Signal handling (safe shutdown)

### Logger Modules (`loggers/`)
- **`base_logger.py`** - Base logger (AppLogger)
- **`control_performance.py`** - Control performance logger (10 metrics)
- **`reward_breakdown.py`** - Reward analysis logger (reward components)
- **`learning_done.py`** - Learning completion logger (folder management)

## 📖 Documentation

See **[CODE_FUNCTIONALITY_3.md](./CODE_FUNCTIONALITY_3.md)** for detailed functionality analysis.

## 🎯 Purpose

These modules handle:
- **Math Utilities** (PID action scaling, initial state creation)
- **Data Management** (saving all data, safe shutdown)
- **Control Performance Logging** (10 key control engineering metrics)
- **Reward Analysis** (reward component breakdown, visualization)
- **Learning Completion** (timestamp-based folder management)

## 📦 Dependencies

- **Internal**: `config.constants`
- **External**: `numpy`, `matplotlib`, `csv`, `signal`

## 📂 Folder Structure

```
utils/
├── utils/
│   ├── math_utils.py
│   ├── data_saver.py
│   └── signals.py
└── loggers/
    ├── base_logger.py
    ├── control_performance.py
    ├── reward_breakdown.py
    └── learning_done.py
```










