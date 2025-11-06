# 📁 Experiment - Reorganized Code Structure

This folder contains the **reorganized code structure** for the PID Gain RL system.

## 📂 Folder Structure

```
experiment/
├── README.md (this file)
├── config/
│   ├── README.md - Execution & configuration modules overview
│   ├── CODE_FUNCTIONALITY_1.md - Detailed analysis
│   ├── __main__.py - Module entry point
│   ├── main.py - CLI interface
│   ├── config.py - Configuration management
│   └── constants.py - Constants definition
├── core/
│   ├── README.md - Core RL modules overview
│   ├── CODE_FUNCTIONALITY_2.md - Detailed analysis
│   ├── agent.py - SAC Agent
│   ├── env.py - RL Environment
│   ├── comm.py - TCP/IP Communication
│   └── monitor.py - Real-time Monitoring
└── utils/
    ├── README.md - Utilities & loggers overview
    ├── CODE_FUNCTIONALITY_3.md - Detailed analysis
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

## 🎯 Purpose

This reorganization separates code into logical modules:
- **Config**: Execution entry points and configuration
- **Core**: Core RL algorithms and environment
- **Utils**: Utilities and logging modules

## 🚀 Usage

### Method 1: Original execution (backward compatible)
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl
python3 -m pid_gain_rl
```

### Method 2: Direct module execution
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl
python3 -m pid_gain_rl.experiment.config
```

### Method 3: CLI interface
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl
python3 -m pid_gain_rl.experiment.config.main --episodes 500 --target-force -40.0
```

## 📖 Documentation

Each folder contains:
- **README.md**: Quick overview of modules in that folder
- **CODE_FUNCTIONALITY_X.md**: Detailed functionality analysis

## ⚠️ Note

All import paths have been updated to reflect the new structure. The root-level `__main__.py` and `main.py` are maintained for backward compatibility.

