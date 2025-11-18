# 📁 Config & Execution Modules

This folder contains **execution and configuration modules** for the PID Gain RL system.

## 📋 Modules in This Folder

### Files
- **`__main__.py`** - Module entry point (Python module execution)
- **`main.py`** - Command-line interface (argparse)
- **`config.py`** - Configuration management (dataclass)
- **`constants.py`** - All constants definition

## 📖 Documentation

See **[CODE_FUNCTIONALITY_1.md](./CODE_FUNCTIONALITY_1.md)** for detailed functionality analysis.

## 🎯 Purpose

These modules handle:
- **Program execution** (entry points, CLI interface)
- **Configuration management** (type-safe settings, constants)
- **Reproducibility** (seed setting)
- **Safe shutdown** (signal handlers)

## 🚀 Usage

### Method 1: Module execution (recommended)
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl
python3 -m pid_gain_rl.experiment.config
```

### Method 2: CLI interface
```bash
cd /home/katech/Robot-Polishing-RL-system/py_rl
python3 -m pid_gain_rl.experiment.config.main --episodes 500 --target-force -50.0
```

## 📦 Dependencies

- **Internal**: `core.env`, `utils.utils`, `utils.loggers`
- **External**: `numpy`, `torch`




