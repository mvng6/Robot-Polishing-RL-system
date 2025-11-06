# 📁 Core RL Modules

This folder contains **core reinforcement learning modules** for the PID Gain RL system.

## 📋 Modules in This Folder

### Files
- **`agent.py`** - SAC Agent (Actor, Critic, ReplayBuffer)
- **`env.py`** - PID Gain RL Environment (main training loop, reward calculation)
- **`comm.py`** - TCP/IP Communication wrapper (robot control PC)
- **`monitor.py`** - Real-time monitoring GUI (matplotlib)

## 📖 Documentation

See **[CODE_FUNCTIONALITY_2.md](./CODE_FUNCTIONALITY_2.md)** for detailed functionality analysis.

## 🎯 Purpose

These modules handle:
- **SAC Algorithm** (Actor-Critic networks, replay buffer, learning)
- **RL Environment** (episode management, state/action/reward, segment splitting)
- **Communication** (1kHz data exchange with robot control PC)
- **Real-time Monitoring** (visualization during training)

## 📦 Dependencies

- **Internal**: `config.constants`, `utils.utils`, `utils.loggers`
- **External**: `numpy`, `torch`, `matplotlib`, `socket`

## 🔗 Module Relationships

```
env.py (main)
  ├── agent.py (SAC algorithm)
  ├── comm.py (TCP/IP communication)
  ├── monitor.py (real-time GUI)
  ├── config.constants (Constants)
  ├── utils.utils.math_utils (state/action utilities)
  └── utils.loggers.* (logging)
```

