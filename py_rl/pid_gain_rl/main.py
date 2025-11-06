"""
Main CLI entry point - Redirects to experiment.config.main
Maintains backward compatibility with: python3 main.py
"""
from pid_gain_rl.experiment.config.main import main

if __name__ == "__main__":
    main()

