"""
Main entry point - Redirects to experiment.config.__main__
Maintains backward compatibility with: python3 -m pid_gain_rl
"""
from pid_gain_rl.experiment.config.__main__ import main

if __name__ == "__main__":
    main()

