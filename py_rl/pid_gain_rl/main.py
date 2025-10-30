"""
간단한 실행 스크립트
이 파일은 py_rl/pid_gain_rl 디렉토리에서 직접 실행 가능하도록 설계됨
"""
import sys
import os

# 현재 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    from pid_gain_rl.__main__ import main
    main()
