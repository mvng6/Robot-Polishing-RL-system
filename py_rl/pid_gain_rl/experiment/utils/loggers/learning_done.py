"""
학습 완료 로거
"""
import os
from datetime import datetime


class LearningDoneLogger:
    """
    학습 완료 시 전체 로깅을 관리하는 클래스
    """

    def __init__(self, log_dir):
        # 타임스탬프 기반 learning_done 폴더 한 번만 생성
        now = datetime.now()
        timestamp = now.strftime("%y%m%d_%Hh%Mm")
        self.log_dir = os.path.join(log_dir, f"learning_done_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        print(f"📁 Learning Done 폴더: {self.log_dir}")

