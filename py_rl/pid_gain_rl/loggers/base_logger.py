"""
기본 로거 - AppLogger (이름 충돌 방지)
"""
from datetime import datetime
import logging


class AppLogger:
    """애플리케이션 로거 - Python 표준 logging 래핑"""
    
    @staticmethod
    def log(level, message):
        """
        로그 출력
        Args:
            level: 로그 레벨 (INFO, SUCCESS, WARNING, ERROR, DEBUG)
            message: 로그 메시지
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍",
        }
        print(f"[{timestamp}] {icons.get(level, 'ℹ️')} {message}")

