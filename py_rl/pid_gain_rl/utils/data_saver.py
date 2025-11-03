"""
데이터 저장 유틸리티
"""
from datetime import datetime


class Logger:
    """간단한 로거"""
    @staticmethod
    def log(level, message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍",
        }
        print(f"[{timestamp}] {icons.get(level, 'ℹ️')} {message}")


class DataSaver:
    """모든 데이터를 저장하는 유틸리티"""
    
    @staticmethod
    def save_all_data(env, current_episode=None, force=True):
        """
        모든 데이터를 저장
        Args:
            env: 환경 객체
            current_episode: 현재 에피소드 수
            force: 강제 저장 여부
        """
        try:
            if current_episode is not None:
                env.rlogger.flush_if_needed(
                    current_episode,
                    force=force,
                    episode_rewards=env.agent.episode_rewards,
                )
            else:
                env.rlogger.flush_if_needed(
                    env.cfg["EPISODES"],
                    force=force,
                    episode_rewards=env.agent.episode_rewards,
                )
        except Exception as e:
            Logger.log("ERROR", f"reward breakdown flush 실패: {e}")
        
        try:
            Logger.log("INFO", "📊 제어 성능 지표 저장 중...")
            env.cplogger.save_performance_summary()
            env.cplogger.generate_plots()
            Logger.log("INFO", "✅ 제어 성능 지표 저장 완료!")
        except Exception as e:
            Logger.log("ERROR", f"제어 성능 지표 저장 실패: {e}")
        
        Logger.log("INFO", "✅ 데이터 저장 완료!")



