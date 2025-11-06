"""
시그널 처리 - 안전한 종료
"""
import signal
import sys

from .data_saver import Logger, DataSaver


def signal_handler(signum, frame, env=None):
    """
    시그널 핸들러 - 안전한 종료
    Args:
        signum: 시그널 번호
        frame: 현재 스택 프레임
        env: 환경 객체 (전역에서 가져올 수 있으면 전달)
    """
    print(f"\n⚠️ Received signal {signum}. Shutting down gracefully...")
    
    if env is not None:
        try:
            # 강제 종료 시 learning_done=True 전송
            print("📡 강화학습 강제 종료 신호 전송 중...")
            try:
                success = env.comm.send_pid_once(
                    0.0, 0.0, 0.0, True, False, True
                )  # learning_done=True
                if success:
                    print("✅ 강화학습 강제 종료 신호 전송 성공")
                else:
                    print("⚠️ 강화학습 강제 종료 신호 전송 실패")
            except Exception as e:
                print(f"⚠️ 강화학습 강제 종료 신호 전송 오류: {e}")

            # 데이터 저장
            print("📈 데이터 저장 중...")
            try:
                current_episode = len(env.agent.episode_rewards)
                env.rlogger.flush_if_needed(
                    current_episode,
                    force=True,
                    episode_rewards=env.agent.episode_rewards,
                )
            except Exception as e:
                Logger.log("ERROR", f"reward breakdown flush 실패: {e}")

            # 제어 성능 지표 저장
            try:
                print("📊 제어 성능 지표 저장 중...")
                env.cplogger.save_performance_summary()
                env.cplogger.generate_plots()
                print("✅ 제어 성능 지표 저장 완료!")
            except Exception as e:
                print(f"⚠️ 제어 성능 지표 저장 실패: {e}")

            Logger.log("INFO", "✅ 데이터 저장 완료!")
        except Exception as e:
            Logger.log("ERROR", f"❌ 데이터 저장 실패: {e}")
    
    sys.exit(0)


def install_signal_handlers():
    """
    시그널 핸들러 설치
    """
    def handler_wrapper(signum, frame):
        # __main__ 모듈에서 _global_env 참조
        # experiment.config.__main__ 또는 pid_gain_rl.__main__에서 참조
        import sys
        import __main__
        # 여러 가능한 모듈 경로 확인
        env_global = None
        if hasattr(__main__, '_global_env'):
            env_global = __main__._global_env
        else:
            # pid_gain_rl.experiment.config.__main__ 모듈 확인
            try:
                from pid_gain_rl.experiment.config import __main__ as config_main
                if hasattr(config_main, '_global_env'):
                    env_global = config_main._global_env
            except:
                pass
        signal_handler(signum, frame, env_global)
    
    signal.signal(signal.SIGINT, handler_wrapper)
    signal.signal(signal.SIGTERM, handler_wrapper)

