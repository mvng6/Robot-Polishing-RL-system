"""
메인 엔트리 포인트
"""
import numpy as np
import torch

from .config import create_config
from .env import PIDGainEnvironment
from .utils.signals import install_signal_handlers
from .utils.data_saver import DataSaver
from .loggers.base_logger import AppLogger


# 전역 env 변수 (signal handler에서 사용)
_global_env = None

def main():
    """메인 실행 함수"""
    global _global_env
    
    # 설정 변경 포인트
    RECV_FREQUENCY_HZ = 1000
    EPISODE_LENGTH_SECONDS = 10.0  # 에피소드 길이 (PID 과도응답 5~10초 완료)
    
    config = create_config(RECV_FREQUENCY_HZ, EPISODE_LENGTH_SECONDS)
    config_dict = config.to_dict()  # 딕셔너리로 변환
    
    print("🚀 PID GAIN OPTIMIZATION VERSION: Modularized")
    print(f"📡 수신 주파수: {RECV_FREQUENCY_HZ}Hz (간격: {config.recv_interval_sec:.3f}초)")
    print(f"🎯 목표 힘: {config.target_force}N 고정")
    print(f"⏱️ 에피소드 길이: {config.episode_seconds}초")
    print("=" * 60)
    
    # 재현성을 위한 시드 설정
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("🎲 재현성 시드 설정 완료 (42)")
    
    env = PIDGainEnvironment(config_dict)
    _global_env = env
    
    # 시그널 핸들러 설치 (env 생성 후)
    install_signal_handlers()
    
    try:
        print(f"🚀 Starting PID Gain optimization training...")
        env.run_pid_optimization_training(config.episodes)
        print("✅ Training completed successfully!")
        
        try:
            print("📈 데이터 저장 중...")
            try:
                env.rlogger.flush_if_needed(
                    config.episodes,
                    force=True,
                    episode_rewards=env.agent.episode_rewards,
                )
            except Exception as e:
                AppLogger.log("ERROR", f"reward breakdown flush 실패: {e}")
            
            DataSaver.save_all_data(env)
        except Exception as e:
            AppLogger.log("ERROR", f"❌ 데이터 저장 실패: {e}")
            
    except KeyboardInterrupt:
        AppLogger.log("WARNING", "Interrupted by user (Ctrl+C).")
        print("\n⚠️ 사용자가 Ctrl+C로 중단했습니다. 안전하게 종료 중...")
        
        try:
            print("📡 강화학습 중단 신호 전송 중...")
            success = env.comm.send_pid_once(0.0, 0.0, 0.0, True, False, True)
            if success:
                print("✅ 강화학습 중단 신호 전송 성공")
                AppLogger.log("INFO", "📤 학습 중단 신호 전송 성공")
            else:
                print("⚠️ 강화학습 중단 신호 전송 실패")
                AppLogger.log("ERROR", "학습 중단 신호 전송 실패")
        except Exception as e:
            print(f"⚠️ 강화학습 중단 신호 전송 오류: {e}")
            AppLogger.log("ERROR", f"학습 중단 신호 전송 오류: {e}")
        
        DataSaver.save_all_data(env)
        env.comm.close()
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        AppLogger.log("ERROR", f"학습 중 오류 발생: {e}")
        
        try:
            print("📡 강화학습 오류 종료 신호 전송 중...")
            success = env.comm.send_pid_once(0.0, 0.0, 0.0, True, False, True)
            if success:
                print("✅ 강화학습 오류 종료 신호 전송 성공")
                AppLogger.log("INFO", "📤 학습 오류 종료 신호 전송 성공")
            else:
                print("⚠️ 강화학습 오류 종료 신호 전송 실패")
                AppLogger.log("ERROR", "학습 오류 종료 신호 전송 실패")
        except Exception as e2:
            print(f"⚠️ 강화학습 오류 종료 신호 전송 오류: {e2}")
            AppLogger.log("ERROR", f"학습 오류 종료 신호 전송 오류: {e2}")
        
        DataSaver.save_all_data(env)
        env.comm.close()
        
    finally:
        AppLogger.log("INFO", "🔚 Training program terminated.")


if __name__ == "__main__":
    main()

