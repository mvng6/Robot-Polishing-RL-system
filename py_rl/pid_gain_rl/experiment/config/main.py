"""
메인 실행 스크립트 (명령줄 인터페이스)
- argparse로 실행 파라미터 설정
- __main__.py의 로직 재사용
- 세그먼트 분할 학습 지원 (1 에피소드 = 5 transition)
- 표준편차 Annealing 지원
- STATE_DIM 동적 계산 (20차원)
"""
import argparse
import os
import sys
import numpy as np
import torch
from datetime import datetime

from .config import create_config
from ..core.env import PIDGainEnvironment
from ..utils.utils.signals import install_signal_handlers
from ..utils.utils.data_saver import DataSaver
from .constants import Constants
from ..utils.loggers.base_logger import AppLogger

# 전역 env 변수 (signal handler에서 사용)
_global_env = None


def main():
    parser = argparse.ArgumentParser(
        description="PID Gain RL Training with Segment Split (Command Line Interface)"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=Constants.DEFAULT_EPISODES,
        help=f"학습할 에피소드 수 (기본값: {Constants.DEFAULT_EPISODES})"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=Constants.DEFAULT_BATCH_SIZE,
        help=f"배치 크기 (기본값: {Constants.DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=Constants.DEFAULT_LR,
        help=f"학습률 (기본값: {Constants.DEFAULT_LR})"
    )
    parser.add_argument(
        "--lr-actor", 
        type=float, 
        default=Constants.DEFAULT_LR_ACTOR,
        help=f"Actor 학습률 (기본값: {Constants.DEFAULT_LR_ACTOR})"
    )
    parser.add_argument(
        "--lr-critic", 
        type=float, 
        default=Constants.DEFAULT_LR_CRITIC,
        help=f"Critic 학습률 (기본값: {Constants.DEFAULT_LR_CRITIC})"
    )
    parser.add_argument(
        "--target-force", 
        type=float, 
        default=Constants.DEFAULT_TARGET_FORCE,
        help=f"목표 힘 (N) (기본값: {Constants.DEFAULT_TARGET_FORCE})"
    )
    parser.add_argument(
        "--episode-seconds", 
        type=float, 
        default=Constants.DEFAULT_EPISODE_SECONDS,
        help=f"에피소드 길이 (초) (기본값: {Constants.DEFAULT_EPISODE_SECONDS})"
    )
    parser.add_argument(
        "--recv-freq", 
        type=float, 
        default=Constants.DEFAULT_RECV_FREQ,
        help=f"수신 주파수 (Hz) (기본값: {Constants.DEFAULT_RECV_FREQ})"
    )
    parser.add_argument(
        "--load-model", 
        type=str, 
        default=None,
        help="로드할 모델 경로 (.pth 파일)"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default=Constants.DEFAULT_LOG_DIR,
        help=f"로그 디렉토리 (기본값: {Constants.DEFAULT_LOG_DIR})"
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default=Constants.DEFAULT_MODEL_SAVE_DIR,
        help=f"모델 저장 디렉토리 (기본값: {Constants.DEFAULT_MODEL_SAVE_DIR})"
    )
    
    args = parser.parse_args()
    
    # 설정 생성 (__main__.py 방식 재사용)
    config = create_config(
        recv_freq_hz=args.recv_freq,
        episode_seconds=args.episode_seconds
    )
    
    # 명령줄 인자로 덮어쓰기
    config.target_force = args.target_force
    config.episodes = args.episodes
    config.lr = args.lr
    config.lr_actor = args.lr_actor
    config.lr_critic = args.lr_critic
    config.batch_size = args.batch_size
    config.log_dir = args.log_dir
    config.model_save_dir = args.model_dir
    
    config_dict = config.to_dict()
    
    # 시작 정보 출력
    print("=" * 80)
    print("🚀 PID GAIN OPTIMIZATION VERSION: Modularized (CLI)")
    print("=" * 80)
    print(f"📡 수신 주파수: {config.recv_freq_hz}Hz (간격: {config.recv_interval_sec:.3f}초)")
    print(f"🎯 목표 힘: {config.target_force}N")
    print(f"⏱️ 에피소드 길이: {config.episode_seconds}초")
    print(f"📊 에피소드 수: {config.episodes}")
    print(f"🧠 학습률: Actor={config.lr_actor:.2e}, Critic={config.lr_critic:.2e}")
    print(f"📦 배치 크기: {config.batch_size}")
    print(f"✅ STATE_DIM: {config_dict['STATE_DIM']}차원 (기본 {Constants.STATE_BASE_DIM} + 궤적 {Constants.STATE_TRAJECTORY_DIM})")
    print(f"✅ 세그먼트 분할: {Constants.NUM_SEGMENTS}개 ({Constants.SEGMENT_LENGTH_S}초씩)")
    print(f"✅ 탐험 설정: alpha={Constants.ACTOR_INITIAL_ALPHA}, log_std_max={Constants.ACTOR_LOG_STD_MAX}")
    print(f"✅ 표준편차 Annealing: {Constants.STD_ANNEAL_INITIAL} → {Constants.STD_ANNEAL_FINAL}")
    print("=" * 80)
    
    # 재현성을 위한 시드 설정
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("🎲 재현성 시드 설정 완료 (42)")
    
    # 디렉토리 생성
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 환경 생성
    global _global_env
    env = PIDGainEnvironment(config_dict)
    _global_env = env
    
    # 시그널 핸들러 설치 (env 생성 후)
    install_signal_handlers()
    
    # 모델 로드
    if args.load_model:
        if env.agent.load_model(args.load_model):
            print(f"✅ 모델 로드 완료: {args.load_model}")
            AppLogger.log("INFO", f"모델 로드 완료: {args.load_model}")
        else:
            print(f"⚠️ 모델 로드 실패 - 새로 학습 시작")
            AppLogger.log("WARNING", f"모델 로드 실패: {args.load_model}")
    
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
        sys.exit(1)
        
    finally:
        AppLogger.log("INFO", "🔚 Training program terminated.")


if __name__ == "__main__":
    main()
