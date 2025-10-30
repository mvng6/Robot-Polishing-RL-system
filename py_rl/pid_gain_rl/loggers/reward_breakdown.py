"""
스텝 단위 보상 분석 로깅
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

class RewardBreakdownLogger:
    """
    스텝 단위 보상 분석 로깅
    - 실시간 보상 구성 요소 수집
    - 에피소드별 보상 통계 생성
    - CSV 저장 및 PNG 시각화
    - Ctrl+C/학습 완료 시 자동 저장
    """

    def __init__(self, log_dir):
        # learning_done 폴더 내부의 reward_breakdown 서브폴더 사용 (중복 방지)
        self.log_dir = os.path.join(log_dir, "reward_breakdown")
        os.makedirs(self.log_dir, exist_ok=True)
        self.rows = []  # 버퍼
        self.csv_path = os.path.join(self.log_dir, "reward_breakdown.csv")
        self.episode_rewards_path = os.path.join(
            self.log_dir, "episode_rewards.csv"
        )
        print(f"📁 Reward breakdown 저장 폴더: {self.log_dir}")

    def log_step(
        self,
        episode,
        step,
        prog,
        in_band_now,
        edot_abs,
        du_abs,
        reward,
        is_her,
    ):
        self.rows.append(
            {
                "episode": episode,
                "step": step,
                "prog": float(prog),
                "in_band_now": int(in_band_now),
                "edot_abs": float(edot_abs),
                "du_abs": float(du_abs),
                "reward": float(reward),
                "is_her": int(is_her),
            }
        )

    def save_episode_rewards(self, episode_rewards):
        """에피소드별 보상을 CSV로 저장"""
        with open(self.episode_rewards_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])  # 헤더
            for i, reward in enumerate(episode_rewards, 1):
                writer.writerow([i, float(reward)])

    def generate_episode_reward_graph(self, episode_rewards):
        """에피소드별 보상 그래프를 PNG로 저장"""
        if not episode_rewards:
            return

        try:
            episodes = list(range(1, len(episode_rewards) + 1))
            plt.figure(figsize=(12, 6))
            plt.plot(
                episodes,
                episode_rewards,
                "b-",
                linewidth=2,
                marker="o",
                markersize=4,
            )
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Episode Reward", fontsize=12)
            plt.title(
                "Episode Rewards Over Time", fontsize=14, fontweight="bold"
            )
            plt.grid(True, alpha=0.3)
            if len(episode_rewards) > 1:
                avg_reward = np.mean(episode_rewards)
                plt.axhline(
                    y=avg_reward,
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Average: {avg_reward:.2f}",
                )
                plt.legend()

            filename = os.path.join(self.log_dir, "episode_rewards.png")
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"   📈 PNG: episode_rewards.png")
        except Exception as e:
            print(f"   ⚠️ 에피소드 리워드 그래프 생성 실패: {e}")

    def _write_csv_append(self):
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "step",
                    "prog",
                    "in_band_now",
                    "edot_abs",
                    "du_abs",
                    "reward",
                    "is_her",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.rows)

    def save_reward_breakdown_csv(self):
        """reward_breakdown 데이터를 CSV로 저장"""
        if not self.rows:
            return

        csv_path = os.path.join(self.log_dir, "reward_breakdown.csv")
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "step",
                    "prog",
                    "in_band_now",
                    "edot_abs",
                    "du_abs",
                    "reward",
                    "is_her",
                ],
            )
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"   📊 CSV: reward_breakdown.csv")

    def _plot_png(self, start_ep, end_ep):
        # start_ep~end_ep 사이의 데이터만 사용
        data = [r for r in self.rows if start_ep <= r["episode"] <= end_ep]
        if not data:
            return
        # 에피소드별 평균
        ep_keys = sorted(set(r["episode"] for r in data))
        avg_prog = []
        avg_in_band = []
        avg_edot = []
        avg_du = []
        avg_R = []
        for ep in ep_keys:
            items = [r for r in data if r["episode"] == ep]
            avg_prog.append(np.mean([r["prog"] for r in items]))
            avg_in_band.append(np.mean([r["in_band_now"] for r in items]))
            avg_edot.append(np.mean([r["edot_abs"] for r in items]))
            avg_du.append(np.mean([r["du_abs"] for r in items]))
            avg_R.append(np.mean([r["reward"] for r in items]))

        # 1) prog
        plt.figure(figsize=(11, 4))
        plt.plot(ep_keys, avg_prog, linewidth=2, marker="o")
        plt.title(f"Average prog per episode ({start_ep}-{end_ep})")
        plt.xlabel("Episode")
        plt.ylabel("prog")
        plt.grid(True, alpha=0.3)
        out1 = os.path.join(
            self.log_dir, f"reward_breakdown_prog_ep{start_ep}-{end_ep}.png"
        )
        plt.tight_layout()
        plt.savefig(out1, dpi=200)
        plt.close()

        # 2) in_band_now
        plt.figure(figsize=(11, 4))
        plt.plot(ep_keys, avg_in_band, linewidth=2, marker="o")
        plt.title(f"Average in_band_now per episode ({start_ep}-{end_ep})")
        plt.xlabel("Episode")
        plt.ylabel("in_band_now (ratio)")
        plt.grid(True, alpha=0.3)
        out2 = os.path.join(
            self.log_dir, f"reward_breakdown_inband_ep{start_ep}-{end_ep}.png"
        )
        plt.tight_layout()
        plt.savefig(out2, dpi=200)
        plt.close()

        # 3) edot_abs
        plt.figure(figsize=(11, 4))
        plt.plot(ep_keys, avg_edot, linewidth=2, marker="o")
        plt.title(f"Average |de/dt| per episode ({start_ep}-{end_ep})")
        plt.xlabel("Episode")
        plt.ylabel("|de/dt|")
        plt.grid(True, alpha=0.3)
        out3 = os.path.join(
            self.log_dir, f"reward_breakdown_edot_ep{start_ep}-{end_ep}.png"
        )
        plt.tight_layout()
        plt.savefig(out3, dpi=200)
        plt.close()

        # 4) du_abs
        plt.figure(figsize=(11, 4))
        plt.plot(ep_keys, avg_du, linewidth=2, marker="o")
        plt.title(f"Average |Δu| per episode ({start_ep}-{end_ep})")
        plt.xlabel("Episode")
        plt.ylabel("|Δu|")
        plt.grid(True, alpha=0.3)
        out4 = os.path.join(
            self.log_dir, f"reward_breakdown_du_ep{start_ep}-{end_ep}.png"
        )
        plt.tight_layout()
        plt.savefig(out4, dpi=200)
        plt.close()

        # 5) reward
        plt.figure(figsize=(11, 4))
        plt.plot(ep_keys, avg_R, linewidth=2, marker="o")
        plt.title(f"Average reward per episode ({start_ep}-{end_ep})")
        plt.xlabel("Episode")
        plt.ylabel("reward")
        plt.grid(True, alpha=0.3)
        out5 = os.path.join(
            self.log_dir, f"reward_breakdown_reward_ep{start_ep}-{end_ep}.png"
        )
        plt.tight_layout()
        plt.savefig(out5, dpi=200)
        plt.close()

    def flush_if_needed(
        self, current_episode, force=False, episode_rewards=None
    ):
        """
        CSV 저장 + PNG 시각화를 수행.
        force=True이면 언제든 실행, False이면 CSV만 저장 (PNG 생성 안 함).
        episode_rewards: 에피소드별 보상 리스트 (선택사항)
        """
        # 데이터가 없으면 실행하지 않음
        if not self.rows:
            return

        # CSV는 항상 저장 (메모리 절약)
        self.save_reward_breakdown_csv()

        # force=True일 때만 PNG 생성 (최종에만)
        if force:
            # 에피소드별 보상 저장 및 그래프 생성
            if episode_rewards is not None:
                self.save_episode_rewards(episode_rewards)
                self.generate_episode_reward_graph(episode_rewards)

            # PNG 생성 (전체 데이터)
            start_ep = min(row["episode"] for row in self.rows)
            end_ep = max(row["episode"] for row in self.rows)
            self._plot_png(start_ep, end_ep)

            print(f"✅ Reward breakdown 저장: {self.log_dir}")

        # 메모리 절약을 위해 rows 유지 (전체 그래프용)

