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
        # 🆕 에피소드별 보상 구성 요소 저장
        self.episode_components = []  # 버퍼
        self.episode_components_path = os.path.join(
            self.log_dir, "episode_reward_components.csv"
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

            # 🆕 보상 구성 요소 CSV 저장 및 그래프 생성
            if self.episode_components:
                self.save_episode_components_csv()
                self.generate_reward_components_graph()

            # PNG 생성 (전체 데이터)
            start_ep = min(row["episode"] for row in self.rows)
            end_ep = max(row["episode"] for row in self.rows)
            self._plot_png(start_ep, end_ep)

            print(f"✅ Reward breakdown 저장: {self.log_dir}")

        # 메모리 절약을 위해 rows 유지 (전체 그래프용)

    def log_episode_components(
        self,
        episode,
        reward_score,
        r_centered,
        r_baseline,
        overshoot,
        settling_time,
        band_ratio,
        reward,
    ):
        """
        🆕 에피소드별 보상 구성 요소 로깅
        
        Args:
            episode: 에피소드 번호
            reward_score: 원시 보상 스코어 (0~1)
            r_centered: 중심화된 보상 (reward_score - baseline)
            r_baseline: 보상 기준선 (EWMA)
            overshoot: 오버슈트 (%)
            settling_time: 정착시간 (초)
            band_ratio: 밴드 유지 비율 (0~1)
            reward: 최종 보상 (tanh 적용 후, -1~1)
        """
        self.episode_components.append(
            {
                "episode": episode,
                "reward_score": float(reward_score),
                "r_centered": float(r_centered),
                "r_baseline": float(r_baseline),
                "overshoot": float(overshoot),
                "settling_time": float(settling_time),
                "band_ratio": float(band_ratio),
                "reward": float(reward),
            }
        )

    def save_episode_components_csv(self):
        """에피소드별 보상 구성 요소를 CSV로 저장"""
        if not self.episode_components:
            return

        with open(self.episode_components_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "reward_score",
                    "r_centered",
                    "r_baseline",
                    "overshoot",
                    "settling_time",
                    "band_ratio",
                    "reward",
                ],
            )
            writer.writeheader()
            writer.writerows(self.episode_components)
        print(f"   📊 CSV: episode_reward_components.csv")

    def _moving_average(self, data, window=10):
        """Moving average 계산 (window 크기만큼의 평균)"""
        if len(data) < window:
            return data
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result.append(np.mean(data[start : i + 1]))
        return result

    def generate_reward_components_graph(self):
        """
        🆕 보상 구성 요소 시각화 그래프 생성
        - reward_score, r_centered, r_baseline을 동일 x축에 플롯
        - overshoot, settling_time, band_ratio를 함께 플롯
        - Moving average (10-ep) 적용
        """
        if not self.episode_components:
            return

        try:
            episodes = [c["episode"] for c in self.episode_components]
            reward_scores = [c["reward_score"] for c in self.episode_components]
            r_centereds = [c["r_centered"] for c in self.episode_components]
            r_baselines = [c["r_baseline"] for c in self.episode_components]
            overshoots = [c["overshoot"] for c in self.episode_components]
            settling_times = [c["settling_time"] for c in self.episode_components]
            band_ratios = [c["band_ratio"] for c in self.episode_components]
            rewards = [c["reward"] for c in self.episode_components]

            # Moving average 계산 (10-ep)
            ma_window = 10
            reward_scores_ma = self._moving_average(reward_scores, ma_window)
            r_centereds_ma = self._moving_average(r_centereds, ma_window)
            overshoots_ma = self._moving_average(overshoots, ma_window)
            settling_times_ma = self._moving_average(settling_times, ma_window)
            band_ratios_ma = self._moving_average(band_ratios, ma_window)

            # ========== 그래프 1: 보상 구성 요소 (왼쪽 y축) ==========
            fig, ax1 = plt.subplots(figsize=(14, 7))

            # 원시 데이터 (투명하게)
            ax1.plot(
                episodes,
                reward_scores,
                "b-",
                alpha=0.3,
                linewidth=1,
                label="reward_score (raw)",
            )
            ax1.plot(
                episodes,
                r_centereds,
                "g-",
                alpha=0.3,
                linewidth=1,
                label="r_centered (raw)",
            )
            ax1.plot(
                episodes,
                r_baselines,
                "r--",
                alpha=0.3,
                linewidth=1,
                label="r_baseline (raw)",
            )

            # Moving average (선명하게)
            ax1.plot(
                episodes,
                reward_scores_ma,
                "b-",
                linewidth=2.5,
                marker="o",
                markersize=4,
                label=f"reward_score (MA{ma_window})",
            )
            ax1.plot(
                episodes,
                r_centereds_ma,
                "g-",
                linewidth=2.5,
                marker="s",
                markersize=4,
                label=f"r_centered (MA{ma_window})",
            )
            ax1.plot(
                episodes,
                r_baselines,
                "r--",
                linewidth=2,
                alpha=0.7,
                label="r_baseline (EWMA)",
            )

            ax1.set_xlabel("Episode", fontsize=12)
            ax1.set_ylabel("Reward Components", fontsize=12, color="k")
            ax1.tick_params(axis="y", labelcolor="k")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper left", fontsize=10)

            # ========== 그래프 2: 제어 성능 지표 (오른쪽 y축) ==========
            ax2 = ax1.twinx()

            # 원시 데이터 (투명하게)
            ax2.plot(
                episodes,
                overshoots,
                "orange",
                alpha=0.3,
                linewidth=1,
                label="overshoot % (raw)",
            )
            ax2.plot(
                episodes,
                settling_times,
                "purple",
                alpha=0.3,
                linewidth=1,
                label="settling_time (raw)",
            )
            ax2.plot(
                episodes,
                [r * 100 for r in band_ratios],
                "brown",
                alpha=0.3,
                linewidth=1,
                label="band_ratio % (raw)",
            )

            # Moving average
            ax2.plot(
                episodes,
                overshoots_ma,
                "orange",
                linewidth=2.5,
                marker="^",
                markersize=4,
                label=f"overshoot % (MA{ma_window})",
            )
            ax2.plot(
                episodes,
                settling_times_ma,
                "purple",
                linewidth=2.5,
                marker="v",
                markersize=4,
                label=f"settling_time (MA{ma_window})",
            )
            ax2.plot(
                episodes,
                [r * 100 for r in band_ratios_ma],
                "brown",
                linewidth=2.5,
                marker="d",
                markersize=4,
                label=f"band_ratio % (MA{ma_window})",
            )

            ax2.set_ylabel("Control Performance Metrics", fontsize=12, color="k")
            ax2.tick_params(axis="y", labelcolor="k")
            ax2.legend(loc="upper right", fontsize=10)

            plt.title(
                "Reward Components & Control Performance Over Episodes",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()

            filename = os.path.join(
                self.log_dir, "episode_reward_components.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"   📈 PNG: episode_reward_components.png")

            # ========== 그래프 3: 최종 보상과 reward_score 비교 ==========
            fig, ax = plt.subplots(figsize=(14, 6))

            # 원시 데이터
            ax.plot(
                episodes,
                reward_scores,
                "b-",
                alpha=0.3,
                linewidth=1,
                label="reward_score (raw)",
            )
            ax.plot(
                episodes,
                rewards,
                "g-",
                alpha=0.3,
                linewidth=1,
                label="reward (tanh 적용, raw)",
            )

            # Moving average
            ax.plot(
                episodes,
                reward_scores_ma,
                "b-",
                linewidth=2.5,
                marker="o",
                markersize=4,
                label=f"reward_score (MA{ma_window})",
            )
            ax.plot(
                episodes,
                self._moving_average(rewards, ma_window),
                "g-",
                linewidth=2.5,
                marker="s",
                markersize=4,
                label=f"reward (MA{ma_window})",
            )

            # 기준선 (0)
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)

            ax.set_xlabel("Episode", fontsize=12)
            ax.set_ylabel("Reward Value", fontsize=12)
            ax.set_title(
                "Reward Score vs Final Reward (tanh clipped)",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=10)

            plt.tight_layout()
            filename2 = os.path.join(
                self.log_dir, "episode_reward_comparison.png"
            )
            plt.savefig(filename2, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"   📈 PNG: episode_reward_comparison.png")

        except Exception as e:
            print(f"   ⚠️ 보상 구성 요소 그래프 생성 실패: {e}")
            import traceback
            traceback.print_exc()

