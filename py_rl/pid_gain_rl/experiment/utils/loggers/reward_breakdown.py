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

    def _compute_moving_average(self, values, window):
        if not values:
            return []
        window = max(1, int(window))
        ma = []
        for idx in range(len(values)):
            start = max(0, idx - window + 1)
            window_vals = values[start : idx + 1]
            ma.append(float(np.mean(window_vals)))
        return ma

    def save_episode_rewards(self, episode_rewards, ma_window=50):
        """에피소드별 보상을 CSV로 저장"""
        moving_avg = self._compute_moving_average(episode_rewards, ma_window)
        with open(self.episode_rewards_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["episode", "reward", f"reward_ma_{ma_window}"]
            )  # 헤더
            for i, reward in enumerate(episode_rewards, 1):
                ma_val = moving_avg[i - 1] if moving_avg else float(reward)
                writer.writerow([i, float(reward), ma_val])

    def generate_episode_reward_graph(self, episode_rewards, ma_window=50):
        """에피소드별 보상 그래프를 PNG로 저장"""
        if not episode_rewards:
            return

        try:
            episodes = list(range(1, len(episode_rewards) + 1))
            moving_avg = self._compute_moving_average(
                episode_rewards, ma_window
            )

            plt.figure(figsize=(12, 6))
            plt.plot(
                episodes,
                episode_rewards,
                color="tab:blue",
                linewidth=2,
                marker="o",
                markersize=4,
                label="Episode Reward",
            )
            if moving_avg:
                plt.plot(
                    episodes,
                    moving_avg,
                    color="tab:orange",
                    linewidth=2.5,
                    linestyle="-",
                    label=f"Moving Avg (window={ma_window})",
                )
            plt.xlabel("Episode", fontsize=18)
            plt.ylabel("Episode Reward", fontsize=18)
            plt.title(
                "Episode Rewards Over Time", fontsize=20, fontweight="bold"
            )
            plt.tick_params(labelsize=15)  # 축 눈금 폰트 크기
            plt.grid(True, alpha=0.3)
            if len(episode_rewards) > 1:
                avg_reward = float(np.mean(episode_rewards))
                plt.axhline(
                    y=avg_reward,
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    label=f"Mean: {avg_reward:.2f}",
                )
            if len(plt.gca().get_lines()) > 0:
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
        plt.title(f"Average prog per episode ({start_ep}-{end_ep})", fontsize=20, fontweight="bold")
        plt.xlabel("Episode", fontsize=18)
        plt.ylabel("prog", fontsize=18)
        plt.tick_params(labelsize=15)
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
        plt.title(f"Average in_band_now per episode ({start_ep}-{end_ep})", fontsize=20, fontweight="bold")
        plt.xlabel("Episode", fontsize=18)
        plt.ylabel("in_band_now (ratio)", fontsize=18)
        plt.tick_params(labelsize=15)
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
        plt.title(f"Average |de/dt| per episode ({start_ep}-{end_ep})", fontsize=20, fontweight="bold")
        plt.xlabel("Episode", fontsize=18)
        plt.ylabel("|de/dt|", fontsize=18)
        plt.tick_params(labelsize=15)
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
        plt.title(f"Average |Δu| per episode ({start_ep}-{end_ep})", fontsize=20, fontweight="bold")
        plt.xlabel("Episode", fontsize=18)
        plt.ylabel("|Δu|", fontsize=18)
        plt.tick_params(labelsize=15)
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
        plt.title(f"Average reward per episode ({start_ep}-{end_ep})", fontsize=20, fontweight="bold")
        plt.xlabel("Episode", fontsize=18)
        plt.ylabel("reward", fontsize=18)
        plt.tick_params(labelsize=15)
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
        os.makedirs(self.log_dir, exist_ok=True)
        # 데이터가 없으면 실행하지 않음
        if not self.rows:
            return

        # CSV는 항상 저장 (메모리 절약)
        self.save_reward_breakdown_csv()

        # force=True일 때만 PNG 생성 (최종에만)
        if force:
            # 에피소드별 보상 저장 및 그래프 생성
            if episode_rewards is not None:
                ma_window = 50
                self.save_episode_rewards(episode_rewards, ma_window=ma_window)
                self.generate_episode_reward_graph(
                    episode_rewards, ma_window=ma_window
                )

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
        rmse,
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
            rmse: RMSE (N)
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
                "rmse": float(rmse),
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
                    "rmse",
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
        🆕 보상 구성 요소 시각화 그래프 생성 (슬림화)
        - 좌측 y축: r_baseline (EWMA), reward_score (MA10)
        - 우측 y축: RMSE, Band Ratio, Overshoot %, Settling Time
        - Raw 데이터 제거, MA10만 표시
        """
        if not self.episode_components:
            return

        try:
            episodes = [c["episode"] for c in self.episode_components]
            reward_scores = [c["reward_score"] for c in self.episode_components]
            r_baselines = [c["r_baseline"] for c in self.episode_components]
            overshoots = [c["overshoot"] for c in self.episode_components]
            settling_times = [c["settling_time"] for c in self.episode_components]
            band_ratios = [c["band_ratio"] for c in self.episode_components]
            rmses = [c.get("rmse", 0.0) for c in self.episode_components]

            # Moving average 계산 (10-ep)
            ma_window = 10
            reward_scores_ma = self._moving_average(reward_scores, ma_window)
            overshoots_ma = self._moving_average(overshoots, ma_window)
            settling_times_ma = self._moving_average(settling_times, ma_window)
            band_ratios_ma = self._moving_average(band_ratios, ma_window)
            rmses_ma = self._moving_average(rmses, ma_window)

            # ========== 그래프: 6개 지표만 표시 ==========
            fig, ax1 = plt.subplots(figsize=(14, 7))

            # ========== 좌측 y축: r_baseline, reward_score(MA10) ==========
            # r_baseline (EWMA) - 점선, 빨간색
            ax1.plot(
                episodes,
                r_baselines,
                "r--",
                linewidth=2.5,
                alpha=0.8,
                label="r_baseline (EWMA)",
            )
            
            # reward_score (MA10) - 실선, 파란색
            ax1.plot(
                episodes,
                reward_scores_ma,
                "b-",
                linewidth=2.5,
                marker="o",
                markersize=4,
                markevery=max(1, len(episodes) // 20),  # 마커 간격 조절
                label=f"reward_score (MA{ma_window})",
            )

            ax1.set_xlabel("Episode", fontsize=18)
            ax1.set_ylabel("Reward Score / Baseline", fontsize=18, color="k")
            ax1.tick_params(axis="y", labelcolor="k", labelsize=15)
            ax1.tick_params(axis="x", labelsize=15)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper left", fontsize=16)

            # ========== 우측 y축: RMSE, Band Ratio, Overshoot %, Settling Time ==========
            ax2 = ax1.twinx()

            # 고정 색상 팔레트
            colors = {
                "rmse": "purple",
                "band_ratio": "green",
                "overshoot": "orange",
                "settling_time": "brown",
            }

            # RMSE (MA10)
            ax2.plot(
                episodes,
                rmses_ma,
                color=colors["rmse"],
                linewidth=2.5,
                marker="^",
                markersize=4,
                markevery=max(1, len(episodes) // 20),
                label=f"RMSE (MA{ma_window})",
            )

            # Band Ratio (MA10) - % 단위로 변환
            ax2.plot(
                episodes,
                [r * 100 for r in band_ratios_ma],
                color=colors["band_ratio"],
                linewidth=2.5,
                marker="s",
                markersize=4,
                markevery=max(1, len(episodes) // 20),
                label=f"Band Ratio % (MA{ma_window})",
            )

            # Overshoot % (MA10)
            ax2.plot(
                episodes,
                overshoots_ma,
                color=colors["overshoot"],
                linewidth=2.5,
                marker="d",
                markersize=4,
                markevery=max(1, len(episodes) // 20),
                label=f"Overshoot % (MA{ma_window})",
            )

            # Settling Time (MA10)
            ax2.plot(
                episodes,
                settling_times_ma,
                color=colors["settling_time"],
                linewidth=2.5,
                marker="v",
                markersize=4,
                markevery=max(1, len(episodes) // 20),
                label=f"Settling Time (MA{ma_window})",
            )

            ax2.set_ylabel("Control Performance Metrics", fontsize=18, color="k")
            ax2.tick_params(axis="y", labelcolor="k", labelsize=15)
            ax2.legend(loc="upper right", fontsize=16)

            plt.title(
                "Reward Components & Control Performance Over Episodes",
                fontsize=20,
                fontweight="bold",
            )
            plt.tight_layout()

            filename = os.path.join(
                self.log_dir, "episode_reward_components.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"   📈 PNG: episode_reward_components.png")

        except Exception as e:
            print(f"   ⚠️ 보상 구성 요소 그래프 생성 실패: {e}")
            import traceback
            traceback.print_exc()
