"""
제어공학 지표 계산 및 저장
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from ...config.constants import Constants

class ControlPerformanceLogger:
    """
    제어공학 지표를 계산하고 저장하는 클래스
    - 기본 성능 지표: RMSE, Steady-State Error, Rise Time, Settling Time, Overshoot, IAE
    - 제어 노력 지표: Input RMS, Total Variation
    - 안정성 지표: Band Ratio, Error Variance
    """

    def __init__(self, log_dir):
        self.base_log_dir = log_dir
        # 실행별 고유 폴더 생성
        # log_dir는 이미 learning_done_YYMMDD_HHhMMm 형태
        self.log_dir = log_dir
        self.control_perf_dir = os.path.join(
            self.log_dir, "control_performance"
        )
        os.makedirs(self.control_perf_dir, exist_ok=True)

        # 폰트 설정 (논문용 Times New Roman)
        self._setup_fonts()

        # 기본 데이터 저장용 리스트들
        self.time_data = []
        self.force_data = []
        self.target_data = []
        self.error_data = []
        self.pi_output_data = []

        # 추가 지표용 데이터 저장
        self.pid_gains_history = []  # PID gain 변화 추적용
        self.input_data = []  # 제어 입력 데이터

        # 에피소드별 지표 저장
        self.episode_metrics = []

        print(f"📁 Control Performance 저장 폴더: {self.control_perf_dir}")

    def _setup_fonts(self):
        """폰트 설정 (기본 크기 +6pt)"""
        try:
            import matplotlib.pyplot as plt

            plt.rcParams["font.family"] = "Times New Roman"
            # 기본 크기 +6pt 설정
            plt.rcParams["axes.labelsize"] = 16  # 축 레이블 (기본 10 → 16)
            plt.rcParams["axes.titlesize"] = 16  # 서브플롯 제목 (기본 10 → 16)
            plt.rcParams["figure.titlesize"] = 18  # 그림 제목 (기본 12 → 18)
            plt.rcParams["xtick.labelsize"] = 15  # x축 눈금 (기본 9 → 15)
            plt.rcParams["ytick.labelsize"] = 15  # y축 눈금 (기본 9 → 15)
            plt.rcParams["legend.fontsize"] = 16  # 범례 (기본 10 → 16)
        except Exception as e:
            pass  # 폰트 설정 실패해도 그래프는 생성됨
            print("기본 폰트 사용")

    def add_data_point(
        self, time, force, target, control_effort, pi_output, pid_gains=None
    ):
        """실시간 데이터 포인트 추가 (1kHz에서 호출)
        Args:
            time: 시간 (초)
            force: 현재 힘 (N)
            target: 목표 힘 (N)
            control_effort: 제어 노력 (PID gain 합) - 사용하지 않음
            pi_output: PID 출력 (실제 제어 입력)
            pid_gains: PID gain 값들 [Kp, Ki, Kd] (선택사항)
        """
        self.time_data.append(time)
        self.force_data.append(force)
        self.target_data.append(target)
        self.error_data.append(abs(force - target))
        self.pi_output_data.append(pi_output)

        # PID gain 정보 저장 (추가 지표 계산용)
        if pid_gains is not None:
            self.pid_gains_history.append(pid_gains.copy())
            # 제어 입력으로 실제 pi_output 사용
            self.input_data.append(np.sum(np.abs(pid_gains)))
        else:
            self.pid_gains_history.append([0.0, 0.0, 0.0])
            self.input_data.append(0.0)

    def calculate_episode_metrics(self, episode_num):
        """에피소드별 제어공학 지표 계산 (논문용 10개 핵심 지표)
        Returns:
            dict: 10개 제어공학 지표
        """
        if not self.time_data:
            return None

        # 논문용 10개 핵심 지표만 계산
        metrics = {
            "episode": episode_num,
            # 기본 성능 지표 (6개)
            "rmse": self._calculate_rmse(),                          # RMSE
            "steady_state_error": self._calculate_steady_state_error(),  # Steady-State Error
            "rise_time": self._calculate_rise_time(),               # Rise Time
            "settling_time": self._calculate_settling_time(),       # Settling Time
            "overshoot": self._calculate_overshoot(),                # Overshoot
            "iae": self._calculate_iae(),                            # IAE
            # 제어 노력 지표 (2개)
            "input_rms": self._calculate_input_rms(),               # Input RMS
            "total_variation": self._calculate_total_variation(),   # Total Variation
            # 안정성 지표 (2개)
            "band_ratio": self._calculate_success_rate(),            # Band Ratio (목표 범위 내 유지 비율)
            "error_variance": self._calculate_error_variance(),     # Error Variance
        }
        metrics["success_rate"] = metrics["band_ratio"]

        self.episode_metrics.append(metrics)
        return metrics

    def _calculate_rmse(self):
        """RMSE 계산"""
        if not self.error_data:
            return None
        return np.sqrt(np.mean(np.square(self.error_data)))

    def _calculate_steady_state_error(self):
        """Steady State Error 계산 (마지막 10% 구간의 평균 절대 오차)"""
        if not self.error_data:
            return None
        last_10_percent = max(1, int(len(self.error_data) * 0.1))
        return np.mean(self.error_data[-last_10_percent:])

    def _calculate_rise_time(self):
        """Stabilization Time 계산 (목표값 ±5% 밴드 내 최초 진입 시간)"""
        if (
            not self.force_data
            or not self.target_data
            or len(self.force_data) < 10
        ):
            return None

        target = self.target_data[0]  # default -50N
        force_array = np.array(self.force_data)
        time_array = np.array(self.time_data)

        # ±5% 밴드 정의
        band = abs(target) * Constants.BAND_RATIO_TOLERANCE
        target_min = target - band
        target_max = target + band

        # 밴드 내 진입 지점 찾기
        in_band = (force_array >= target_min) & (force_array <= target_max)
        band_indices = np.where(in_band)[0]

        if len(band_indices) > 0:
            # 첫 번째로 밴드에 진입한 시간
            return float(time_array[band_indices[0]] - time_array[0])

        return None

    def _calculate_settling_time(self):
        """Settling Time 계산 (연속 유지 기준) - 에피소드 보상과 동일한 기준"""
        if not self.force_data or not self.target_data:
            return None

        target = self.target_data[0]
        band = abs(target) * Constants.BAND_RATIO_TOLERANCE  # ±5% 밴드
        force_array = np.array(self.force_data)
        time_array = np.array(self.time_data)

        # 연속 유지 구간 찾기 (2초 연속 유지)
        within = np.abs(force_array - target) <= band
        hold_duration = int(2.0 * 1000)  # 2초 연속 유지 (1kHz 기준)

        run_length = 0
        settling_time = None
        for k, in_band in enumerate(within):
            if in_band:
                run_length += 1
                if run_length >= hold_duration:
                    settling_time = max(0.0, (k - hold_duration) / 1000.0)
                    break
            else:
                run_length = 0

        return float(settling_time) if settling_time is not None else None

    def _calculate_overshoot(self):
        """Overshoot 계산 (목표값을 넘어선 최대 편차)"""
        if not self.force_data or not self.target_data:
            return None

        target = self.target_data[0]  # default -50N
        force_array = np.array(self.force_data)

        # 목표값보다 더 나쁜 방향으로의 최대 편차 계산
        if target < 0:  # 음수 목표값 (압축력)
            # 더 큰 음수값 (더 큰 압축력)을 찾음
            extreme_force = np.min(force_array)  # 가장 작은 값 (가장 큰 음수)
            if extreme_force < target:
                # 오버슈트 = (목표 - 실제) / |목표| * 100
                # 예: (-40 - (-60)) / 40 * 100 = 20/40 * 100 = 50%
                overshoot = ((target - extreme_force) / abs(target)) * 100.0
                return float(overshoot)
        else:  # 양수 목표값
            extreme_force = np.max(force_array)
            if extreme_force > target:
                # 오버슈트 = (실제 - 목표) / |목표| * 100
                # 예: (60 - 40) / 40 * 100 = 20/40 * 100 = 50%
                overshoot = ((extreme_force - target) / abs(target)) * 100.0
                return float(overshoot)

        return 0.0

    def _calculate_iae(self):
        """IAE (Integral Absolute Error) 계산 - 연마 공정에서 편차 누적"""
        if not self.error_data or not self.time_data:
            return None
        dt = (
            np.mean(np.diff(self.time_data))
            if len(self.time_data) > 1
            else 0.001
        )
        return float(np.sum(np.abs(self.error_data)) * dt)

    def _calculate_input_rms(self):
        """Input RMS 계산 - PID gain 합의 RMS 값 (제어 노력 분리)"""
        if not self.input_data:
            return None
        arr = np.asarray(self.input_data, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(arr))))

    def _calculate_total_variation(self):
        """Total Variation 계산 - 실제 제어 출력 변화 총량 (밸브 마모와 직결)"""
        if len(self.pi_output_data) < 2:
            return None
        return float(np.sum(np.abs(np.diff(self.pi_output_data))))

    def _calculate_success_rate(self):
        """Success Rate 계산 - 목표 범위 내 유지 비율"""
        if not self.force_data or not self.target_data or not self.time_data:
            return None
        target = self.target_data[0]  # default -50N
        tolerance = (
            abs(target) * 0.02
        )  # ±2% 오차 범위 (±0.8N) - 더 엄격한 기준
        in_band = np.abs(np.array(self.force_data) - target) <= tolerance
        return float(np.sum(in_band) / len(in_band))

    def _calculate_error_variance(self):
        """Error Variance 계산 - 오차 분산 (안정성 지표)"""
        if not self.error_data:
            return None
        return float(np.var(self.error_data))

    def save_episode_metrics(self, episode_num):
        """에피소드별 지표를 CSV로 저장"""
        metrics = self.calculate_episode_metrics(episode_num)
        if metrics is None:
            return

        # 개별 지표별 CSV 저장
        for metric_name, value in metrics.items():
            if metric_name == "episode" or value is None:
                continue

            csv_path = os.path.join(
                self.control_perf_dir, f"{metric_name}.csv"
            )
            file_exists = os.path.exists(csv_path)

            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["episode", metric_name])
                writer.writerow([episode_num, value])

    def save_performance_summary(self):
        """전체 성능 요약 저장 (논문용 10개 핵심 지표)"""
        if not self.episode_metrics:
            return

        summary_path = os.path.join(
            self.control_perf_dir, "performance_summary.csv"
        )

        with open(summary_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Metric", "Mean", "Std", "Min", "Max", "Unit", "Description"]
            )

            # 논문용 10개 핵심 지표만
            all_metrics = [
                # 기본 성능 지표 (6개)
                "rmse",                     # RMSE
                "steady_state_error",       # Steady-State Error
                "rise_time",                # Rise Time
                "settling_time",            # Settling Time
                "overshoot",                # Overshoot
                "iae",                      # IAE
                # 제어 노력 지표 (2개)
                "input_rms",                # Input RMS
                "total_variation",          # Total Variation
                # 안정성 지표 (2개)
                "band_ratio",               # Band Ratio (목표 범위 내 유지 비율)
                "error_variance",           # Error Variance
            ]

            for metric_name in all_metrics:
                values = [
                    ep[metric_name]
                    for ep in self.episode_metrics
                    if ep[metric_name] is not None
                ]

                if values:
                    writer.writerow(
                        [
                            metric_name,
                            f"{np.mean(values):.4f}",
                            f"{np.std(values):.4f}",
                            f"{np.min(values):.4f}",
                            f"{np.max(values):.4f}",
                            self._get_metric_unit(metric_name),
                            self._get_metric_description(metric_name),
                        ]
                    )

        print(f"📊 성능 요약 저장 완료: {summary_path}")

    def _get_metric_unit(self, metric_name):
        """지표별 단위 반환 (10개 핵심 지표)"""
        units = {
            # 기본 성능 지표
            "rmse": "N",
            "steady_state_error": "N",
            "rise_time": "s",
            "settling_time": "s",
            "overshoot": "%",
            "iae": "N·s",
            # 제어 노력 지표
            "input_rms": "N",
            "total_variation": "N",
            # 안정성 지표
            "band_ratio": "-",
            "success_rate": "-",
            "error_variance": "N²",
        }
        return units.get(metric_name, "")

    def _get_metric_description(self, metric_name):
        """지표별 설명 반환 (10개 핵심 지표)"""
        descriptions = {
            # 기본 성능 지표
            "rmse": "Root Mean Square Error - 제어 정확도",
            "steady_state_error": "Steady State Error - 정상상태 오차",
            "rise_time": "Rise Time - 상승시간 (10%→90%)",
            "settling_time": "Settling Time - 정착시간 (±5%)",
            "overshoot": "Overshoot - 오버슈트 (%)",
            "iae": "Integral Absolute Error - 절대 오차 적분",
            # 제어 노력 지표
            "input_rms": "Input RMS - 제어 입력 RMS",
            "total_variation": "Total Variation - 총 변화량 (밸브 마모)",
            # 안정성 지표
            "band_ratio": "Band Ratio - 목표 범위 내 유지 비율",
            "success_rate": "Success Rate - 목표 범위 내 유지 비율 (동일 지표)",
            "error_variance": "Error Variance - 오차 분산 (안정성)",
        }
        return descriptions.get(metric_name, "")

    def generate_plots(self):
        """각 지표별 시각화 생성 (논문용 10개 핵심 지표)"""
        if not self.episode_metrics:
            return

        print("📈 논문용 고품질 그래프 생성 중...")

        # 논문용 10개 핵심 지표만
        all_metrics = [
            # 기본 성능 지표 (6개)
            "rmse",
            "steady_state_error",
            "rise_time",
            "settling_time",
            "overshoot",
            "iae",
            # 제어 노력 지표 (2개)
            "input_rms",
            "total_variation",
            # 안정성 지표 (2개)
            "band_ratio",
            "error_variance",
        ]

        for metric_name in all_metrics:
            self._plot_metric(metric_name)

        # 추가로 종합 대시보드 생성
        self._generate_comprehensive_dashboard()

        # Step 축 지표들도 생성 (논문용)
        self._generate_step_based_plots()

        print(f"✅ 총 {len(all_metrics)}개 지표 그래프 생성 완료")

    def _plot_metric(self, metric_name):
        """개별 지표 시각화 (논문용 고품질)"""
        values = [
            ep[metric_name]
            for ep in self.episode_metrics
            if ep[metric_name] is not None
        ]
        episodes = [
            ep["episode"]
            for ep in self.episode_metrics
            if ep[metric_name] is not None
        ]

        if not values:
            return

        # 폰트 설정 재적용 (각 그래프마다)
        self._setup_fonts()

        plt.figure(figsize=(12, 8))
        plt.plot(
            episodes,
            values,
            "b-",
            linewidth=3,
            marker="o",
            markersize=6,
            markerfacecolor="blue",
            markeredgecolor="darkblue",
            markeredgewidth=1,
        )
        plt.xlabel("Episode Number", fontweight="bold", fontsize=18)
        plt.ylabel(f"{metric_name.upper()}", fontweight="bold", fontsize=18)
        plt.title(f"{metric_name.upper()} Over Episodes", fontweight="bold", fontsize=20)
        plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

        # 평균선 추가
        if len(values) > 1:
            avg_value = np.mean(values)
            std_value = np.std(values)
            plt.axhline(
                y=avg_value,
                color="r",
                linestyle="--",
                alpha=0.8,
                linewidth=2,
                label=f"Mean: {avg_value:.4f}±{std_value:.4f}",
            )
            plt.legend(loc="best", frameon=True, fancybox=True, shadow=True, fontsize=16)

        # 축 범위 조정
        plt.xlim(min(episodes) - 0.5, max(episodes) + 0.5)

        png_path = os.path.join(self.control_perf_dir, f"{metric_name}.png")
        plt.tight_layout()
        plt.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"  📊 {metric_name.upper()} 그래프 저장: {png_path}")

    def _generate_comprehensive_dashboard(self):
        """종합 대시보드 생성 (논문용)"""
        if len(self.episode_metrics) < 2:
            return

        # 폰트 설정 재적용
        self._setup_fonts()

        # 3x4 서브플롯 생성 (10개 지표)
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle(
            "PID Gain Optimization Performance Dashboard", fontweight="bold", fontsize=20
        )

        # 논문용 10개 핵심 지표만
        key_metrics = [
            "rmse",
            "steady_state_error",
            "rise_time",
            "settling_time",
            "overshoot",
            "iae",
            "input_rms",
            "total_variation",
            "band_ratio",
            "error_variance",
        ]

        for i, metric_name in enumerate(key_metrics):
            row, col = i // 4, i % 4
            ax = axes[row, col]

            values = [
                ep[metric_name]
                for ep in self.episode_metrics
                if ep[metric_name] is not None
            ]
            episodes = [
                ep["episode"]
                for ep in self.episode_metrics
                if ep[metric_name] is not None
            ]

            if values:
                ax.plot(
                    episodes,
                    values,
                    "b-",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )
                ax.set_title(f"{metric_name.upper()}", fontweight="bold", fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=15)  # 축 눈금 폰트 크기

                # 평균선 추가
                if len(values) > 1:
                    avg_value = np.mean(values)
                    ax.axhline(
                        y=avg_value, color="r", linestyle="--", alpha=0.7
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{metric_name.upper()}", fontweight="bold")

        # 빈 서브플롯 숨기기 (3x4 레이아웃)
        for i in range(len(key_metrics), 12):
            row, col = i // 4, i % 4
            if row < 3:  # 3x4 레이아웃 검증
                axes[row, col].set_visible(False)

        dashboard_path = os.path.join(
            self.control_perf_dir, "comprehensive_dashboard.png"
        )
        plt.tight_layout()
        plt.savefig(
            dashboard_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"📊 종합 대시보드 저장: {dashboard_path}")

    def _generate_step_based_plots(self):
        """Step 축 지표 그래프 생성 (논문용 - 에피소드 내부 시간적 추세)"""
        if not self.time_data or not self.force_data:
            return

        print("📈 Step 축 지표 그래프 생성 중...")

        # 폰트 설정 재적용
        self._setup_fonts()

        # 1. Force Tracking Curve (목표힘 vs 실제힘)
        self._plot_force_tracking_curve()

        # 2. Error Time Series (순간 오차)
        self._plot_error_time_series()

        # 3. Control Input Time Series (제어 입력)
        self._plot_control_input_series()

        # 4. Reward Breakdown (보상 구성 요소)
        self._plot_reward_breakdown()

        # 5. Step 축 종합 대시보드
        self._generate_step_dashboard()

        print("✅ Step 축 지표 그래프 생성 완료")

    def _plot_force_tracking_curve(self):
        """Force Tracking Curve (목표힘 vs 실제힘)"""
        if not self.force_data or not self.target_data or not self.time_data:
            return

        plt.figure(figsize=(14, 8))
        time_array = np.array(self.time_data)
        force_array = np.array(self.force_data)
        target_array = np.array(self.target_data)

        plt.plot(
            time_array,
            target_array,
            "r--",
            linewidth=3,
            label="Target Force",
            alpha=0.8,
        )
        plt.plot(
            time_array,
            force_array,
            "b-",
            linewidth=2,
            label="Actual Force",
            alpha=0.9,
        )
        tolerance = 0.05 * target_array[0] if len(target_array) > 0 else 2.25
        plt.fill_between(
            time_array,
            target_array - tolerance,
            target_array + tolerance,
            alpha=0.2,
            color="green",
            label="±5% Tolerance Band",
        )

        plt.xlabel("Time (s)", fontweight="bold", fontsize=18)
        plt.ylabel("Force (N)", fontweight="bold", fontsize=18)
        plt.title("Force Tracking Performance (Step-based)", fontweight="bold", fontsize=20)
        plt.legend(loc="best", frameon=True, fancybox=True, shadow=True, fontsize=16)
        plt.grid(True, alpha=0.3)

        png_path = os.path.join(
            self.control_perf_dir, "force_tracking_curve.png"
        )
        plt.tight_layout()
        plt.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"  📊 Force Tracking Curve 저장: {png_path}")

    def _plot_error_time_series(self):
        """Error Time Series (순간 오차)"""
        if not self.error_data or not self.time_data:
            return

        plt.figure(figsize=(14, 8))
        time_array = np.array(self.time_data)
        error_array = np.array(self.error_data)
        target_array = np.array(self.target_data)

        plt.plot(
            time_array, error_array, "r-", linewidth=2, label="Absolute Error"
        )
        tolerance = 0.05 * target_array[0] if len(target_array) > 0 else 2.25
        plt.axhline(
            y=tolerance,
            color="g",
            linestyle="--",
            alpha=0.7,
            label="±5% Tolerance",
        )
        plt.axhline(y=-tolerance, color="g", linestyle="--", alpha=0.7)
        plt.fill_between(
            time_array,
            -tolerance,
            tolerance,
            alpha=0.1,
            color="green",
            label="Tolerance Band",
        )

        plt.xlabel("Time (s)", fontweight="bold", fontsize=18)
        plt.ylabel("Force Error (N)", fontweight="bold", fontsize=18)
        plt.title("Force Error Time Series (Step-based)", fontweight="bold", fontsize=20)
        plt.legend(loc="best", frameon=True, fancybox=True, shadow=True, fontsize=16)
        plt.grid(True, alpha=0.3)

        png_path = os.path.join(self.control_perf_dir, "error_time_series.png")
        plt.tight_layout()
        plt.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"  📊 Error Time Series 저장: {png_path}")

    def _plot_control_input_series(self):
        """Control Input Time Series (제어 입력)"""
        if not self.input_data or not self.time_data:
            return

        plt.figure(figsize=(14, 8))
        time_array = np.array(self.time_data)
        input_array = np.array(self.input_data)

        plt.plot(
            time_array,
            input_array,
            "purple",
            linewidth=2,
            label="Control Input (PID Gain Sum)",
        )
        plt.xlabel("Time (s)", fontweight="bold", fontsize=18)
        plt.ylabel("Control Input", fontweight="bold", fontsize=18)
        plt.title("Control Input Time Series (Step-based)", fontweight="bold", fontsize=20)
        plt.legend(loc="best", frameon=True, fancybox=True, shadow=True, fontsize=16)
        plt.grid(True, alpha=0.3)

        png_path = os.path.join(
            self.control_perf_dir, "control_input_series.png"
        )
        plt.tight_layout()
        plt.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"  📊 Control Input Series 저장: {png_path}")

    def _plot_reward_breakdown(self):
        """Reward Breakdown (보상 구성 요소) - Step 단위"""
        if not self.time_data or not self.force_data or not self.target_data:
            return

        # Step 단위 보상 구성 요소 계산
        time_array = np.array(self.time_data)
        force_array = np.array(self.force_data)
        target_array = np.array(self.target_data)
        error_array = np.abs(force_array - target_array)

        # 1. Progress Reward (목표에 가까워질수록 높은 보상)
        progress_reward = np.exp(
            -error_array / 5.0
        )  # 오차가 작을수록 높은 보상

        # 2. In-band Reward (±5% 범위 내에 있을 때 보상)
        tolerance = target_array[0] * 0.05
        in_band = np.abs(force_array - target_array) <= tolerance
        in_band_reward = in_band.astype(float)

        # 3. Error Penalty (오차에 대한 페널티)
        error_penalty = -error_array / 10.0

        # 4. Stability Reward (안정성 보상)
        if len(error_array) > 1:
            error_derivative = np.abs(
                np.diff(error_array, prepend=error_array[0])
            )
            stability_reward = np.exp(-error_derivative / 2.0)
        else:
            stability_reward = np.ones_like(error_array)

        plt.figure(figsize=(16, 10))

        # 서브플롯 1: Progress Reward
        plt.subplot(2, 2, 1)
        plt.plot(time_array, progress_reward, "b-", linewidth=2)
        plt.title("Progress Reward (Step-based)", fontweight="bold", fontsize=20)
        plt.xlabel("Time (s)", fontsize=18)
        plt.ylabel("Progress Reward", fontsize=18)
        plt.grid(True, alpha=0.3)

        # 서브플롯 2: In-band Reward
        plt.subplot(2, 2, 2)
        plt.plot(time_array, in_band_reward, "g-", linewidth=2)
        plt.title("In-band Reward (Step-based)", fontweight="bold", fontsize=20)
        plt.xlabel("Time (s)", fontsize=18)
        plt.ylabel("In-band Reward", fontsize=18)
        plt.grid(True, alpha=0.3)

        # 서브플롯 3: Error Penalty
        plt.subplot(2, 2, 3)
        plt.plot(time_array, error_penalty, "r-", linewidth=2)
        plt.title("Error Penalty (Step-based)", fontweight="bold", fontsize=20)
        plt.xlabel("Time (s)", fontsize=18)
        plt.ylabel("Error Penalty", fontsize=18)
        plt.grid(True, alpha=0.3)

        # 서브플롯 4: Stability Reward
        plt.subplot(2, 2, 4)
        plt.plot(time_array, stability_reward, "purple", linewidth=2)
        plt.title("Stability Reward (Step-based)", fontweight="bold", fontsize=20)
        plt.xlabel("Time (s)", fontsize=18)
        plt.ylabel("Stability Reward", fontsize=18)
        plt.grid(True, alpha=0.3)

        png_path = os.path.join(
            self.control_perf_dir, "reward_breakdown_step.png"
        )
        plt.tight_layout()
        plt.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"  📊 Reward Breakdown 저장: {png_path}")

    def _generate_step_dashboard(self):
        """Step 축 종합 대시보드"""
        if not self.time_data or not self.force_data:
            return

        # 폰트 설정 재적용
        self._setup_fonts()

        # 2x2 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Step-based Performance Dashboard", fontweight="bold", fontsize=20)

        time_array = np.array(self.time_data)
        force_array = np.array(self.force_data)
        target_array = np.array(self.target_data)
        error_array = np.array(self.error_data)

        # 1. Force Tracking
        axes[0, 0].plot(
            time_array, target_array, "r--", linewidth=2, label="Target"
        )
        axes[0, 0].plot(
            time_array, force_array, "b-", linewidth=1.5, label="Actual"
        )
        axes[0, 0].set_title("Force Tracking", fontweight="bold", fontsize=16)
        axes[0, 0].set_xlabel("Time (s)", fontsize=18)
        axes[0, 0].set_ylabel("Force (N)", fontsize=18)
        axes[0, 0].legend(fontsize=16)
        axes[0, 0].tick_params(labelsize=15)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Error Time Series
        axes[0, 1].plot(time_array, error_array, "r-", linewidth=1.5)
        target_array = np.array(self.target_data)
        tolerance = 0.05 * target_array[0] if len(target_array) > 0 else 2.25
        axes[0, 1].axhline(y=tolerance, color="g", linestyle="--", alpha=0.7)
        axes[0, 1].axhline(y=-tolerance, color="g", linestyle="--", alpha=0.7)
        axes[0, 1].set_title("Error Time Series", fontweight="bold", fontsize=16)
        axes[0, 1].set_xlabel("Time (s)", fontsize=18)
        axes[0, 1].set_ylabel("Error (N)", fontsize=18)
        axes[0, 1].tick_params(labelsize=15)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Control Input
        if self.input_data:
            input_array = np.array(self.input_data)
            axes[1, 0].plot(time_array, input_array, "purple", linewidth=1.5)
            axes[1, 0].set_title("Control Input", fontweight="bold", fontsize=16)
            axes[1, 0].set_xlabel("Time (s)", fontsize=18)
            axes[1, 0].set_ylabel("Input", fontsize=18)
            axes[1, 0].tick_params(labelsize=15)
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Error Distribution
        axes[1, 1].hist(
            error_array, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        target_array = np.array(self.target_data)
        tolerance = 0.05 * target_array[0] if len(target_array) > 0 else 2.25
        axes[1, 1].axvline(
            x=tolerance,
            color="r",
            linestyle="--",
            alpha=0.7,
            label="±5% Tolerance",
        )
        axes[1, 1].axvline(x=-tolerance, color="r", linestyle="--", alpha=0.7)
        axes[1, 1].set_title("Error Distribution", fontweight="bold", fontsize=16)
        axes[1, 1].set_xlabel("Error (N)", fontsize=18)
        axes[1, 1].set_ylabel("Frequency", fontsize=18)
        axes[1, 1].legend(fontsize=16)
        axes[1, 1].tick_params(labelsize=15)
        axes[1, 1].grid(True, alpha=0.3)

        dashboard_path = os.path.join(
            self.control_perf_dir, "step_dashboard.png"
        )
        plt.tight_layout()
        plt.savefig(
            dashboard_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"📊 Step 축 대시보드 저장: {dashboard_path}")

    def reset_episode_data(self):
        """에피소드 데이터 초기화 (모든 데이터 변수 포함)"""
        # 기본 데이터
        self.time_data.clear()
        self.force_data.clear()
        self.target_data.clear()
        self.error_data.clear()
        self.pi_output_data.clear()

        # 추가 지표용 데이터
        self.pid_gains_history.clear()
        self.input_data.clear()

# =========================
# Learning Done Logger
# =========================
