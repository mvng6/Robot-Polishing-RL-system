"""
실시간 모니터링 GUI
"""
from queue import Empty, Full
from multiprocessing import Process, Queue

class RLRealtimeMonitor:
    def __init__(self, title="PID Gain RL Monitor", hz=10, rolling_window=30.0):
        self.hz = max(1, int(hz))
        self.rolling_window = float(rolling_window)
        self.q = Queue(maxsize=256)
        self.proc = None
        self.title = title

    def start(self):
        self.proc = Process(target=self._run, daemon=True)
        self.proc.start()

    def stop(self, timeout=2.0):
        if self.proc is None:
            return
        try:
            self.q.put_nowait({"type": "quit"})
        except Full:
            pass
        self.proc.join(timeout=timeout)

    def reset_force_buffers(self):
        try:
            self.q.put_nowait({"type": "reset_force"})
        except Full:
            pass

    def post_force(self, t_sec: float, current_f: float, desired_f: float):
        msg = {
            "type": "force",
            "t": float(t_sec),
            "cf": float(current_f),
            "df": float(desired_f),
        }
        try:
            self.q.put_nowait(msg)
        except Full:
            pass  # 최신만 유지

    def post_reward(self, episode: int, reward: float):
        msg = {"type": "reward", "ep": int(episode), "rew": float(reward)}
        try:
            self.q.put_nowait(msg)
        except Full:
            pass

    def post_pi_output(self, t_sec: float, pi_output: float):
        msg = {"type": "pi", "t": float(t_sec), "pi": float(pi_output)}
        try:
            self.q.put_nowait(msg)
        except Full:
            pass

    def _run(self):
        # 백엔드는 먼저 설정해야 함 (import 전에)
        import matplotlib
        
        tk_ok = True
        backend_name = "Agg"
        try:
            import tkinter  # noqa: F401
            matplotlib.use("TkAgg", force=True)  # 강제로 TkAgg 사용
            backend_name = "TkAgg"
            print(f"[Monitor] TkAgg backend enabled successfully")
        except Exception as e:
            tk_ok = False
            print(f"[Monitor] TkAgg failed: {e}, using Agg backend")
            matplotlib.use("Agg")

        # 설정된 백엔드로 pyplot import
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as np
        import matplotlib.gridspec as gridspec
        import time

        print(f"[Monitor] Active backend: {matplotlib.get_backend()}")
        
        tbuf, cbuf, dbuf = [], [], []
        latest_pi = None
        ep_idx, ep_rew = [], []

        if not tk_ok:
            print(
                "[Monitor] TkAgg unavailable (headless). "
                "Realtime window is disabled."
            )
            last_note = 0.0
            while True:
                try:
                    msg = self.q.get(timeout=1.0)
                except Empty:
                    if time.time() - last_note > 5.0:
                        print("[Monitor] Headless mode: plotting disabled.")
                        last_note = time.time()
                    continue
                tp = msg.get("type")
                if tp == "quit":
                    break
            return
        fig = plt.figure(figsize=(9, 7.2))
        fig.suptitle(self.title)
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.0, 1.0], hspace=0.35)

        # Force subplot
        axF = fig.add_subplot(gs[0, 0])
        (ln_c,) = axF.plot([], [], label="current_force [N]", linewidth=1.5)
        (ln_d,) = axF.plot([], [], linestyle="--", label="desired_force [N]", linewidth=1.5)
        axF.set_xlim(0.0, self.rolling_window)
        axF.set_ylabel("Force [N]", fontsize=10)
        axF.set_xlabel("Time [s]", fontsize=10)
        axF.grid(True, alpha=0.3)
        axF.legend(loc="best", fontsize=9)

        # Reward subplot
        axR = fig.add_subplot(gs[1, 0])
        (ln_r,) = axR.plot([], [], marker="o", linewidth=1.0, markersize=4)
        axR.set_xlabel("Episode", fontsize=10)
        axR.set_ylabel("Episode Reward", fontsize=10)
        axR.grid(True, alpha=0.3)

        # Pressure annotation (outside plot, top-left of figure)
        pressure_text = fig.text(
            0.02,
            0.97,
            "Pressure: --- MPa",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        interval_ms = int(1000 / self.hz)

        def on_timer(_frame):
            nonlocal tbuf, cbuf, dbuf, latest_pi
            while True:
                try:
                    msg = self.q.get_nowait()
                except Empty:
                    break

                tp = msg.get("type")
                if tp == "quit":
                    plt.close(fig)
                    return
                elif tp == "reset_force":
                    tbuf.clear()
                    cbuf.clear()
                    dbuf.clear()
                    ln_c.set_data([], [])
                    ln_d.set_data([], [])
                    axF.relim()
                    axF.autoscale_view()
                elif tp == "force":
                    t_new = float(msg["t"])
                    c_val = float(msg["cf"])
                    d_val = float(msg["df"])
                    tbuf.append(t_new)
                    cbuf.append(c_val)
                    dbuf.append(d_val)
                    # 디버그: 처음 몇 개만 출력
                    if len(tbuf) <= 3:
                        print(f"[Monitor] Force data #{len(tbuf)}: t={t_new:.3f}, current={c_val:.2f}N, desired={d_val:.2f}N")
                    while (
                        len(tbuf) > 0
                        and (t_new - tbuf[0]) > self.rolling_window
                    ):
                        tbuf.pop(0)
                        cbuf.pop(0)
                        dbuf.pop(0)
                elif tp == "reward":
                    ep_idx.append(int(msg["ep"]))
                    ep_rew.append(float(msg["rew"]))
                elif tp == "pi":
                    latest_pi = float(msg["pi"])

            if len(tbuf) >= 2:
                t = np.asarray(tbuf, dtype=float)
                c = np.asarray(cbuf, dtype=float)
                d = np.asarray(dbuf, dtype=float)
                ln_c.set_data(t, c)
                ln_d.set_data(t, d)
                axF.relim()
                axF.autoscale_view()
                t_max = t[-1]
                t_min = max(0.0, t_max - self.rolling_window)
                axF.set_xlim(t_min, t_max)

            if len(ep_idx) >= 1:
                ln_r.set_data(ep_idx, ep_rew)
                axR.relim()
                axR.autoscale_view()
                axR.set_xlim(0, max(10, ep_idx[-1] + 1))

            if latest_pi is not None:
                pressure_text.set_text(f"Pressure: {latest_pi:.3f} MPa")

        # 애니메이션 객체를 변수에 저장 (GC 방지)
        ani = FuncAnimation(fig, on_timer, interval=interval_ms, blit=False)
        try:
            plt.show()
        except Exception:
            pass
