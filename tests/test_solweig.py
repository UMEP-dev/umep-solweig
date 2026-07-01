from pathlib import Path
import sys
import time
import threading
import pytest
import matplotlib.pyplot as plt

# Try importing tracking libraries
try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

try:
    import psutil
except ImportError:
    psutil = None

# Project root: adding /src allows Python to find 'umep_solweig' as a package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Regular imports (resolves the relative import issue automatically)
import umep_solweig.solweig_run as mod_cpu
import umep_solweig.solweig_run_gpu as mod_gpu

# Pointing to the test directory and configuration file
TEST_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TEST_DIR / "tests_data/configsolweig.ini"

# Artifact outputs kept in tests folder for automated checking
OUTPUT_CHART_GLOBAL = TEST_DIR / "tests_out/solweig_global_comparison.png"
OUTPUT_CHART_PROFILE = TEST_DIR / "tests_out/solweig_gpu_resource_profile.png"


# ==========================================
# CLASS FOR RESOURCE MONITORING
# ==========================================
class ResourceMonitor(threading.Thread):
    """Background thread to sample CPU, RAM, GPU, and VRAM."""

    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self._stop_event = threading.Event()
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_samples = []
        self.vram_samples = []
        self.timestamps = []
        self.start_time = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        if not psutil:
            return
        self.start_time = time.time()
        psutil.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            self.timestamps.append(time.time() - self.start_time)
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.ram_samples.append(psutil.virtual_memory().percent)

            if GPU_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_samples.append(util.gpu)

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_percent = (mem_info.used / mem_info.total) * 100
                    self.vram_samples.append(vram_percent)
                except Exception:
                    self.gpu_samples.append(0.0)
                    self.vram_samples.append(0.0)
            else:
                self.gpu_samples.append(0.0)
                self.vram_samples.append(0.0)

            time.sleep(self.interval)

    def get_metrics(self):
        def calc(samples):
            if not samples:
                return {"1%_low": 0.0, "avg": 0.0, "max": 0.0}
            sorted_s = sorted(samples)
            low_idx = max(0, int(len(sorted_s) * 0.01))
            return {
                "1%_low": sorted_s[low_idx],
                "avg": sum(samples) / len(samples),
                "max": max(samples),
            }

        return {
            "CPU": calc(self.cpu_samples),
            "RAM": calc(self.ram_samples),
            "GPU": calc(self.gpu_samples),
            "VRAM": calc(self.vram_samples),
        }


# ==========================================
# AUTOMATED TEST CASE
# ==========================================
def test_solweig_src_files_benchmark():
    # Ensure dependencies and inputs exist
    if psutil is None:
        pytest.fail("Missing required dependency: psutil")
    if not CONFIG_PATH.exists():
        pytest.fail(f"Missing configuration file! Expected at: {CONFIG_PATH}")

    # Configuration
    iterations = 1
    cpu_results = []
    gpu_results = []

    # 1. --- CPU Mode Run ---
    monitor_cpu = ResourceMonitor(interval=0.1)
    monitor_cpu.start()

    for _ in range(iterations):
        start_time = time.time()
        mod_cpu.solweig_run(str(CONFIG_PATH), None)
        cpu_results.append(time.time() - start_time)

    monitor_cpu.stop()
    monitor_cpu.join()

    # 2. --- GPU Mode Run ---
    monitor_gpu = ResourceMonitor(interval=0.1)
    monitor_gpu.start()

    for _ in range(iterations):
        start_time = time.time()
        mod_gpu.solweig_run(str(CONFIG_PATH), None)
        gpu_results.append(time.time() - start_time)

    monitor_gpu.stop()
    monitor_gpu.join()

    # 3. --- Calculations & Basic Verification Asserts ---
    assert len(cpu_results) == iterations
    assert len(gpu_results) == iterations

    avg_cpu_time = sum(cpu_results) / iterations
    avg_gpu_time = sum(gpu_results) / iterations
    speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0

    # 4. --- Generate Graph 1: Execution Time Comparison ---
    plt.figure(figsize=(8, 5))
    modes = ["CPU (Average)", "GPU (Average)"]
    values = [avg_cpu_time, avg_gpu_time]
    colors = ["#1f77b4", "#2ca02c"]

    bars = plt.bar(
        modes, values, color=colors, width=0.4, edgecolor="black", alpha=0.8
    )
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (max(values) * 0.02),
            f"{height:.3f} s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title(
        "Comparison of the global execution time : CPU vs GPU",
        fontsize=12,
        pad=15,
    )
    plt.ylabel("Mean execution time (seconds)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    if speedup > 0:
        plt.text(
            0.5,
            max(values) * 0.85,
            f"Le GPU est {speedup:.1f}x plus rapide",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5", fc="#fff9e6", ec="#ffa500", lw=1.5
            ),
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART_GLOBAL, dpi=300)
    plt.close()

    # 5. --- Generate Graph 2: Resource Profiles ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(
        "Graph 2 : Usage profile of resources in real time (GPU Mode)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plots_config = [
        (
            axs[0, 0],
            monitor_gpu.timestamps,
            monitor_gpu.cpu_samples,
            "CPU usage",
            "#1f77b4",
        ),
        (
            axs[0, 1],
            monitor_gpu.timestamps,
            monitor_gpu.ram_samples,
            "RAM usage",
            "#ff7f0e",
        ),
        (
            axs[1, 0],
            monitor_gpu.timestamps,
            monitor_gpu.gpu_samples,
            "GPU Core usage",
            "#2ca02c",
        ),
        (
            axs[1, 1],
            monitor_gpu.timestamps,
            monitor_gpu.vram_samples,
            "VRAM usage",
            "#d62728",
        ),
    ]

    for ax, x, y, title, color in plots_config:
        if x and y:
            ax.plot(x, y, color=color, linewidth=1.5, label=title)
            ax.fill_between(x, y, color=color, alpha=0.15)
            avg_val = sum(y) / len(y)
            ax.axhline(
                avg_val,
                color="black",
                linestyle="--",
                alpha=0.7,
                label=f"Moyenne ({avg_val:.1f}%)",
            )
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylabel("% Utilisé", fontsize=10)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)

    axs[1, 0].set_xlabel("Time spent (seconds)", fontsize=10)
    axs[1, 1].set_xlabel("Time spent (seconds)", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART_PROFILE, dpi=300)
    plt.close()

    # Final Assertions to guarantee charts were successfully written
    assert OUTPUT_CHART_GLOBAL.exists()
    assert OUTPUT_CHART_PROFILE.exists()
