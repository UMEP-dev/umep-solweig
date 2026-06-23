from pathlib import Path
import sys
import types
import importlib
import time
import threading
import torch
import matplotlib.pyplot as plt

# Try importing pynvml for NVIDIA GPU tracking
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

try:
    import psutil
except ImportError:
    print("Error: 'psutil' library is required. Install it using 'pip install psutil'")
    sys.exit(1)

# --- Import du package installé ---
base = Path(__file__).resolve().parent
sys.path.insert(0, str(base / "src"))
import umep_solweig

# --- Chargement du module ---
mod_gpu = importlib.import_module("umep_solweig.solweig_run_gpu")
mod_cpu = importlib.import_module("umep_solweig.solweig_run")

# --- Chemins des fichiers de configuration ---
cpu_config_path = "/home/lemap/Documents/suede/umep_process_execute/UMEP-processing/processor/configsolweig.ini"
gpu_config_path = "/home/lemap/Documents/suede/umep_process_execute/configsolweig.ini"

cpu_results = []
gpu_results = []
iterations = 5

# ==========================================
# CLASSE DE MONITORING DES RESSOURCES
# ==========================================
class ResourceMonitor(threading.Thread):
    """Fils d'exécution d'arrière-plan pour échantillonner le CPU, la RAM, le GPU et le VRAM."""
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
        self.start_time = time.time()
        # Initialisation du calcul CPU non-bloquant
        psutil.cpu_percent(interval=None)
        
        while not self._stop_event.is_set():
            self.timestamps.append(time.time() - self.start_time)
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.ram_samples.append(psutil.virtual_memory().percent)
            
            if GPU_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    # Métrique 1 : Utilisation des cœurs GPU
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_samples.append(util.gpu)
                    
                    # Métrique 2 : Utilisation de la VRAM (Mémoire Vidéo)
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
        """Calcule le 1% low, la moyenne et le max pour chaque métrique."""
        def calc(samples):
            if not samples:
                return {"1%_low": 0.0, "avg": 0.0, "max": 0.0}
            sorted_s = sorted(samples)
            low_idx = max(0, int(len(sorted_s) * 0.01))
            return {
                "1%_low": sorted_s[low_idx],
                "avg": sum(samples) / len(samples),
                "max": max(samples)
            }
        
        return {
            "CPU": calc(self.cpu_samples),
            "RAM": calc(self.ram_samples),
            "GPU": calc(self.gpu_samples),
            "VRAM": calc(self.vram_samples)
        }

# ==========================================
# FONCTION POUR LA BARRE DE PROGRESSION
# ==========================================
def print_progress_bar(iteration, total, prefix='', length=30):
    """Affiche une barre de progression dynamique sur une seule ligne."""
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    remaining = total - iteration
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% ({iteration}/{total}) — Il reste {remaining} run(s)')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

# ==========================================
# 1. EXÉCUTION DES BENCHMARKS
# ==========================================
print(f"Lancement du benchmark global ({iterations} itérations par mode)...")
if not GPU_AVAILABLE:
    print("[Attention] pynvml non configuré ou GPU NVIDIA introuvable. Les métriques GPU/VRAM renverront 0%.")
print("-" * 60)

#--- Mode CPU (Commenté comme dans l'original) ---
monitor_cpu = ResourceMonitor(interval=0.1)
monitor_cpu.start()
print_progress_bar(0, iterations, prefix='Progression CPU')
for i in range(iterations):
    start_time = time.time()
    mod_cpu.solweig_run(cpu_config_path, None)
    cpu_results.append(time.time() - start_time)
    print_progress_bar(i + 1, iterations, prefix='Progression CPU')
monitor_cpu.stop()
monitor_cpu.join()
cpu_metrics = monitor_cpu.get_metrics()
print("-" * 60)

# Initialisation par défaut si CPU inactif
cpu_metrics = {
    "CPU": {"1%_low": 0, "avg": 0, "max": 0}, 
    "RAM": {"1%_low": 0, "avg": 0, "max": 0}, 
    "GPU": {"1%_low": 0, "avg": 0, "max": 0},
    "VRAM": {"1%_low": 0, "avg": 0, "max": 0}
}

# --- Mode GPU ---
monitor_gpu = ResourceMonitor(interval=0.1)
monitor_gpu.start()

# print_progress_bar(0, iterations, prefix='Progression GPU')
# for i in range(iterations):
#     start_time = time.time()
#     mod_gpu.solweig_run(gpu_config_path, None)
#     gpu_results.append(time.time() - start_time)
#     print_progress_bar(i + 1, iterations, prefix='Progression GPU')

monitor_gpu.stop()
monitor_gpu.join()
gpu_metrics = monitor_gpu.get_metrics()

print("-" * 60)

# ==========================================
# 2. CALCUL DES MÉTRIQUES GLOBALES
# ==========================================
avg_cpu_time = sum(cpu_results) / iterations if cpu_results else 0
avg_gpu_time = sum(gpu_results) / iterations if gpu_results else 0
speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0

print("\n" + "="*55)
print("               RÉSULTATS GLOBAUX          ")
print("="*55)
print(f"Temps moyen CPU : {avg_cpu_time:.3f} secondes")
print(f"Temps moyen GPU : {avg_gpu_time:.3f} secondes")
print(f"Facteur d'accélération (Speedup) : x{speedup:.2f}")
print("-" * 55)

def print_hardware_table(mode_name, metrics):
    print(f" Métriques Matériel ({mode_name}) :")
    print(f" {'Ressource':<12} | {'1% Low':<10} | {'Moyenne':<10} | {'Maximum':<10}")
    print(f" {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for key, val in metrics.items():
        print(f" {key:<12} | {val['1%_low']:>8.1f}% | {val['avg']:>8.1f}% | {val['max']:>8.1f}%")
    print("-" * 55)

if cpu_results:
    print_hardware_table("Mode CPU", cpu_metrics)
print_hardware_table("Mode GPU", gpu_metrics)
print("="*55)

# ==========================================
# 3. GÉNÉRATION DES GRAPHIQUES
# ==========================================

# Graphique 1 : Comparaison du temps d'exécution global
plt.figure(figsize=(8, 5))
modes = ['CPU (Moyenne)', 'GPU (Moyenne)']
values = [avg_cpu_time, avg_gpu_time]
colors = ['#1f77b4', '#2ca02c']

bars = plt.bar(modes, values, color=colors, width=0.4, edgecolor='black', alpha=0.8)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (max(values) * 0.02),
             f'{height:.3f} s', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Comparaison globale des temps d'exécution : CPU vs GPU", fontsize=12, pad=15)
plt.ylabel("Temps d'exécution moyen (secondes)", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

if speedup > 0:
    plt.text(0.5, max(values) * 0.85, f"Le GPU est {speedup:.1f}x plus rapide", 
             ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff9e6", ec="#ffa500", lw=1.5))

plt.tight_layout()
output_chart1 = base / "solweig_global_comparison.png"
plt.savefig(output_chart1, dpi=300)
print(f"\n[Info] Graphique temporel sauvegardé sous : {output_chart1}")


# Graphique 2 : Profil d'utilisation des ressources au cours du temps (Mode GPU)
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig.suptitle("Profil d'utilisation des ressources en temps réel (Mode GPU)", fontsize=14, fontweight='bold', y=0.98)

# Liste des configurations pour chaque sous-graphique
plots_config = [
    (axs[0, 0], monitor_gpu.timestamps, monitor_gpu.cpu_samples, "Utilisation CPU", "#1f77b4"),
    (axs[0, 1], monitor_gpu.timestamps, monitor_gpu.ram_samples, "Utilisation RAM", "#ff7f0e"),
    (axs[1, 0], monitor_gpu.timestamps, monitor_gpu.gpu_samples, "Utilisation Cœur GPU", "#2ca02c"),
    (axs[1, 1], monitor_gpu.timestamps, monitor_gpu.vram_samples, "Allocation VRAM (Mémoire)", "#d62728")
]

for ax, x, y, title, color in plots_config:
    if x and y:
        ax.plot(x, y, color=color, linewidth=1.5, label=title)
        ax.fill_between(x, y, color=color, alpha=0.15)
        # Ligne de moyenne
        avg_val = sum(y) / len(y)
        ax.axhline(avg_val, color='black', linestyle='--', alpha=0.7, label=f"Moyenne ({avg_val:.1f}%)")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel("% Utilisé", fontsize=10)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

# Configuration de l'axe des abscisses pour la rangée du bas
axs[1, 0].set_xlabel("Temps écoulé (secondes)", fontsize=10)
axs[1, 1].set_xlabel("Temps écoulé (secondes)", fontsize=10)

plt.tight_layout()
output_chart2 = base / "solweig_gpu_resource_profile.png"
plt.savefig(output_chart2, dpi=300)
print(f"[Info] Graphique des profils matériels sauvegardé sous : {output_chart2}")

# Affichage final de tous les graphiques ouverts
plt.show()