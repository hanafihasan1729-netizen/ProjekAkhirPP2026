import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import time
import math

# --- 1. Parameter Simulasi ---
N_PARTICLES = 200
PARTICLES_PER_FRAME = 25
ANIMATION_INTERVAL_MS = 80
energies = [4.270, 4.198] # MeV
probs = [0.79, 0.21]
detector_resolution = 0.02 # MeV

# Geometri
Z_DETECTOR = 4.0 # cm
R_DETECTOR_RING = 2.0 # cm

# Parameter Waktu Fisis (Asumsi aktivitas sumber U-238 sederhana)
# Misal aktivitas sumber adalah 1000 Becquerel (1000 peluruhan/detik)
ACTIVITY_BQ = 1000.0 

# --- 2. Sampling Monte Carlo (Vectorized untuk N partikel) ---
start_total_time = time.perf_counter()
start_calc_time = time.perf_counter()

# Sampling Energi
rand_p = np.random.rand(N_PARTICLES)
e_true = np.where(rand_p < probs[0], energies[0], energies[1])
e_measured = np.random.normal(e_true, detector_resolution)

# Sampling Arah (Di dalam kerucut detektor visual)
theta_max = np.arctan(R_DETECTOR_RING / Z_DETECTOR)
cos_theta = 1.0 - np.random.rand(N_PARTICLES) * (1.0 - np.cos(theta_max))
theta = np.arccos(cos_theta)
phi = 2.0 * np.pi * np.random.rand(N_PARTICLES)

# Titik Tumbukan
r_hit = Z_DETECTOR * np.tan(theta)
x_hit = r_hit * np.cos(phi)
y_hit = r_hit * np.sin(phi)
z_hit = np.full(N_PARTICLES, Z_DETECTOR)

# Sampling Waktu Fisis (Distribusi Eksponensial untuk interval antar peluruhan)
# dt = -ln(U) / Aktivitas
dt = -np.log(np.random.rand(N_PARTICLES)) / ACTIVITY_BQ
physical_time_cumulative = np.cumsum(dt)

end_calc_time = time.perf_counter()
print(f"Waktu Komputasi CPU untuk {N_PARTICLES} partikel: {(end_calc_time - start_calc_time)*1000:.4f} milidetik")
print(f"Waktu Fisis Alamiah untuk {N_PARTICLES} partikel terdeteksi: {physical_time_cumulative[-1]:.4f} detik")

# Siapkan seluruh segmen lintasan sekali saja agar animasi hanya mengambil irisan data per batch.
trajectory_segments = np.stack(
    (
        np.zeros((N_PARTICLES, 3)),
        np.column_stack((x_hit, y_hit, z_hit))
    ),
    axis=1
)

# --- 3. Setup Visualisasi 3D ---
start_visualization_time = time.perf_counter()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_title(f'Akumulasi Deteksi {N_PARTICLES} Partikel Alfa U-238', fontsize=14, pad=20)
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
ax.set_zlabel('z (cm)')
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([0, 4.5])

# Gambar Cincin Detektor & Sumber
theta_ring = np.linspace(0, 2 * np.pi, 100)
ax.plot(R_DETECTOR_RING * np.cos(theta_ring), R_DETECTOR_RING * np.sin(theta_ring), np.full_like(theta_ring, Z_DETECTOR), color='#1f2937', linewidth=2, label='Bidang detektor')
ax.scatter([0], [0], [0], color='#8b5a62', s=60, label='Sumber U-238')

# Satu koleksi garis dan satu scatter cukup untuk memvisualisasikan banyak partikel per frame.
line_collection = Line3DCollection([], colors='gray', linewidths=0.8, alpha=0.3)
ax.add_collection3d(line_collection)
hit_scatter = ax.scatter([], [], [], color='#d97706', s=16, depthshade=False, label='Titik tumbukan')

info_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.9))

# --- 4. Fungsi Animasi ---
def init():
    line_collection.set_segments([])
    hit_scatter._offsets3d = ([], [], [])
    info_text.set_text("")
    return [line_collection, hit_scatter, info_text]

timing_state = {"reported": False}

def update(frame):
    # Setiap frame menambahkan beberapa partikel sekaligus.
    start_idx = frame * PARTICLES_PER_FRAME
    end_idx = min(start_idx + PARTICLES_PER_FRAME, N_PARTICLES)
    last_idx = end_idx - 1
    
    line_collection.set_segments(trajectory_segments[:end_idx])
    hit_scatter._offsets3d = (x_hit[:end_idx], y_hit[:end_idx], z_hit[:end_idx])
    
    # Update teks informasi
    info_str = (f"Deteksi Partikel: {end_idx} / {N_PARTICLES}\n"
                f"Batch visualisasi: {start_idx + 1} - {end_idx}\n"
                f"Energi Terakhir = {e_measured[last_idx]:.3f} MeV\n"
                f"Waktu Fisis Berjalan = {physical_time_cumulative[last_idx]:.4f} detik")
    info_text.set_text(info_str)

    if end_idx == N_PARTICLES and not timing_state["reported"]:
        timing_state["reported"] = True
        end_total_time = time.perf_counter()
        print(f"Waktu Visualisasi sampai frame terakhir: {(end_total_time - start_visualization_time):.4f} detik")
        print(f"Waktu Total Simulasi + Visualisasi: {(end_total_time - start_total_time):.4f} detik")
    
    return [line_collection, hit_scatter, info_text]

# --- 5. Eksekusi Animasi ---
total_frames = math.ceil(N_PARTICLES / PARTICLES_PER_FRAME)
ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=False, interval=ANIMATION_INTERVAL_MS, repeat=False)

plt.tight_layout()
plt.show()
