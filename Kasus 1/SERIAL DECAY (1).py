import math
import time

import matplotlib.pyplot as plt
import numpy as np

DAY = 1 / 365.25
MINUTE = 1 / (365.25 * 24 * 60)
SECOND = 1 / (365.25 * 24 * 3600)
MICROSECOND = 1e-6 * SECOND

DECAY_CHAIN = [
    ("U-238", 4.468e9),
    ("Th-234", 24.10 * DAY),
    ("Pa-234m", 1.17 * MINUTE),
    ("U-234", 245500.0),
    ("Th-230", 75380.0),
    ("Ra-226", 1600.0),
    ("Rn-222", 3.8235 * DAY),
    ("Po-218", 3.10 * MINUTE),
    ("Pb-214", 26.8 * MINUTE),
    ("Bi-214", 19.9 * MINUTE),
    ("Po-214", 164.3 * MICROSECOND),
    ("Pb-210", 22.3),
    ("Bi-210", 5.012 * DAY),
    ("Po-210", 138.376 * DAY),
]

MEAN_LIFETIMES = np.array([half_life / math.log(2) for _, half_life in DECAY_CHAIN])


def simulate_serial(n_events, seed=12345):
    rng = np.random.default_rng(seed)
    total_times = np.empty(n_events)
    for event in range(n_events):
        waiting_times = rng.exponential(MEAN_LIFETIMES)
        total_times[event] = waiting_times.sum()
    return total_times


def main():
    n_events = 100000
    start = time.perf_counter()
    total_times = simulate_serial(n_events)
    runtime = time.perf_counter() - start

    print(f"Waktu komputasi serial N={n_events}: {runtime:.6f} detik")
    print(f"Rata-rata total waktu peluruhan: {total_times.mean():.6e} tahun")
    print(f"Median total waktu peluruhan: {np.median(total_times):.6e} tahun")

    plt.figure(figsize=(9, 5.2), dpi=150)
    plt.hist(total_times / 1e9, bins=35, color="#2d6a9f", edgecolor="white")
    plt.xlabel("Total waktu sampai Pb-206 stabil (miliar tahun)")
    plt.ylabel("Jumlah event")
    plt.title("Distribusi waktu peluruhan U-238 ke Pb-206, serial, N=2.000")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig("hist_serial_N2000.png")
    plt.show()


if __name__ == "__main__":
    main()