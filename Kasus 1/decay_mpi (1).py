"""
Simulasi peluruhan rantai U-238 → Pb-206 menggunakan MPI (mpi4py)

=============================================================
CARA MENJALANKAN:
  mpiexec -n <JUMLAH_PROSES> python decay_mpi.py

Contoh:
  mpiexec -n 4 python decay_mpi.py   # 4 proses
  mpiexec -n 8 python decay_mpi.py   # 8 proses (untuk uji skalabilitas)

=============================================================
PARAMETER YANG BISA DIUBAH:
  - N_EVENTS  : jumlah event Monte Carlo (baris ~36)
  - Jumlah proses diatur lewat argumen -n pada mpiexec (bukan di kode)

CATATAN SKALABILITAS:
  Untuk mengukur speedup, jalankan ulang dengan -n 1, 2, 4, 8, 16, dst.
  lalu bandingkan "Waktu komputasi" yang tercetak oleh rank 0.
=============================================================
"""

import math
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

# ─────────────────────────────────────────────
# PARAMETER UTAMA  ← UBAH DI SINI
# ─────────────────────────────────────────────
N_EVENTS = 100_000   # <── jumlah event total; coba 10_000 / 100_000 / 1_000_000
MASTER_SEED = 12345  # seed global untuk reprodusibilitas
# ─────────────────────────────────────────────

# Unit konversi ke satuan TAHUN
DAY        = 1 / 365.25
MINUTE     = 1 / (365.25 * 24 * 60)
SECOND     = 1 / (365.25 * 24 * 3600)
MICROSECOND = 1e-6 * SECOND

# Rantai peluruhan U-238 → Pb-206 (nama isotop, waktu paruh dalam tahun)
DECAY_CHAIN = [
    ("U-238",  4.468e9),
    ("Th-234", 24.10  * DAY),
    ("Pa-234m", 1.17  * MINUTE),
    ("U-234",  245_500.0),
    ("Th-230",  75_380.0),
    ("Ra-226",   1_600.0),
    ("Rn-222",   3.8235 * DAY),
    ("Po-218",   3.10  * MINUTE),
    ("Pb-214",  26.8   * MINUTE),
    ("Bi-214",  19.9   * MINUTE),
    ("Po-214", 164.3   * MICROSECOND),
    ("Pb-210",  22.3),
    ("Bi-210",   5.012 * DAY),
    ("Po-210", 138.376 * DAY),
    # Pb-206 stabil → rantai berhenti di sini
]

# Mean lifetime τ = t½ / ln2
MEAN_LIFETIMES = np.array([t / math.log(2) for _, t in DECAY_CHAIN])


def simulate_local(n_local: int, seed: int) -> np.ndarray:
    """
    Setiap proses MPI memanggil fungsi ini untuk mensimulasikan
    n_local event. Setiap event = satu atom U-238 yang meluruh
    melalui seluruh rantai hingga mencapai Pb-206 stabil.

    Waktu tunggu pada setiap tahap rantai berdistribusi eksponensial
    dengan mean = τ_i (mean lifetime isotop ke-i).
    Total waktu peluruhan satu event = jumlah semua waktu tunggu.
    """
    rng = np.random.default_rng(seed)
    # shape: (n_local, jumlah_isotop_dalam_rantai)
    waiting_times = rng.exponential(MEAN_LIFETIMES, size=(n_local, len(MEAN_LIFETIMES)))
    return waiting_times.sum(axis=1)   # total waktu tiap event


def main():
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()   # ID proses ini  (0 = master/root)
    size  = comm.Get_size()   # jumlah proses total  ← diatur via -n

    # ── Distribusi beban ─────────────────────────────────────────────
    # Rank 0 menghitung berapa event yang dikerjakan masing-masing rank.
    base   = N_EVENTS // size
    extra  = N_EVENTS  % size
    # Rank 0..extra-1 masing-masing mengerjakan satu event lebih banyak
    n_local = base + (1 if rank < extra else 0)

    # ── Seed independen per proses (SeedSequence menjamin bebas korelasi) ──
    seed_seq    = np.random.SeedSequence(MASTER_SEED)
    child_seeds = seed_seq.spawn(size)
    local_seed  = int(child_seeds[rank].generate_state(1)[0])

    # ── Timer mulai (barrier memastikan semua proses mulai bersamaan) ──
    comm.Barrier()
    t_start = MPI.Wtime()   # waktu wall-clock MPI (lebih presisi dari time.time)

    # ── Komputasi lokal ──────────────────────────────────────────────
    local_results = simulate_local(n_local, local_seed)

    # ── Kumpulkan hasil di rank 0 (Gather) ───────────────────────────
    all_results = comm.gather(local_results, root=0)

    comm.Barrier()
    t_end = MPI.Wtime()

    # ── Hanya rank 0 yang mencetak & memplot ─────────────────────────
    if rank == 0:
        total_times = np.concatenate(all_results)   # gabung semua event

        elapsed = t_end - t_start
        print("=" * 60)
        print(f"  Jumlah proses MPI      : {size}")
        print(f"  Jumlah event (N)       : {N_EVENTS:,}".replace(",", "."))
        print(f"  Waktu komputasi        : {elapsed:.6f} detik")
        print(f"  Rata-rata t peluruhan  : {total_times.mean():.6e} tahun")
        print(f"  Median  t peluruhan    : {np.median(total_times):.6e} tahun")
        print(f"  Std dev t peluruhan    : {total_times.std():.6e} tahun")
        print("=" * 60)

        # Plot distribusi
        matplotlib.use("Agg")   # non-interactive; ganti "TkAgg" jika ingin pop-up
        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)

        ax.hist(
            total_times / 1e9,
            bins=50,
            color="#2d6a9f",
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_xlabel("Total waktu peluruhan U-238 → Pb-206  (miliar tahun)", fontsize=11)
        ax.set_ylabel("Jumlah event", fontsize=11)
        ax.set_title(
            f"Distribusi waktu peluruhan  |  N = {N_EVENTS:,}  |  {size} proses MPI".replace(",", "."),
            fontsize=12,
        )
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        fname = f"hist_mpi_N{N_EVENTS}_p{size}.png"
        fig.savefig(fname)
        print(f"  Histogram tersimpan    : {fname}")


if __name__ == "__main__":
    main()
