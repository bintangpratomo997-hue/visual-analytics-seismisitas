"""
DBSCAN Clustering: Seismisitas Indonesia
=========================================

Dependensi:
    pip install pandas geopandas scikit-learn psycopg2-binary matplotlib kneed
"""

# ── Library standar untuk manipulasi data tabular
import pandas as pd
# ── Library untuk data geospasial (proyeksi koordinat)
import geopandas as gpd
# ── Library komputasi numerik
import numpy as np

# ── Matplotlib diset ke backend non-interaktif "Agg" agar grafik bisa
#    disimpan sebagai file PNG tanpa perlu membuka jendela GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Driver koneksi Python ke PostgreSQL
import psycopg2
from psycopg2.extras import execute_values   # insert batch sekaligus (efisien)

# ── Implementasi DBSCAN dari scikit-learn
from sklearn.cluster import DBSCAN
# ── Digunakan untuk menghitung k-distance plot & penanganan noise
from sklearn.neighbors import NearestNeighbors
# ── Dua metrik evaluasi kualitas clustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

import logging
import warnings
warnings.filterwarnings("ignore")   # sembunyikan peringatan yang tidak kritis

# ── kneed adalah library opsional untuk mendeteksi titik "knee" secara otomatis
#    pada k-distance plot; jika tidak terpasang, deteksi knee dilewati
try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False


# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

# Parameter koneksi ke database PostgreSQL lokal
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 5432,
    "dbname"  : "seismisitas_db",
    "user"    : "postgres",
    "password": "black007",   # ← ganti dengan password Anda
}

# ── Kandidat nilai ε (epsilon) dalam satuan meter, setelah proyeksi UTM.
#    Rentang 150–250 km dipilih karena mencerminkan jarak antar zona seismik
#    utama di Indonesia (mis. zona subduksi Sunda, Banda, Sulawesi).
#    Nilai optimal dipilih berdasarkan grid search + metrik evaluasi.
EPS_CANDIDATES = [
    150_000,   # 150 km
    175_000,   # 175 km
    200_000,   # 200 km
    225_000,   # 225 km
    250_000,   # 250 km
]

# ── Kandidat min_samples: jumlah minimum titik dalam radius ε agar suatu
#    titik disebut "core point". Nilai tinggi (500–3000) dipilih karena
#    dataset berskala nasional (puluhan ribu titik), sehingga hanya zona
#    yang benar-benar padat secara seismologi yang terbentuk sebagai kluster.
#    Referensi: ~1–7% dari total dataset untuk data geografis skala besar.
MIN_SAMPLES_CANDIDATES = [500, 1000, 2000, 3000]

# ── Syarat hasil clustering yang diterima oleh grid search:
#    - Minimal 3 kluster agar analisis bermakna (tidak hanya 1–2 zona besar)
#    - Maksimal 20 kluster agar tidak terlalu granular/terfragmentasi
#    - Noise di bawah 40% agar mayoritas data terklasifikasi
MIN_KLUSTER   = 3     # minimal kluster yang bermakna
MAX_KLUSTER   = 20    # maksimal kluster (hindari terlalu granular)
MAX_NOISE_PCT = 40.0  # maksimal persentase noise yang diperbolehkan

# ── k-distance plot menggunakan k=5 tetangga terdekat (standar umum DBSCAN)
K_NEIGHBORS  = 5
OUTPUT_KDIST = "dbscan_kdistance.png"
OUTPUT_EVAL  = "dbscan_evaluation.png"

# ── Konfigurasi format log: tampilkan waktu, level, dan pesan
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. EXTRACT
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Tahap EXTRACT pada pipeline ETL.
    Mengambil id_fakta, latitude, dan longitude dari database PostgreSQL
    melalui join antara tabel FACT_GEMPA dan DIM_LOKASI.
    Hanya kolom koordinat yang diambil karena DBSCAN hanya membutuhkan
    fitur spasial (lat, lon) untuk proses clustering.
    """
    log.info("[EXTRACT] Mengambil data dari PostgreSQL ...")
    conn  = psycopg2.connect(**DB_CONFIG)

    # Query mengambil koordinat tiap kejadian gempa beserta id_fakta
    # sebagai primary key yang nanti digunakan saat update FACT_GEMPA
    query = """
        SELECT
            f.id_fakta,
            l.lat,
            l.lon
        FROM FACT_GEMPA f
        JOIN DIM_LOKASI l ON f.id_lokasi = l.id_lokasi
        ORDER BY f.id_fakta
    """
    df = pd.read_sql(query, conn)
    conn.close()
    log.info(f"  → {len(df):,} baris berhasil diambil")
    return df


# ─────────────────────────────────────────────
# 2. TRANSFORM — Proyeksi ke EPSG:32750
# ─────────────────────────────────────────────

def project_to_utm(df: pd.DataFrame) -> np.ndarray:
    """
    Tahap TRANSFORM: konversi koordinat dari sistem geografis WGS84
    (EPSG:4326, satuan derajat) ke sistem proyeksi UTM Zone 50S
    (EPSG:32750, satuan meter).

    Alasan proyeksi diperlukan:
        DBSCAN menggunakan jarak Euclidean. Jika koordinat dalam derajat,
        1 derajat lintang ≠ 1 derajat bujur dalam jarak nyata (km).
        Proyeksi UTM mengubah koordinat ke satuan meter sehingga jarak
        antar titik dapat dihitung secara akurat. (Sesuai Bab 3.4.1)

    UTM Zone 50S dipilih karena mencakup wilayah Indonesia bagian tengah–barat.
    """
    log.info("[TRANSFORM] Proyeksi EPSG:4326 → EPSG:32750 (UTM Zone 50S) ...")

    # Buat GeoDataFrame dengan geometri titik dari kolom lon/lat
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"      # sistem koordinat asal: WGS84 (derajat)
    )

    # Proyeksikan ke UTM Zone 50S (satuan meter)
    gdf_utm = gdf.to_crs("EPSG:32750")

    # Ekstrak koordinat x (easting) dan y (northing) dalam meter
    # sebagai array NumPy 2D: [[x1,y1], [x2,y2], ...]
    X = np.column_stack([
        gdf_utm.geometry.x,   # koordinat timur (meter)
        gdf_utm.geometry.y    # koordinat utara (meter)
    ])
    log.info(f"  → Shape fitur: {X.shape} (x_utm_m, y_utm_m)")
    return X


# ─────────────────────────────────────────────
# 3a. K-DISTANCE PLOT
# ─────────────────────────────────────────────

def plot_kdistance(X: np.ndarray):
    """
    Hitung dan simpan k-distance plot sebagai panduan visual pemilihan ε.

    Cara kerja k-distance plot:
        Untuk setiap titik, hitung jarak ke tetangga ke-k terdekat,
        lalu urutkan dari terkecil ke terbesar. "Titik knee" (siku)
        pada kurva menunjukkan ε optimal: di bawah knee = kluster padat,
        di atas knee = noise / outlier.

    Sesuai Bab 3.4.1 skripsi: "k-distance plot... mengidentifikasi titik knee".

    Catatan: sampling 10.000 titik acak digunakan agar proses cepat
    namun tetap representatif untuk dataset besar.
    """
    log.info(f"[EVAL] Menghitung k-distance plot (k={K_NEIGHBORS}) ...")

    # Ambil sampel acak maksimal 10.000 titik (cukup representatif)
    np.random.seed(42)   # seed tetap agar hasil reproducible
    idx_sample = np.random.choice(len(X), min(10000, len(X)), replace=False)
    X_sample   = X[idx_sample]

    # Hitung jarak ke-k tetangga terdekat menggunakan Ball Tree
    # (efisien untuk data berdimensi rendah seperti koordinat 2D)
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="ball_tree")
    nbrs.fit(X_sample)
    distances, _ = nbrs.kneighbors(X_sample)

    # Ambil jarak ke tetangga paling jauh (kolom terakhir), lalu urutkan naik
    kdist = np.sort(distances[:, -1])

    # Deteksi titik knee secara otomatis menggunakan library kneed (jika tersedia)
    knee_val = None
    if KNEED_AVAILABLE:
        kl = KneeLocator(
            range(len(kdist)), kdist,
            curve="convex", direction="increasing"
        )
        if kl.knee is not None:
            knee_val = kdist[kl.knee]
            log.info(f"  → Knee terdeteksi: ε ≈ {knee_val/1000:.1f} km")

    # ── Buat visualisasi k-distance plot ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(kdist / 1000, color="steelblue", linewidth=1.5)   # konversi ke km

    # Tandai posisi knee jika berhasil terdeteksi
    if knee_val:
        ax.axhline(knee_val / 1000, color="red", linestyle="--",
                   label=f"Knee ≈ {knee_val/1000:.1f} km")

    # Tampilkan semua kandidat ε sebagai garis referensi oranye
    for eps in EPS_CANDIDATES:
        ax.axhline(eps / 1000, color="orange", linestyle=":",
                   alpha=0.7, linewidth=1.2)
    ax.axhline(EPS_CANDIDATES[0] / 1000, color="orange", linestyle=":",
               alpha=0.7, linewidth=1.2, label="Kandidat ε")

    ax.set_title(
        f"k-Distance Plot (k={K_NEIGHBORS}) — Penentuan ε DBSCAN\n"
        "Sri Bintang Pratomo (825220070) — UNTAR Prodi SI",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Titik Data (diurutkan)")
    ax.set_ylabel("Jarak ke Tetangga ke-k (km)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_KDIST, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  → Disimpan: {OUTPUT_KDIST}")
    return knee_val


# ─────────────────────────────────────────────
# 3b. GRID SEARCH
# ─────────────────────────────────────────────

def grid_search_dbscan(X: np.ndarray) -> list:
    """
    Cari kombinasi parameter DBSCAN terbaik dengan metode Grid Search.

    Grid Search menguji semua kombinasi ε × min_samples yang didefinisikan
    di EPS_CANDIDATES dan MIN_SAMPLES_CANDIDATES. Setiap kombinasi dievaluasi
    menggunakan dua metrik:
        - Silhouette Score : semakin tinggi semakin baik (maks 1.0)
        - Davies-Bouldin Index (DBI) : semakin rendah semakin baik (min 0.0)

    Syarat valid sebuah hasil (sesuai parameter di bagian KONFIGURASI):
        1. Jumlah kluster antara MIN_KLUSTER dan MAX_KLUSTER
        2. Persentase noise di bawah MAX_NOISE_PCT

    Metrik dihitung pada subset evaluasi (maks 15.000 titik) agar
    komputasi silhouette_score tidak terlalu lambat pada dataset besar.
    """
    log.info("[MODELING] Grid search parameter DBSCAN ...")
    log.info(f"  ε kandidat (km)  : {[e//1000 for e in EPS_CANDIDATES]}")
    log.info(f"  min_samples      : {MIN_SAMPLES_CANDIDATES}")
    log.info(f"  Syarat valid     : {MIN_KLUSTER}–{MAX_KLUSTER} kluster, "
             f"noise < {MAX_NOISE_PCT}%")

    # Subset acak untuk evaluasi metrik (lebih cepat daripada seluruh dataset)
    np.random.seed(42)
    idx_eval = np.random.choice(len(X), min(15000, len(X)), replace=False)
    X_eval   = X[idx_eval]

    hasil = []   # daftar semua kombinasi yang memenuhi syarat

    for eps in EPS_CANDIDATES:
        for min_samp in MIN_SAMPLES_CANDIDATES:
            # Jalankan DBSCAN pada seluruh dataset dengan kombinasi parameter ini
            # algorithm="ball_tree": efisien untuk data spasial 2D
            # n_jobs=-1: manfaatkan semua core CPU
            db     = DBSCAN(
                eps=eps, min_samples=min_samp,
                algorithm="ball_tree", n_jobs=-1
            )
            labels    = db.fit_predict(X)

            # Hitung statistik dasar hasil clustering (label -1 dikecualikan karena itu noise)
            n_kluster = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise   = int(np.sum(labels == -1))
            pct_noise = n_noise / len(labels) * 100

            # Periksa apakah hasil memenuhi syarat yang ditetapkan
            valid = (MIN_KLUSTER <= n_kluster <= MAX_KLUSTER
                     and pct_noise < MAX_NOISE_PCT)

            if not valid:
                log.info(
                    f"  ε={eps//1000:3d}km | min={min_samp:4d} | "
                    f"kluster={n_kluster:3d} | noise={pct_noise:5.1f}% → SKIP"
                )
                continue

            # Ambil label untuk subset evaluasi, cek ada minimal 2 kluster
            labels_eval = labels[idx_eval]
            if len(set(labels_eval)) < 2:
                # silhouette_score membutuhkan minimal 2 kluster
                continue

            # Hitung metrik evaluasi pada subset (bukan seluruh data)
            sil = silhouette_score(X_eval, labels_eval)
            dbi = davies_bouldin_score(X_eval, labels_eval)

            # Simpan semua informasi hasil ke dalam daftar
            hasil.append({
                "eps"        : eps,
                "min_samples": min_samp,
                "n_kluster"  : n_kluster,
                "n_noise"    : n_noise,
                "pct_noise"  : round(pct_noise, 2),
                "silhouette" : round(sil, 4),
                "dbi"        : round(dbi, 4),
                "labels"     : labels,    # label array lengkap (dibutuhkan tahap berikutnya)
            })

            log.info(
                f"  ε={eps//1000:3d}km | min={min_samp:4d} | "
                f"kluster={n_kluster:3d} | noise={pct_noise:5.1f}% | "
                f"Silhouette={sil:.4f} | DBI={dbi:.4f} ✓"
            )

    # Jika tidak ada kombinasi yang memenuhi syarat, jalankan fallback
    if not hasil:
        log.warning(
            "[MODELING] Tidak ada kombinasi yang memenuhi syarat! "
            "Melonggarkan syarat dan mencoba ulang ..."
        )
        return grid_search_fallback(X, X_eval)

    return hasil


def grid_search_fallback(X: np.ndarray, X_eval: np.ndarray) -> list:
    """
    Fallback: dijalankan jika grid search utama tidak menghasilkan kombinasi
    yang memenuhi syarat MIN_KLUSTER / MAX_KLUSTER / MAX_NOISE_PCT.

    Perbedaan dari grid search utama:
        Tidak ada batasan jumlah kluster — syarat minimal hanya n_kluster >= 2
        agar metrik evaluasi tetap dapat dihitung.
    Tujuan: pastikan program selalu menghasilkan sebuah model terbaik
    meski parameter data tidak ideal.
    """
    log.info("[MODELING] Fallback: tanpa batasan jumlah kluster ...")
    hasil = []

    for eps in EPS_CANDIDATES:
        for min_samp in MIN_SAMPLES_CANDIDATES:
            db     = DBSCAN(
                eps=eps, min_samples=min_samp,
                algorithm="ball_tree", n_jobs=-1
            )
            labels    = db.fit_predict(X)
            n_kluster = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise   = int(np.sum(labels == -1))
            pct_noise = n_noise / len(labels) * 100

            # Lewati jika hanya terbentuk 0 atau 1 kluster
            if n_kluster < 2:
                continue

            # Ambil ulang subset evaluasi secara acak
            labels_eval = labels[X_eval] if len(X_eval) < len(X) else labels
            labels_eval = labels[np.random.choice(
                len(X), min(15000, len(X)), replace=False
            )]
            if len(set(labels_eval)) < 2:
                continue

            sil = silhouette_score(X_eval, labels_eval)
            dbi = davies_bouldin_score(X_eval, labels_eval)

            hasil.append({
                "eps"        : eps,
                "min_samples": min_samp,
                "n_kluster"  : n_kluster,
                "n_noise"    : n_noise,
                "pct_noise"  : round(pct_noise, 2),
                "silhouette" : round(sil, 4),
                "dbi"        : round(dbi, 4),
                "labels"     : labels,
            })
            log.info(
                f"  ε={eps//1000:3d}km | min={min_samp:4d} | "
                f"kluster={n_kluster:3d} | Silhouette={sil:.4f}"
            )

    if not hasil:
        raise ValueError(
            "Tidak ada hasil valid sama sekali. "
            "Periksa koneksi database dan data input."
        )
    return hasil


def pilih_parameter_optimal(hasil: list) -> dict:
    """
    Dari semua kombinasi valid, pilih yang memiliki Silhouette Score tertinggi.

    Silhouette Score mengukur seberapa mirip suatu titik dengan klusternya
    sendiri dibanding kluster lain (rentang -1 hingga 1; semakin tinggi
    semakin kohesif dan terpisah antar kluster). Dipilih sebagai kriteria
    utama karena lebih intuitif dan umum digunakan dalam literatur clustering.
    """
    terbaik = max(hasil, key=lambda x: x["silhouette"])
    log.info(
        f"[MODELING] Parameter optimal: "
        f"ε={terbaik['eps']//1000}km | "
        f"min_samples={terbaik['min_samples']} | "
        f"kluster={terbaik['n_kluster']} | "
        f"noise={terbaik['pct_noise']}% | "
        f"Silhouette={terbaik['silhouette']} | "
        f"DBI={terbaik['dbi']}"
    )
    return terbaik


# ─────────────────────────────────────────────
# 3c. PLOT EVALUASI
# ─────────────────────────────────────────────

def plot_evaluasi(hasil: list, terbaik: dict):
    """
    Buat grafik evaluasi parameter DBSCAN (disimpan sebagai PNG).

    Grafik terdiri dari dua panel:
        Kiri  — Silhouette Score vs ε untuk setiap nilai min_samples
                 (lebih tinggi = kluster lebih kohesif)
        Kanan — Davies-Bouldin Index vs ε untuk setiap nilai min_samples
                 (lebih rendah = kluster lebih terpisah)
    Garis merah vertikal menandai posisi ε optimal yang dipilih.
    Grafik ini digunakan sebagai bahan analisis pada Bab 4 skripsi.
    """
    log.info(f"[PLOT] Menyimpan grafik evaluasi ke '{OUTPUT_EVAL}' ...")

    # Buat DataFrame dari hasil grid search (tanpa kolom 'labels' yang besar)
    df_h = pd.DataFrame([
        {k: v for k, v in h.items() if k != "labels"}
        for h in hasil
    ])
    if df_h.empty:
        log.warning("  → Tidak ada data valid untuk diplot")
        return

    df_h["eps_km"] = df_h["eps"] / 1000   # konversi ke km untuk sumbu x
    colors = ["steelblue", "darkorange", "green", "purple"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Evaluasi Parameter DBSCAN — Seismisitas Indonesia\n"
        "Sri Bintang Pratomo (825220070) — UNTAR Prodi SI",
        fontsize=11, fontweight="bold"
    )

    # Plot satu kurva per nilai min_samples pada kedua panel
    for i, ms in enumerate(MIN_SAMPLES_CANDIDATES):
        sub = df_h[df_h["min_samples"] == ms].sort_values("eps_km")
        if sub.empty:
            continue
        c = colors[i % len(colors)]
        axes[0].plot(sub["eps_km"], sub["silhouette"], "o-",
                     color=c, label=f"min_samples={ms}", linewidth=2,
                     markersize=7)
        axes[1].plot(sub["eps_km"], sub["dbi"], "s-",
                     color=c, label=f"min_samples={ms}", linewidth=2,
                     markersize=7)

    # Tandai posisi ε optimal dengan garis vertikal merah
    opt_eps = terbaik["eps"] / 1000
    for ax in axes:
        ax.axvline(opt_eps, color="red", linestyle="--",
                   label=f"Optimal ε={opt_eps:.0f}km")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Silhouette Score (lebih tinggi = lebih baik)")
    axes[0].set_xlabel("ε / Epsilon (km)")
    axes[0].set_ylabel("Silhouette Score")

    axes[1].set_title("Davies-Bouldin Index (lebih rendah = lebih baik)")
    axes[1].set_xlabel("ε / Epsilon (km)")
    axes[1].set_ylabel("Davies-Bouldin Index")

    plt.tight_layout()
    plt.savefig(OUTPUT_EVAL, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  → Disimpan: {OUTPUT_EVAL}")


# ─────────────────────────────────────────────
# 3d. TANGANI NOISE
# ─────────────────────────────────────────────

def tangani_noise(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Tangani titik noise (label = -1) dengan strategi Nearest Core Point.

    Strategi:
        Setiap titik noise ditugaskan ulang ke kluster milik core point
        terdekatnya. Ini memastikan semua titik gempa memiliki label kluster
        yang valid sehingga tabel FACT_GEMPA tidak memiliki id_kluster NULL.

    Mengapa tidak dibiarkan sebagai noise?
        Dalam konteks seismologi, sebuah gempa tetap terjadi di suatu wilayah
        meskipun kepadatannya rendah. Menetapkan zona terdekat memberikan
        makna geografis pada setiap kejadian. (Sesuai Bab 3.4.1 skripsi)
    """
    labels_clean = labels.copy()   # salin agar array asli tidak dimodifikasi
    idx_noise    = np.where(labels == -1)[0]    # indeks titik-titik noise
    idx_core     = np.where(labels != -1)[0]    # indeks titik-titik non-noise (core/border)

    if len(idx_noise) == 0:
        log.info("[NOISE] Tidak ada titik noise")
        return labels_clean

    log.info(f"[NOISE] Menangani {len(idx_noise):,} noise → nearest core point ...")

    # Bangun model Nearest Neighbor hanya dari titik-titik non-noise
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nbrs.fit(X[idx_core])

    # Untuk setiap titik noise, cari core point terdekat
    _, idx_nearest = nbrs.kneighbors(X[idx_noise])

    # Salin label dari core point terdekat ke titik noise
    for i, noise_i in enumerate(idx_noise):
        labels_clean[noise_i] = labels[idx_core[idx_nearest[i, 0]]]

    log.info(f"  → {len(idx_noise):,} titik noise berhasil diberi label")
    return labels_clean


# ─────────────────────────────────────────────
# 4. LOAD
# ─────────────────────────────────────────────

def bangun_dim_kluster(
    df: pd.DataFrame,
    labels_clean: np.ndarray,
    terbaik: dict
) -> pd.DataFrame:
    """
    Bangun tabel DIM_KLUSTER (dimensi kluster untuk data warehouse).

    Untuk setiap kluster, dihitung:
        - n_anggota    : jumlah gempa dalam kluster
        - centroid_lat/lon : titik pusat kluster (rata-rata koordinat)
        - rata_mag     : rata-rata magnitudo gempa dalam kluster
        - rata_depth   : rata-rata kedalaman gempa dalam kluster
        - deskripsi    : string metadata parameter DBSCAN yang digunakan

    Kolom deskripsi memudahkan traceability: pengguna dashboard dapat
    mengetahui parameter apa yang menghasilkan kluster tersebut.
    """
    log.info("[LOAD] Membangun DIM_KLUSTER ...")

    # Ambil data lengkap (magnitude + kedalaman) untuk menghitung statistik kluster
    conn  = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT
            f.id_fakta,
            l.lat,
            l.lon,
            f.nilai_magnitude,
            f.nilai_kedalaman
        FROM FACT_GEMPA f
        JOIN DIM_LOKASI l ON f.id_lokasi = l.id_lokasi
        ORDER BY f.id_fakta
    """
    df_full = pd.read_sql(query, conn)
    conn.close()

    # Gabungkan label hasil clustering ke DataFrame
    df_full["label_kluster"] = labels_clean

    rows = []
    for label in sorted(df_full["label_kluster"].unique()):
        subset       = df_full[df_full["label_kluster"] == label]
        n_anggota    = len(subset)
        centroid_lat = round(float(subset["lat"].mean()), 6)
        centroid_lon = round(float(subset["lon"].mean()), 6)
        rata_mag     = round(float(subset["nilai_magnitude"].mean()), 2)
        rata_depth   = round(float(subset["nilai_kedalaman"].mean()), 2)

        # Deskripsi otomatis yang menyertakan parameter DBSCAN dan metrik evaluasi
        deskripsi = (
            f"Kluster {label} — DBSCAN "
            f"(eps={terbaik['eps']//1000}km, "
            f"min_samples={terbaik['min_samples']}). "
            f"N={n_anggota:,} gempa. "
            f"Silhouette={terbaik['silhouette']}, "
            f"DBI={terbaik['dbi']}."
        )

        rows.append({
            "label_kluster": int(label),
            "n_anggota"    : int(n_anggota),
            "centroid_lat" : centroid_lat,
            "centroid_lon" : centroid_lon,
            "rata_mag"     : rata_mag,
            "rata_depth"   : rata_depth,
            "deskripsi"    : deskripsi,
        })

    dim_kluster = pd.DataFrame(rows)
    # id_kluster dimulai dari 1 (bukan 0) sesuai konvensi primary key database
    dim_kluster.insert(0, "id_kluster", dim_kluster.index + 1)
    log.info(f"  → {len(dim_kluster)} kluster dibangun")
    return dim_kluster


def load_dim_kluster(dim_kluster: pd.DataFrame):
    """
    Load (insert) data DIM_KLUSTER ke PostgreSQL.

    Langkah:
        1. DELETE semua baris lama di DIM_KLUSTER (truncate logis)
        2. INSERT batch menggunakan execute_values (jauh lebih cepat
           dibanding INSERT satu per satu untuk ratusan baris)

    Transaksi di-commit hanya setelah INSERT selesai. Jika terjadi error,
    rollback dilakukan agar data tidak setengah-masuk.
    """
    log.info("[LOAD] Insert ke DIM_KLUSTER ...")
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        # Hapus data lama sebelum insert batch baru
        with conn.cursor() as cur:
            cur.execute("DELETE FROM DIM_KLUSTER")
        conn.commit()

        cols   = list(dim_kluster.columns)
        values = [tuple(r) for r in dim_kluster.itertuples(index=False, name=None)]
        sql    = f"INSERT INTO DIM_KLUSTER ({', '.join(cols)}) VALUES %s"

        with conn.cursor() as cur:
            execute_values(cur, sql, values)   # batch insert satu query
        conn.commit()
        log.info(f"  → {len(dim_kluster)} baris di-load ke DIM_KLUSTER")
    except Exception as e:
        conn.rollback()   # batalkan perubahan jika ada error
        log.error(f"[LOAD] Error: {e}")
        raise
    finally:
        conn.close()


def update_fact_gempa(
    df: pd.DataFrame,
    labels_clean: np.ndarray,
    dim_kluster: pd.DataFrame
):
    """
    Update kolom id_kluster pada tabel FACT_GEMPA di PostgreSQL.

    Proses:
        1. Set semua id_kluster ke NULL (reset bersih)
        2. Buat mapping: label_kluster → id_kluster (foreign key ke DIM_KLUSTER)
        3. Update per kelompok id_kluster menggunakan WHERE id_fakta = ANY(...)
           agar lebih efisien daripada update baris satu per satu.

    Setelah langkah ini, setiap baris FACT_GEMPA memiliki id_kluster yang
    mengacu ke DIM_KLUSTER, melengkapi skema bintang (star schema).
    """
    log.info("[LOAD] Update FACT_GEMPA.id_kluster ...")

    # Buat kamus pemetaan: label integer DBSCAN → id_kluster (PK di DIM_KLUSTER)
    label_to_id  = dict(zip(dim_kluster["label_kluster"], dim_kluster["id_kluster"]))
    df           = df.copy()
    df["id_kluster"] = [label_to_id[l] for l in labels_clean]

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        # Reset semua nilai id_kluster ke NULL terlebih dahulu
        with conn.cursor() as cur:
            cur.execute("UPDATE FACT_GEMPA SET id_kluster = NULL")
        conn.commit()

        # Update per kluster: kirim semua id_fakta dalam satu kluster sekaligus
        # menggunakan array PostgreSQL (ANY) agar lebih sedikit round-trip ke DB
        for id_kluster, group in df.groupby("id_kluster"):
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE FACT_GEMPA SET id_kluster = %s "
                    "WHERE id_fakta = ANY(%s)",
                    (int(id_kluster), group["id_fakta"].tolist())
                )
        conn.commit()
        log.info(f"  → {len(df):,} baris FACT_GEMPA berhasil diupdate")
    except Exception as e:
        conn.rollback()
        log.error(f"[LOAD] Error: {e}")
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
# 5. RINGKASAN
# ─────────────────────────────────────────────

def cetak_ringkasan(terbaik: dict, dim_kluster: pd.DataFrame):
    """
    Cetak ringkasan hasil clustering ke konsol setelah seluruh pipeline selesai.

    Menampilkan:
        - Parameter DBSCAN yang dipilih (ε dan min_samples)
        - Jumlah kluster dan persentase noise awal
        - Nilai metrik evaluasi (Silhouette dan DBI)
        - Tabel ringkasan tiap kluster: jumlah anggota, centroid, rata-rata
          magnitudo dan kedalaman

    Output ini berguna untuk verifikasi cepat dan dokumentasi hasil pada
    Bab 4 skripsi (Analisis Hasil Clustering).
    """
    print("\n" + "=" * 65)
    print("  RINGKASAN HASIL DBSCAN CLUSTERING")
    print("=" * 65)
    print(f"  ε (Epsilon)        : {terbaik['eps']//1000} km")
    print(f"  min_samples        : {terbaik['min_samples']}")
    print(f"  Jumlah Kluster     : {terbaik['n_kluster']}")
    print(f"  Noise Asli (−1)    : {terbaik['n_noise']:,} titik "
          f"({terbaik['pct_noise']:.1f}%) → diisi zona terdekat")
    print(f"  Silhouette Score   : {terbaik['silhouette']}")
    print(f"  Davies-Bouldin Idx : {terbaik['dbi']}")
    print("-" * 65)
    print(f"  {'Label':<8} {'N Gempa':>10} {'Ctrd Lat':>12} "
          f"{'Ctrd Lon':>12} {'Rata Mag':>10} {'Rata Depth':>12}")
    print("-" * 65)
    for _, row in dim_kluster.iterrows():
        print(
            f"  {int(row['label_kluster']):<8} "
            f"{int(row['n_anggota']):>10,} "
            f"{row['centroid_lat']:>12.4f} "
            f"{row['centroid_lon']:>12.4f} "
            f"{row['rata_mag']:>10.2f} "
            f"{row['rata_depth']:>12.2f}"
        )
    print("=" * 65)
    print(f"  k-distance plot : {OUTPUT_KDIST}")
    print(f"  Grafik evaluasi : {OUTPUT_EVAL}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    """
    Orkestrasi pipeline ETL + Clustering secara berurutan:

        1. EXTRACT  — ambil data dari PostgreSQL (load_data)
        2. TRANSFORM — proyeksi koordinat ke UTM (project_to_utm)
        3a. k-distance plot — visualisasi panduan pemilihan ε
        3b. Grid search    — cari parameter DBSCAN optimal
        3c. Plot evaluasi  — simpan grafik Silhouette & DBI
        3d. Tangani noise  — tetapkan label zona terdekat untuk titik noise
        4. LOAD      — simpan DIM_KLUSTER & update FACT_GEMPA ke PostgreSQL
        5. Ringkasan — cetak hasil akhir ke konsol
    """
    log.info("=" * 65)
    log.info("DBSCAN CLUSTERING — Seismisitas Indonesia")
    log.info("Sri Bintang Pratomo (825220070) — UNTAR Prodi SI")
    log.info("=" * 65)

    # 1. Extract: ambil data koordinat gempa dari database
    df = load_data()

    # 2. Transform: proyeksi WGS84 → UTM (meter) agar jarak Euclidean akurat
    X = project_to_utm(df)

    # 3a. Visualisasi k-distance untuk referensi pemilihan ε
    plot_kdistance(X)

    # 3b. Grid search: uji semua kombinasi ε × min_samples, pilih terbaik
    hasil    = grid_search_dbscan(X)
    terbaik  = pilih_parameter_optimal(hasil)

    # 3c. Simpan grafik perbandingan Silhouette Score dan DBI
    plot_evaluasi(hasil, terbaik)

    # 3d. Tetapkan label kluster untuk titik-titik noise (label -1)
    labels_clean = tangani_noise(X, terbaik["labels"])

    # 4. Load hasil ke PostgreSQL: isi DIM_KLUSTER dan update FACT_GEMPA
    dim_kluster = bangun_dim_kluster(df, labels_clean, terbaik)
    load_dim_kluster(dim_kluster)
    update_fact_gempa(df, labels_clean, dim_kluster)

    # 5. Cetak ringkasan akhir ke konsol
    cetak_ringkasan(terbaik, dim_kluster)

    log.info("DBSCAN selesai. Database siap untuk Dashboard.")


if __name__ == "__main__":
    main()
