"""
ETL Pipeline: BMKG Gempa + Patahan Aktif → PostgreSQL Star Schema

Pipeline:
    1. EXTRACT  : Baca CSV (BMKG) + GeoJSON (Patahan PSG/Badan Geologi ESDM)
    2. TRANSFORM : Bangun tabel dimensi, hitung jarak spasial ke patahan terdekat
    3. LOAD      : Insert ke PostgreSQL (Star Schema)

Dependensi:
    pip install pandas geopandas shapely psycopg2-binary sqlalchemy tqdm
"""

# ─────────────────────────────────────────────
# IMPORT LIBRARY
# ─────────────────────────────────────────────
# pandas    : manipulasi data tabular (DataFrame)
# geopandas : ekstensi pandas untuk data geospasial (GeoDataFrame)
# numpy     : operasi numerik
# shapely   : representasi objek geometri (Point, LineString, dll.)
# psycopg2  : koneksi dan eksekusi query ke PostgreSQL
# logging   : pencatatan log proses pipeline
# tqdm      : progress bar saat iterasi data besar

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import psycopg2
from psycopg2.extras import execute_values
import logging
from tqdm import tqdm

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

# Path file sumber data:
# - CSV berisi catatan gempa BMKG dari Januari 2000 s.d. Februari 2026
# - GeoJSON berisi data spasial patahan aktif dari portal ESDM
CSV_PATH     = "bmkg_gempa_indonesia_Ja2000_Fe2026.csv"
GEOJSON_PATH = "Patahan Aktif geoportal.esdm.go.id.geojson"

# Konfigurasi koneksi ke database PostgreSQL lokal
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 5432,
    "dbname"  : "seismisitas_db",
    "user"    : "postgres",
    "password": "black007",   # ← password
}

# Hanya gunakan patahan dengan klasifikasi "Aktif"
# (mengabaikan patahan tidak aktif / historis agar analisis lebih relevan)
FILTER_KLSPTHN = ["Aktif"]

# Threshold: gempa dianggap "dekat" patahan jika < X km
# Nilai 50 km dipilih berdasarkan literatur zona pengaruh patahan aktif
JARAK_DEKAT_KM = 50.0

# Konfigurasi format logging: tampilkan waktu, level, dan pesan
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. EXTRACT
# ─────────────────────────────────────────────
# Tahap Extract: membaca data mentah dari sumber file eksternal.
# Tidak ada transformasi di sini — data dibaca apa adanya.

def extract_gempa(path: str) -> pd.DataFrame:
    """Baca data gempa dari file CSV BMKG ke dalam DataFrame."""
    log.info(f"[EXTRACT] Membaca CSV gempa: {path}")
    df = pd.read_csv(path)
    # Tampilkan jumlah baris dan nama kolom untuk validasi awal
    log.info(f"  → {len(df):,} baris, kolom: {list(df.columns)}")
    return df


def extract_patahan(path: str) -> gpd.GeoDataFrame:
    """Baca data patahan aktif dari GeoJSON, lalu filter hanya yang berstatus 'Aktif'."""
    log.info(f"[EXTRACT] Membaca GeoJSON patahan: {path}")
    gdf = gpd.read_file(path)

    # Filter berdasarkan kolom 'klspthn' (klasifikasi patahan)
    # Hanya patahan aktif yang relevan untuk analisis seismisitas
    gdf_aktif = gdf[gdf["klspthn"].isin(FILTER_KLSPTHN)].copy()
    log.info(f"  → Total fitur: {len(gdf):,} | Aktif: {len(gdf_aktif):,}")
    return gdf_aktif


# ─────────────────────────────────────────────
# 2. TRANSFORM
# ─────────────────────────────────────────────
# Tahap Transform: mengubah data mentah menjadi struktur Star Schema.
# Star Schema terdiri dari:
#   - Tabel DIMENSI (DIM_*): atribut deskriptif (waktu, lokasi, magnitude, dll.)
#   - Tabel FAKTA (FACT_GEMPA): data pengukuran + foreign key ke semua dimensi

def transform_dim_waktu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bangun tabel dimensi waktu dari kolom tanggal ('tgl') dan waktu kejadian ('ot').
    Dimensi waktu menyimpan atribut temporal seperti tahun, bulan, hari, jam,
    nama hari, kuartal, dan status akhir pekan — berguna untuk analisis tren musiman.
    """
    log.info("[TRANSFORM] Membangun DIM_WAKTU ...")

    # Gabung kolom tgl + ot menjadi datetime untuk ekstraksi atribut temporal
    dt = pd.to_datetime(df["tgl"] + " " + df["ot"], errors="coerce")

    waktu = pd.DataFrame({
        "tgl"        : df["tgl"],
        "ot"         : df["ot"],
        "tahun"      : df["tahun"],
        "bulan"      : df["bulan"],
        "hari"       : dt.dt.day,
        "jam"        : dt.dt.hour,
        "hari_minggu": dt.dt.day_name(),     # Senin, Selasa, dst.
        "kuartal"    : dt.dt.quarter,        # Q1–Q4
        "is_weekend" : dt.dt.dayofweek >= 5, # Sabtu=5, Minggu=6
    # Deduplikasi: satu baris per kombinasi tanggal+waktu unik
    }).drop_duplicates(subset=["tgl", "ot"]).reset_index(drop=True)

    # Tambah surrogate key (id urut mulai dari 1)
    waktu.insert(0, "id_waktu", waktu.index + 1)
    log.info(f"  → {len(waktu):,} entri unik")
    return waktu


def transform_dim_lokasi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bangun tabel dimensi lokasi dari koordinat (lat, lon) beserta
    nama provinsi dan keterangan lokasi.
    Deduplikasi dilakukan berdasarkan pasangan lat+lon unik.
    """
    log.info("[TRANSFORM] Membangun DIM_LOKASI ...")
    lokasi = df[["lat", "lon", "provinsi", "remark"]].drop_duplicates(
        subset=["lat", "lon"]
    ).reset_index(drop=True)

    # Tambah surrogate key
    lokasi.insert(0, "id_lokasi", lokasi.index + 1)
    log.info(f"  → {len(lokasi):,} lokasi unik")
    return lokasi


def transform_dim_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bangun tabel dimensi magnitude berdasarkan kategori tekstual
    (Mikro, Minor, Ringan, Sedang, Kuat, Besar, Sangat Besar).
    Ditambahkan kolom urutan dan rentang skala Richter untuk keperluan
    visualisasi dan pengurutan.
    """
    log.info("[TRANSFORM] Membangun DIM_MAGNITUDE ...")

    # Urutan numerik untuk sorting kategori dari kecil ke besar
    urutan = {
        "Mikro": 1, "Minor": 2, "Ringan": 3,
        "Sedang": 4, "Kuat": 5, "Besar": 6, "Sangat Besar": 7
    }
    # Deskripsi rentang nilai skala Richter per kategori
    rentang = {
        "Mikro"       : "< 2.0",
        "Minor"       : "2.0 - 2.9",
        "Ringan"      : "3.0 - 3.9",
        "Sedang"      : "4.0 - 4.9",
        "Kuat"        : "5.0 - 5.9",
        "Besar"       : "6.0 - 6.9",
        "Sangat Besar": ">= 7.0",
    }
    mag = df[["kategori_magnitude"]].drop_duplicates().reset_index(drop=True)
    mag["urutan"]        = mag["kategori_magnitude"].map(urutan)
    mag["skala_richter"] = mag["kategori_magnitude"].map(rentang)
    mag = mag.sort_values("urutan").reset_index(drop=True)
    mag.insert(0, "id_magnitude", mag.index + 1)
    log.info(f"  → {len(mag):,} kategori")
    return mag


def transform_dim_kedalaman(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bangun tabel dimensi kedalaman gempa dengan klasifikasi:
    - Dangkal  : 0–70 km   (paling berbahaya, energi dekat permukaan)
    - Menengah : 70–300 km
    - Dalam    : > 300 km
    Klasifikasi mengikuti standar USGS/BMKG.
    """
    log.info("[TRANSFORM] Membangun DIM_KEDALAMAN ...")
    rentang_km = {
        "Dangkal" : "0 - 70 km",
        "Menengah": "70 - 300 km",
        "Dalam"   : "> 300 km",
    }
    ked = df[["kategori_kedalaman"]].drop_duplicates().reset_index(drop=True)
    ked["rentang_km"] = ked["kategori_kedalaman"].map(rentang_km)
    ked.insert(0, "id_kedalaman", ked.index + 1)
    log.info(f"  → {len(ked):,} kategori")
    return ked


def transform_dim_patahan(gdf_aktif: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Bangun tabel dimensi patahan dari GeoDataFrame patahan aktif.
    Kolom yang diambil: kode simbol, nama, klasifikasi, riwayat gempa,
    panjang patahan (km), lokasi, dan kode fitur (fcode).
    Data bersumber dari Peta Geologi Patahan Aktif — ESDM.
    """
    log.info("[TRANSFORM] Membangun DIM_PATAHAN ...")
    pat = gdf_aktif[[
        "simobj", "namobj", "klspthn", "sjrhgempa",
        "pjgpthn", "lokasi", "fcode"
    ]].copy().reset_index(drop=True)

    # Rename kolom agar lebih deskriptif di database
    pat.columns = [
        "kode_simbol", "nama_patahan", "klasifikasi",
        "riwayat_gempa", "panjang_km", "lokasi", "fcode"
    ]
    pat.insert(0, "id_patahan", pat.index + 1)
    log.info(f"  → {len(pat):,} patahan aktif")
    return pat


def hitung_jarak_ke_patahan(
    df_gempa: pd.DataFrame,
    gdf_patahan: gpd.GeoDataFrame,
    dim_patahan: pd.DataFrame
):
    """
    Hitung jarak spasial (km) dari setiap titik gempa ke patahan aktif terdekat.

    Langkah:
    1. Konversi koordinat gempa ke GeoDataFrame dengan CRS WGS84 (EPSG:4326)
    2. Proyeksikan ke UTM Zone 50S (EPSG:32750) agar jarak dalam meter (bukan derajat)
    3. Untuk tiap titik gempa, hitung jarak ke gabungan seluruh geometri patahan
    4. Identifikasi patahan terdekat (id_patahan) dari jarak individual per patahan

    Catatan: UTM Zone 50S dipilih karena mencakup sebagian besar wilayah Indonesia.
    """
    log.info("[TRANSFORM] Menghitung jarak spasial gempa -> patahan terdekat ...")

    # Buat GeoDataFrame titik gempa dari kolom lat/lon
    gempa_pts = gpd.GeoDataFrame(
        df_gempa[["lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(df_gempa["lon"], df_gempa["lat"]),
        crs="EPSG:4326"  # Sistem koordinat geografis standar (derajat)
    )

    # Proyeksikan ke sistem koordinat metrik (meter) agar distance() akurat
    gempa_utm   = gempa_pts.to_crs("EPSG:32750")
    patahan_utm = gdf_patahan.to_crs("EPSG:32750").reset_index(drop=True)

    # union_all() / unary_union: gabungkan semua geometri patahan jadi satu objek
    # untuk efisiensi kalkulasi jarak minimum (tidak perlu looping per patahan dulu)
    # Kompatibel dengan semua versi GeoPandas
    try:
        patahan_union = patahan_utm.geometry.union_all()   # GeoPandas >= 0.14
    except AttributeError:
        patahan_union = patahan_utm.geometry.unary_union   # GeoPandas lama

    jarak_km_list   = []
    id_patahan_list = []

    # Iterasi setiap titik gempa — gunakan tqdm untuk progress bar
    for _, row in tqdm(gempa_utm.iterrows(), total=len(gempa_utm), desc="  Hitung jarak"):
        pt = row.geometry

        # Hitung jarak dari titik ke union patahan (hasil dalam meter → konversi ke km)
        dist_m   = pt.distance(patahan_union)
        jarak_km = dist_m / 1000.0
        jarak_km_list.append(round(jarak_km, 3))

        # Cari patahan MANA yang paling dekat (untuk foreign key id_patahan)
        jarak_individual = patahan_utm.geometry.distance(pt)
        idx_terdekat     = jarak_individual.idxmin()
        id_pat           = dim_patahan.loc[idx_terdekat, "id_patahan"]
        id_patahan_list.append(int(id_pat))

    # Kembalikan dua Series: jarak (km) dan id patahan terdekat
    return (
        pd.Series(jarak_km_list,   name="jarak_patahan_km"),
        pd.Series(id_patahan_list, name="id_patahan")
    )


def transform_fact_gempa(
    df: pd.DataFrame,
    dim_waktu: pd.DataFrame,
    dim_lokasi: pd.DataFrame,
    dim_magnitude: pd.DataFrame,
    dim_kedalaman: pd.DataFrame,
    jarak_series: pd.Series,
    id_patahan_series: pd.Series,
) -> pd.DataFrame:
    """
    Bangun tabel fakta FACT_GEMPA yang menjadi pusat Star Schema.

    Tabel fakta menyimpan:
    - Foreign key ke semua tabel dimensi (id_waktu, id_lokasi, dst.)
    - Nilai ukuran (measures): magnitude, kedalaman, jarak ke patahan
    - Flag dekat_patahan: True jika jarak < JARAK_DEKAT_KM (50 km)
    - id_kluster: diisi None dulu, akan diupdate setelah proses clustering

    Join dilakukan secara bertahap (merge) menggunakan natural key
    dari masing-masing dimensi.
    """
    log.info("[TRANSFORM] Membangun FACT_GEMPA ...")

    # Join ke DIM_WAKTU berdasarkan pasangan tgl+ot
    df_merge = df.merge(
        dim_waktu[["id_waktu", "tgl", "ot"]], on=["tgl", "ot"], how="left"
    )
    # Join ke DIM_LOKASI berdasarkan koordinat lat+lon
    df_merge = df_merge.merge(
        dim_lokasi[["id_lokasi", "lat", "lon"]], on=["lat", "lon"], how="left"
    )
    # Join ke DIM_MAGNITUDE berdasarkan kategori tekstual
    df_merge = df_merge.merge(
        dim_magnitude[["id_magnitude", "kategori_magnitude"]],
        on="kategori_magnitude", how="left"
    )
    # Join ke DIM_KEDALAMAN berdasarkan kategori kedalaman
    df_merge = df_merge.merge(
        dim_kedalaman[["id_kedalaman", "kategori_kedalaman"]],
        on="kategori_kedalaman", how="left"
    )

    # Susun kolom akhir tabel fakta
    fact = pd.DataFrame({
        "id_waktu"        : df_merge["id_waktu"].astype(int),
        "id_lokasi"       : df_merge["id_lokasi"].astype(int),
        "id_magnitude"    : df_merge["id_magnitude"].astype(int),
        "id_kedalaman"    : df_merge["id_kedalaman"].astype(int),
        "id_patahan"      : id_patahan_series.values,
        "id_kluster"      : None,                             # Diisi setelah clustering
        "nilai_magnitude" : df["mag"].values,                 # Nilai numerik asli
        "nilai_kedalaman" : df["depth"].values,               # Nilai numerik asli (km)
        "jarak_patahan_km": jarak_series.values,
        "dekat_patahan"   : (jarak_series < JARAK_DEKAT_KM).values,  # Boolean flag
    })

    log.info(f"  → {len(fact):,} baris FACT_GEMPA")
    return fact


# ─────────────────────────────────────────────
# 3. DDL — Buat Skema PostgreSQL
# ─────────────────────────────────────────────
# DDL (Data Definition Language): SQL untuk mendefinisikan struktur tabel.
# Menggunakan pola "CREATE TABLE IF NOT EXISTS" agar pipeline bisa
# dijalankan ulang tanpa error jika tabel sudah ada.
#
# Struktur Star Schema:
#   DIM_WAKTU, DIM_LOKASI, DIM_MAGNITUDE, DIM_KEDALAMAN,
#   DIM_PATAHAN, DIM_KLUSTER  →  FACT_GEMPA (center)
#
# Index dibuat pada kolom foreign key di FACT_GEMPA
# untuk mempercepat query join dan filter analitik.

DDL = """
CREATE TABLE IF NOT EXISTS DIM_WAKTU (
    id_waktu     SERIAL PRIMARY KEY,
    tgl          DATE        NOT NULL,
    ot           TIME        NOT NULL,
    tahun        SMALLINT    NOT NULL,
    bulan        SMALLINT    NOT NULL,
    hari         SMALLINT    NOT NULL,
    jam          SMALLINT    NOT NULL,
    hari_minggu  VARCHAR(10),
    kuartal      SMALLINT,
    is_weekend   BOOLEAN
);

CREATE TABLE IF NOT EXISTS DIM_LOKASI (
    id_lokasi    SERIAL PRIMARY KEY,
    lat          DOUBLE PRECISION NOT NULL,
    lon          DOUBLE PRECISION NOT NULL,
    provinsi     VARCHAR(100),
    remark       TEXT
);

CREATE TABLE IF NOT EXISTS DIM_MAGNITUDE (
    id_magnitude       SERIAL PRIMARY KEY,
    kategori_magnitude VARCHAR(30) NOT NULL,
    skala_richter      VARCHAR(20),
    urutan             SMALLINT
);

CREATE TABLE IF NOT EXISTS DIM_KEDALAMAN (
    id_kedalaman       SERIAL PRIMARY KEY,
    kategori_kedalaman VARCHAR(30) NOT NULL,
    rentang_km         VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS DIM_PATAHAN (
    id_patahan    SERIAL PRIMARY KEY,
    kode_simbol   TEXT,
    nama_patahan  TEXT,
    klasifikasi   TEXT,
    riwayat_gempa TEXT,
    panjang_km    NUMERIC(10,2),
    lokasi        TEXT,
    fcode         TEXT
);

CREATE TABLE IF NOT EXISTS DIM_KLUSTER (
    id_kluster    SERIAL PRIMARY KEY,
    label_kluster INTEGER,
    n_anggota     INTEGER,
    centroid_lat  DOUBLE PRECISION,
    centroid_lon  DOUBLE PRECISION,
    rata_mag      NUMERIC(5,2),
    rata_depth    NUMERIC(8,2),
    deskripsi     TEXT
);

-- Tabel fakta utama: setiap baris = satu kejadian gempa
-- Semua kolom id_* adalah foreign key ke tabel dimensi masing-masing
CREATE TABLE IF NOT EXISTS FACT_GEMPA (
    id_fakta          SERIAL PRIMARY KEY,
    id_waktu          INTEGER REFERENCES DIM_WAKTU(id_waktu),
    id_lokasi         INTEGER REFERENCES DIM_LOKASI(id_lokasi),
    id_magnitude      INTEGER REFERENCES DIM_MAGNITUDE(id_magnitude),
    id_kedalaman      INTEGER REFERENCES DIM_KEDALAMAN(id_kedalaman),
    id_patahan        INTEGER REFERENCES DIM_PATAHAN(id_patahan),
    id_kluster        INTEGER REFERENCES DIM_KLUSTER(id_kluster),
    nilai_magnitude   NUMERIC(4,1),
    nilai_kedalaman   NUMERIC(8,2),
    jarak_patahan_km  NUMERIC(10,3),
    dekat_patahan     BOOLEAN
);

-- Index pada foreign key mempercepat query JOIN dan GROUP BY
CREATE INDEX IF NOT EXISTS idx_fact_waktu     ON FACT_GEMPA(id_waktu);
CREATE INDEX IF NOT EXISTS idx_fact_lokasi    ON FACT_GEMPA(id_lokasi);
CREATE INDEX IF NOT EXISTS idx_fact_magnitude ON FACT_GEMPA(id_magnitude);
CREATE INDEX IF NOT EXISTS idx_fact_patahan   ON FACT_GEMPA(id_patahan);
CREATE INDEX IF NOT EXISTS idx_fact_kluster   ON FACT_GEMPA(id_kluster);
"""


# ─────────────────────────────────────────────
# 4. LOAD ke PostgreSQL
# ─────────────────────────────────────────────
# Tahap Load: insert semua DataFrame hasil transform ke PostgreSQL.
# Menggunakan execute_values (psycopg2) untuk bulk insert yang efisien
# — jauh lebih cepat dibanding INSERT satu per satu.

def get_connection():
    """Buka koneksi ke PostgreSQL menggunakan konfigurasi DB_CONFIG."""
    return psycopg2.connect(**DB_CONFIG)


def execute_ddl(conn):
    """Eksekusi DDL untuk membuat semua tabel jika belum ada."""
    log.info("[LOAD] Membuat skema tabel PostgreSQL ...")
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    log.info("  → DDL selesai")


def load_table(conn, df: pd.DataFrame, table_name: str, exclude_cols=None):
    """
    Insert seluruh baris DataFrame ke tabel PostgreSQL secara bulk.

    Parameter:
    - conn         : koneksi psycopg2 aktif
    - df           : DataFrame yang akan diinsert
    - table_name   : nama tabel target di database
    - exclude_cols : kolom yang dikecualikan (misal: kolom SERIAL auto-increment)
    """
    if exclude_cols:
        df = df.drop(columns=exclude_cols, errors="ignore")

    # Ganti NaN dengan None agar psycopg2 insert sebagai NULL di PostgreSQL
    df = df.where(pd.notnull(df), None)

    cols   = list(df.columns)
    values = [tuple(row) for row in df.itertuples(index=False, name=None)]

    # Bangun query INSERT dinamis sesuai nama kolom DataFrame
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES %s"

    with conn.cursor() as cur:
        # page_size=1000: insert 1000 baris per batch untuk performa optimal
        execute_values(cur, sql, values, page_size=1000)
    conn.commit()
    log.info(f"  → {len(df):,} baris di-load ke {table_name}")


def load_all(
    dim_waktu, dim_lokasi, dim_magnitude,
    dim_kedalaman, dim_patahan, fact_gempa
):
    """
    Orkestrasi load semua tabel ke PostgreSQL secara berurutan.

    Urutan load penting: dimensi harus diisi SEBELUM fakta,
    karena FACT_GEMPA memiliki foreign key ke semua tabel DIM_*.
    Jika terjadi error, transaksi di-rollback untuk menjaga konsistensi data.
    """
    log.info("[LOAD] Mulai proses load ke PostgreSQL ...")
    conn = get_connection()
    try:
        execute_ddl(conn)

        # Load dimensi terlebih dahulu (urutan bebas antar dimensi)
        load_table(conn, dim_waktu,     "DIM_WAKTU")
        load_table(conn, dim_lokasi,    "DIM_LOKASI")
        load_table(conn, dim_magnitude, "DIM_MAGNITUDE")
        load_table(conn, dim_kedalaman, "DIM_KEDALAMAN")
        load_table(conn, dim_patahan,   "DIM_PATAHAN")

        # Load tabel fakta terakhir (bergantung pada semua dimensi di atas)
        # exclude_cols=["id_fakta"] karena kolom ini SERIAL (auto-increment di DB)
        load_table(conn, fact_gempa,    "FACT_GEMPA", exclude_cols=["id_fakta"])

        log.info("[LOAD] Semua tabel berhasil di-load!")
    except Exception as e:
        # Rollback jika terjadi error agar data tidak setengah tersimpan
        conn.rollback()
        log.error(f"[LOAD] Error: {e}")
        raise
    finally:
        # Selalu tutup koneksi meskipun terjadi error
        conn.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
# Fungsi utama yang mengorkestrasi seluruh alur ETL secara berurutan:
# Extract → Transform → Load

def main():
    log.info("=" * 60)
    log.info("ETL PIPELINE: Seismisitas Indonesia — Sri Bintang Pratomo")
    log.info("=" * 60)

    # ── TAHAP 1: EXTRACT ──────────────────────────────────────────
    # Baca data mentah dari file sumber
    df_gempa    = extract_gempa(CSV_PATH)
    gdf_patahan = extract_patahan(GEOJSON_PATH)

    # ── TAHAP 2: TRANSFORM ────────────────────────────────────────
    # Bangun semua tabel dimensi dari data gempa
    dim_waktu     = transform_dim_waktu(df_gempa)
    dim_lokasi    = transform_dim_lokasi(df_gempa)
    dim_magnitude = transform_dim_magnitude(df_gempa)
    dim_kedalaman = transform_dim_kedalaman(df_gempa)

    # Bangun dimensi patahan dari data GeoJSON
    dim_patahan   = transform_dim_patahan(gdf_patahan)

    # Hitung jarak spasial gempa ke patahan terdekat (proses paling berat)
    jarak_series, id_patahan_series = hitung_jarak_ke_patahan(
        df_gempa, gdf_patahan, dim_patahan
    )

    # Bangun tabel fakta utama dengan semua foreign key dan nilai ukuran
    fact_gempa = transform_fact_gempa(
        df_gempa, dim_waktu, dim_lokasi,
        dim_magnitude, dim_kedalaman,
        jarak_series, id_patahan_series
    )

    # ── TAHAP 3: LOAD ─────────────────────────────────────────────
    # Insert semua tabel ke PostgreSQL
    load_all(
        dim_waktu, dim_lokasi, dim_magnitude,
        dim_kedalaman, dim_patahan, fact_gempa
    )

    log.info("=" * 60)
    log.info("ETL selesai. Database siap untuk clustering.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
