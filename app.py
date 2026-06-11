"""
Struktur:
    app.py  ← file utama (jalankan ini)

Halaman:
    1. Ikhtisar (Overview)    — metrik, tren, distribusi
    2. Peta Interaktif        — episenter, patahan, KDE
    3. Analisis Kluster       — hasil DBSCAN, evaluasi

Dependensi:
    pip install dash dash-bootstrap-components plotly pandas
               geopandas psycopg2-binary scikit-learn sqlalchemy
"""

# ─────────────────────────────────────────────
# IMPOR LIBRARY
# ─────────────────────────────────────────────
# Pandas & NumPy: manipulasi dan komputasi data tabular
import pandas as pd
import geopandas as gpd   # Pandas versi geospasial — membaca shapefile/GeoJSON
import numpy as np
import json
import os   # Membaca environment variable (DATABASE_URL, PORT) saat deploy
# psycopg2 / SQLAlchemy: driver & ORM untuk koneksi ke PostgreSQL
import psycopg2
from sqlalchemy import create_engine

# Dash: framework web interaktif berbasis Python (tidak perlu menulis JS)
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc   # Komponen UI bergaya Bootstrap
# Plotly: membuat grafik interaktif (bar, pie, map, scatter, heatmap)
import plotly.express as px
import plotly.graph_objects as go
# Scikit-learn: algoritma DBSCAN dan metrik evaluasi kluster
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
# Kredensial koneksi database PostgreSQL untuk pengembangan LOKAL.
# Saat deploy (mis. Railway), nilai ini DIABAIKAN — koneksi memakai
# environment variable DATABASE_URL yang otomatis disediakan platform.
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 5432,
    "dbname"  : "seismisitas_db",
    "user"    : "postgres",
    "password": "black007",
}
# Path file GeoJSON patahan aktif dari Geoportal ESDM
GEOJSON_PATH = "Patahan Aktif geoportal.esdm.go.id.geojson"
# Gaya peta dasar — open-street-map tidak memerlukan token API
MAPBOX_STYLE = "open-street-map"

# Parameter DBSCAN final dari skripsi
# eps: jarak maksimum antar titik agar dianggap bertetangga (200 km → meter)
EPS_DEFAULT        = 200_000   # 200 km dalam meter
# min_samples: minimum titik dalam radius eps untuk membentuk inti kluster
MIN_SAMPLES_DEFAULT = 1000

# Palet warna untuk kluster — dipilih ramah buta warna (Wong color-blind palette)
WARNA_KLUSTER = [
    "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
]

# ─────────────────────────────────────────────
# KONEKSI DATABASE
# ─────────────────────────────────────────────
def get_engine():
    """
    Membuat SQLAlchemy engine untuk koneksi ke PostgreSQL.
    Engine ini digunakan oleh pandas.read_sql() untuk menarik data
    dari data warehouse tanpa menulis query koneksi secara manual.

    Prioritas koneksi:
      1. Jika ada environment variable DATABASE_URL (kondisi deploy, mis.
         Railway/Heroku) → pakai URL tersebut.
      2. Jika tidak ada (pengembangan lokal) → pakai DB_CONFIG di atas.
    """
    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        # Railway memberi URL berformat "postgresql://..." atau "postgres://...".
        # SQLAlchemy + psycopg2 mengharuskan prefix "postgresql+psycopg2://",
        # jadi kita normalisasi terlebih dahulu.
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+psycopg2://", 1
            )
        return create_engine(database_url)

    # ── Fallback: koneksi lokal memakai DB_CONFIG ──
    cfg = DB_CONFIG
    return create_engine(
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )

# ─────────────────────────────────────────────
# LOAD DATA (cache di memori)
# ─────────────────────────────────────────────
# Data dimuat sekali saat aplikasi pertama dijalankan dan disimpan di RAM.
# Pendekatan ini menghindari query berulang ke database setiap ada interaksi.
print("Memuat data dari PostgreSQL...")
engine = get_engine()

# ┌─────────────────────────────────────────────────────────────┐
# │  SUMBER DATA UTAMA — Star Schema Join                       │
# │  Tabel: FACT_GEMPA (tabel pusat)                            │
# │    JOIN  DIM_WAKTU      → tgl, ot, tahun, bulan             │
# │    JOIN  DIM_LOKASI     → lat, lon, provinsi, remark        │
# │    JOIN  DIM_MAGNITUDE  → kategori_magnitude                │
# │    JOIN  DIM_KEDALAMAN  → kategori_kedalaman                │
# │    LEFT JOIN DIM_KLUSTER → label_kluster (nullable)         │
# │  Sumber GeoJSON: PSG/Badan Geologi ESDM (patahan aktif)     │
# └─────────────────────────────────────────────────────────────┘
# Query JOIN star-schema: menggabungkan fact_gempa dengan semua tabel dimensi.
# Arsitektur data warehouse bintang ini memisahkan fakta (nilai numerik)
# dari dimensi (atribut deskriptif) agar query analitik lebih efisien.
df_gempa = pd.read_sql("""
    SELECT
        f.id_fakta,
        w.tgl, w.ot, w.tahun, w.bulan,
        l.lat, l.lon, l.provinsi, l.remark,
        f.nilai_magnitude, f.nilai_kedalaman,
        f.jarak_patahan_km, f.dekat_patahan,
        m.kategori_magnitude,
        k.kategori_kedalaman,
        f.id_kluster,
        kl.label_kluster
    FROM fact_gempa f
    JOIN dim_waktu     w  ON f.id_waktu    = w.id_waktu
    JOIN dim_lokasi    l  ON f.id_lokasi   = l.id_lokasi
    JOIN dim_magnitude m  ON f.id_magnitude = m.id_magnitude
    JOIN dim_kedalaman k  ON f.id_kedalaman = k.id_kedalaman
    LEFT JOIN dim_kluster kl ON f.id_kluster = kl.id_kluster
""", engine)

# Memuat dimensi kluster yang sudah tersimpan di database (hasil DBSCAN final)
df_kluster = pd.read_sql(
    "SELECT * FROM dim_kluster ORDER BY id_kluster", engine
)

# Membaca GeoJSON patahan aktif, lalu filter hanya segmen berkelas "Aktif"
gdf_patahan = gpd.read_file(GEOJSON_PATH)
gdf_aktif   = gdf_patahan[gdf_patahan["klspthn"] == "Aktif"].copy()

print(f"  Data gempa   : {len(df_gempa):,} baris")
print(f"  Kluster      : {len(df_kluster)} kluster")
print(f"  Patahan aktif: {len(gdf_aktif)} segmen")

# ─────────────────────────────────────────────
# INISIALISASI APP
# ─────────────────────────────────────────────
# Membuat instance aplikasi Dash.
# suppress_callback_exceptions=True diperlukan karena komponen halaman
# dibuat secara dinamis (multi-page routing) sehingga belum ada saat startup.
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Visual Analytics Seismisitas Indonesia"
)
# server di-expose agar kompatibel dengan deployment WSGI (Gunicorn/uWSGI)
server = app.server

# ─────────────────────────────────────────────
# KOMPONEN NAVBAR
# ─────────────────────────────────────────────
# Navbar tetap di bagian atas (sticky="top") dengan tautan navigasi
# ke tiga halaman utama dashboard.
navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.I(className="fas fa-globe-asia me-2")),
                dbc.Col(dbc.NavbarBrand(
                    "Visual Analytics Seismisitas Indonesia",
                    className="fw-bold"
                )),
            ], align="center"),
            href="/",
        ),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("📊 Ikhtisar",       href="/",         active="exact")),
            dbc.NavItem(dbc.NavLink("🗺️ Peta Interaktif", href="/peta",     active="exact")),
            dbc.NavItem(dbc.NavLink("🔵 Analisis Kluster", href="/kluster", active="exact")),
        ], navbar=True, className="ms-auto"),
    ]),
    color="dark", dark=True, sticky="top"
)

# ─────────────────────────────────────────────
# LAYOUT UTAMA
# ─────────────────────────────────────────────
# Kerangka halaman global.
# dcc.Location melacak URL browser → callback routing memilih halaman yang ditampilkan.
# html.Div id="page-content" adalah slot tempat konten setiap halaman disisipkan.
app.layout = html.Div([
    dcc.Location(id="url"),
    navbar,
    html.Div(id="page-content", style={"minHeight": "90vh", "backgroundColor": "#f8f9fa"}),
    html.Footer(
        dbc.Container(
            html.P(
                "Sri Bintang Pratomo (825220070) — UNTAR Prodi SI | "
                "Data: BMKG 2000–2026 & PSG/Badan Geologi ESDM",
                className="text-center text-muted py-2 mb-0 small"
            )
        ),
        className="bg-light border-top"
    )
])

# ═════════════════════════════════════════════
# HALAMAN 1: IKHTISAR
# ═════════════════════════════════════════════
def layout_ikhtisar():
    """
    Membangun layout halaman Ikhtisar yang memuat:
    - 4 kartu KPI (Key Performance Indicator) ringkasan data
    - Slider filter rentang tahun
    - Grafik tren kejadian per tahun (bar chart)
    - Distribusi kategori magnitudo (donut pie chart)
    - Tabel 5 provinsi dengan gempa terbanyak

    Nilai KPI dihitung dari seluruh dataset (sebelum filtering),
    sedangkan grafik & tabel diperbarui secara reaktif oleh callback.
    """
    tahun_min = int(df_gempa["tahun"].min())
    tahun_max = int(df_gempa["tahun"].max())

    # ── Hitung KPI statis (ditampilkan di kartu atas) ──────────────────────

    # KPI 1 — Total Kejadian
    # Sumber : FACT_GEMPA (COUNT semua baris)
    # Kolom  : id_fakta
    total        = len(df_gempa)

    # KPI 2 — Rata-rata Magnitudo
    # Sumber : FACT_GEMPA
    # Kolom  : nilai_magnitude
    rata_mag     = df_gempa["nilai_magnitude"].mean()

    # KPI 3 — Provinsi Paling Aktif
    # Sumber : FACT_GEMPA JOIN DIM_LOKASI
    # Kolom  : DIM_LOKASI.provinsi → value_counts() → ambil index[0]
    prov_aktif   = df_gempa["provinsi"].value_counts().index[0]

    # KPI 4 — Persentase Gempa Dangkal
    # Sumber : FACT_GEMPA JOIN DIM_KEDALAMAN
    # Kolom  : DIM_KEDALAMAN.kategori_kedalaman == "Dangkal"
    pct_dangkal  = (df_gempa["kategori_kedalaman"] == "Dangkal").mean() * 100

    return dbc.Container([
        html.H4("📊 Ikhtisar Seismisitas Indonesia",
                className="my-3 fw-bold text-dark"),

        # ── Kartu KPI — empat metrik ringkas ────────────────────────────────
        # KPI 1: FACT_GEMPA (COUNT id_fakta)
        # KPI 2: FACT_GEMPA (AVG nilai_magnitude)
        # KPI 3: FACT_GEMPA JOIN DIM_LOKASI (MAX COUNT provinsi)
        # KPI 4: FACT_GEMPA JOIN DIM_KEDALAMAN (% kategori_kedalaman = Dangkal)
        dbc.Row([
            dbc.Col(_kpi_card("Total Kejadian",      f"{total:,}",          "gempa (2000–2026)", "primary"), md=3),
            dbc.Col(_kpi_card("Rata-rata Magnitudo", f"{rata_mag:.2f} M",   "skala Richter",     "warning"), md=3),
            dbc.Col(_kpi_card("Provinsi Paling Aktif", prov_aktif,          "kejadian terbanyak","danger"),  md=3),
            dbc.Col(_kpi_card("Gempa Dangkal",       f"{pct_dangkal:.1f}%", "kedalaman 0–70 km", "success"), md=3),
        ], className="mb-4"),

        # ── Filter Tahun — mengontrol semua grafik di bawahnya ──────────────
        dbc.Card([
            dbc.CardBody([
                html.Label("Filter Rentang Tahun:", className="fw-semibold"),
                # RangeSlider menghasilkan list [tahun_awal, tahun_akhir]
                # yang menjadi input callback update_ikhtisar()
                dcc.RangeSlider(
                    id="slider-tahun",
                    min=tahun_min, max=tahun_max,
                    value=[tahun_min, tahun_max],
                    marks={
                        y: {"label": str(y), "style": {"fontSize": "11px"}}
                        for y in range(tahun_min, tahun_max + 1, 5)
                    },
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ])
        ], className="mb-4"),

        # ── Grafik Tren (kiri) + Pie Chart (kanan) — lebar 8:4 kolom ────────
        # Tren Tahunan : FACT_GEMPA JOIN DIM_WAKTU
        #                GROUP BY DIM_WAKTU.tahun → COUNT(*)
        # Pie Chart    : FACT_GEMPA JOIN DIM_MAGNITUDE
        #                GROUP BY DIM_MAGNITUDE.kategori_magnitude → COUNT(*)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 Tren Kejadian Gempa per Tahun"),
                    dbc.CardBody(dcc.Graph(id="chart-tren"))
                ])
            ], md=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🥧 Distribusi Kategori Magnitudo"),
                    dbc.CardBody(dcc.Graph(id="chart-pie"))
                ])
            ], md=4),
        ], className="mb-4"),

        # ── Tabel 5 Provinsi Teratas ─────────────────────────────────────────
        # Sumber : FACT_GEMPA JOIN DIM_LOKASI
        # Kolom  : DIM_LOKASI.provinsi → value_counts().head(5)
        dbc.Card([
            dbc.CardHeader("🏆 Lima Provinsi dengan Kejadian Gempa Terbanyak"),
            dbc.CardBody(html.Div(id="tabel-provinsi"))
        ]),

    ], fluid=True, className="py-3")


def _kpi_card(judul, nilai, sub, warna):
    """
    Helper untuk membuat kartu KPI seragam.
    judul: label metrik  |  nilai: angka utama  |  sub: keterangan kecil
    warna: kelas Bootstrap ('primary', 'warning', dst.)
    """
    return dbc.Card([
        dbc.CardBody([
            html.H6(judul, className="text-muted small mb-1"),
            html.H3(nilai, className=f"fw-bold text-{warna} mb-0"),
            html.Small(sub, className="text-muted"),
        ])
    ], className="shadow-sm h-100")


# ═════════════════════════════════════════════
# HALAMAN 2: PETA INTERAKTIF
# ═════════════════════════════════════════════
def layout_peta():
    """
    Membangun layout halaman Peta Interaktif dengan:
    - Panel filter (tahun, magnitudo, kedalaman, layer) di kiri
    - Peta Mapbox full-height di kanan

    Tiga layer yang dapat diaktifkan:
      1. Episenter — titik lokasi gempa, warna berdasarkan kedalaman
      2. Patahan Aktif — garis patahan dari GeoJSON ESDM
      3. Heatmap KDE — visualisasi densitas episenter
    """
    tahun_min = int(df_gempa["tahun"].min())
    tahun_max = int(df_gempa["tahun"].max())

    # Ambil nilai unik kategori untuk opsi filter checklist
    kat_mag = sorted(df_gempa["kategori_magnitude"].unique())
    kat_ked = sorted(df_gempa["kategori_kedalaman"].unique())

    return dbc.Container([
        html.H4("🗺️ Peta Interaktif Episenter Gempa",
                className="my-3 fw-bold text-dark"),

        dbc.Row([
            # ── Panel Filter kiri (lebar 2 kolom Bootstrap) ─────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("⚙️ Filter Data"),
                    dbc.CardBody([
                        html.Label("Rentang Tahun:", className="fw-semibold small"),
                        # Default ditampilkan: seluruh rentang tahun
                        dcc.RangeSlider(
                            id="peta-slider-tahun",
                            min=tahun_min, max=tahun_max,
                            # Default: tampilkan seluruh rentang tahun yang tersedia
                            value=[tahun_min, tahun_max],
                            # Panel samping sempit → label tiap 10 tahun + titik
                            # ujung (tahun_max) agar tidak saling menimpa.
                            marks={
                                **{
                                    y: {"label": str(y),
                                        "style": {"fontSize": "10px"}}
                                    for y in range(tahun_min, tahun_max + 1, 10)
                                },
                                tahun_max: {"label": str(tahun_max),
                                            "style": {"fontSize": "10px"}},
                            },
                            tooltip={"placement": "bottom"}
                        ),
                        html.Hr(),
                        html.Label("Kategori Magnitudo:", className="fw-semibold small"),
                        # Default: hanya gempa sedang ke atas untuk performa peta
                        dcc.Checklist(
                            id="peta-filter-mag",
                            options=[{"label": f" {k}", "value": k} for k in kat_mag],
                            value=["Sedang", "Kuat", "Besar", "Sangat Besar"],
                            labelStyle={"display": "block", "fontSize": "0.85rem"}
                        ),
                        html.Hr(),
                        html.Label("Kategori Kedalaman:", className="fw-semibold small"),
                        dcc.Checklist(
                            id="peta-filter-ked",
                            options=[{"label": f" {k}", "value": k} for k in kat_ked],
                            value=kat_ked,   # default: semua kedalaman diaktifkan
                            labelStyle={"display": "block", "fontSize": "0.85rem"}
                        ),
                        html.Hr(),
                        html.Label("Layer Tampilan:", className="fw-semibold small"),
                        # Pilihan layer yang dapat dikombinasikan secara bebas
                        dcc.Checklist(
                            id="peta-layer",
                            options=[
                                {"label": " Episenter Gempa",  "value": "episenter"},
                                {"label": " Patahan Aktif",    "value": "patahan"},
                                {"label": " Heatmap KDE",      "value": "kde"},
                            ],
                            value=["episenter"],   # default: hanya episenter
                            labelStyle={"display": "block", "fontSize": "0.85rem"}
                        ),
                        html.Hr(),
                        # Info teks dinamis: menampilkan jumlah titik yang terlihat
                        html.Div(id="peta-info", className="text-muted small")
                    ])
                ], className="sticky-top", style={"top": "70px"})
            ], md=2),

            # ── Peta Mapbox (lebar 10 kolom Bootstrap) ──────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(
                        dcc.Graph(
                            id="peta-map",
                            style={"height": "75vh"},
                            config={"scrollZoom": True}   # aktifkan zoom dengan scroll mouse
                        )
                    )
                ])
            ], md=10),
        ])
    ], fluid=True, className="py-3")


# ═════════════════════════════════════════════
# HALAMAN 3: ANALISIS KLUSTER
# ═════════════════════════════════════════════
def layout_kluster():
    """
    Membangun layout halaman Analisis Kluster DBSCAN dengan:
    - Slider ε (epsilon) dalam km: jarak radius pencarian tetangga
    - Slider min_samples: minimum titik untuk membentuk kluster inti
    - Tombol 'Jalankan' yang memicu callback DBSCAN secara eksplisit
    - Peta hasil kluster, metrik evaluasi, bar chart, dan scatter plot

    Penggunaan dcc.Loading membuat spinner muncul selama komputasi DBSCAN
    (yang dapat memakan waktu beberapa detik pada dataset besar).
    """
    return dbc.Container([
        html.H4("🔵 Analisis Kluster DBSCAN",
                className="my-3 fw-bold text-dark"),

        # ── Panel Kontrol Parameter DBSCAN ─────────────────────────────────
        dbc.Card([
            dbc.CardHeader("⚙️ Parameter DBSCAN (Real-time)"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("ε / Epsilon (km):", className="fw-semibold small"),
                        # eps: menentukan seberapa dekat titik harus berada agar
                        # dianggap bertetangga — makin besar → kluster makin luas
                        dcc.Slider(
                            id="kluster-eps",
                            min=50, max=300, step=25,
                            value=200,   # nilai default dari tuning skripsi
                            marks={v: f"{v}km" for v in range(50, 301, 50)},
                            tooltip={"placement": "bottom"}
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label("min_samples:", className="fw-semibold small"),
                        # min_samples: makin besar → lebih selektif, lebih banyak noise
                        dcc.Slider(
                            id="kluster-min-samples",
                            min=100, max=3000, step=100,
                            value=1000,  # nilai default dari tuning skripsi
                            marks={v: str(v) for v in [100, 500, 1000, 2000, 3000]},
                            tooltip={"placement": "bottom"}
                        )
                    ], md=5),
                    dbc.Col([
                        html.Br(),
                        # Tombol ini menjadi Input callback DBSCAN (n_clicks)
                        # sehingga komputasi hanya berjalan saat diklik, bukan
                        # otomatis setiap slider bergerak
                        dbc.Button(
                            "▶ Jalankan", id="btn-cluster",
                            color="primary", size="sm", className="mt-1"
                        )
                    ], md=1),
                ])
            ])
        ], className="mb-3"),

        # ── Loading wrapper: menampilkan spinner saat DBSCAN berjalan ───────
        dcc.Loading(
            id="loading-kluster",
            type="circle",
            children=[
                dbc.Row([
                    # Peta hasil kluster (kiri, lebar 8 kolom)
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🗺️ Peta Zona Kluster DBSCAN"),
                            dbc.CardBody(
                                dcc.Graph(id="kluster-map",
                                          style={"height": "55vh"})
                            )
                        ])
                    ], md=8),

                    # Metrik evaluasi + bar chart (kanan, lebar 4 kolom)
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("📊 Metrik Evaluasi"),
                            dbc.CardBody(html.Div(id="kluster-metrik"))
                        ], className="mb-3"),
                        dbc.Card([
                            dbc.CardHeader("📦 Jumlah Gempa per Kluster"),
                            dbc.CardBody(
                                dcc.Graph(id="kluster-bar",
                                          style={"height": "25vh"})
                            )
                        ])
                    ], md=4),
                ], className="mb-3"),

                # Scatter plot magnitudo vs kedalaman (full width)
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "⚡ Scatter Plot Magnitudo vs Kedalaman "
                                "(warna per kluster)"
                            ),
                            dbc.CardBody(
                                dcc.Graph(id="kluster-scatter",
                                          style={"height": "40vh"})
                            )
                        ])
                    ])
                ])
            ]
        )
    ], fluid=True, className="py-3")


# ═════════════════════════════════════════════
# ROUTING
# ═════════════════════════════════════════════
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def routing(pathname):
    """
    Callback navigasi multi-halaman.
    Setiap kali URL berubah, fungsi ini dipanggil dan mengembalikan
    layout halaman yang sesuai ke slot 'page-content'.
    /       → Ikhtisar
    /peta   → Peta Interaktif
    /kluster → Analisis Kluster
    """
    if pathname == "/peta":
        return layout_peta()
    elif pathname == "/kluster":
        return layout_kluster()
    return layout_ikhtisar()


# ═════════════════════════════════════════════
# CALLBACKS — HALAMAN 1: IKHTISAR
# ═════════════════════════════════════════════
@app.callback(
    Output("chart-tren",      "figure"),   # grafik tren tahunan
    Output("chart-pie",       "figure"),   # pie chart magnitudo
    Output("tabel-provinsi",  "children"), # tabel 5 provinsi teratas
    Input("slider-tahun",     "value")     # dipicu setiap slider bergerak
)
def update_ikhtisar(tahun_range):
    """
    Callback reaktif Halaman Ikhtisar.
    Input: tahun_range = [tahun_awal, tahun_akhir] dari RangeSlider.
    Setiap perubahan slider memfilter df_gempa dan memperbarui ketiga output.
    """
    # ┌─────────────────────────────────────────────────────────────┐
    # │  HALAMAN 1: IKHTISAR — Sumber Tabel                        │
    # │  KPI Cards   : FACT_GEMPA + DIM_LOKASI + DIM_KEDALAMAN     │
    # │  Tren Tahunan: FACT_GEMPA + DIM_WAKTU                      │
    # │  Pie Chart   : FACT_GEMPA + DIM_MAGNITUDE                  │
    # │  Tabel Prov  : FACT_GEMPA + DIM_LOKASI                     │
    # └─────────────────────────────────────────────────────────────┘
    # Filter baris berdasarkan rentang tahun yang dipilih
    df = df_gempa[
        (df_gempa["tahun"] >= tahun_range[0]) &
        (df_gempa["tahun"] <= tahun_range[1])
    ]

    # ── Tren tahunan: hitung jumlah kejadian per tahun ───────────────────
    tren = df.groupby("tahun").size().reset_index(name="jumlah")
    fig_tren = px.bar(
        tren, x="tahun", y="jumlah",
        color_discrete_sequence=["#0072B2"],
        labels={"tahun": "Tahun", "jumlah": "Jumlah Kejadian"},
    )
    fig_tren.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white"
    )

    # ── Pie chart magnitudo: proporsi setiap kategori ────────────────────
    mag_dist = df["kategori_magnitude"].value_counts().reset_index()
    mag_dist.columns = ["kategori", "jumlah"]
    # Urutkan sesuai skala kekuatan gempa (dari terkecil ke terbesar)
    urutan = ["Mikro","Minor","Ringan","Sedang","Kuat","Besar","Sangat Besar"]
    mag_dist["urutan"] = mag_dist["kategori"].map(
        {k: i for i, k in enumerate(urutan)}
    )
    mag_dist = mag_dist.sort_values("urutan")
    fig_pie = px.pie(
        mag_dist, names="kategori", values="jumlah",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4   # hole=0.4 membuat donut chart (bukan pie penuh)
    )
    fig_pie.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(font=dict(size=10)),
        paper_bgcolor="white"
    )

    # ── Tabel 5 provinsi: value_counts lalu ambil 5 teratas ──────────────
    prov = df["provinsi"].value_counts().head(5).reset_index()
    prov.columns = ["Provinsi", "Jumlah Gempa"]
    prov["Proporsi (%)"] = (prov["Jumlah Gempa"] / len(df) * 100).round(2)
    tabel = dbc.Table.from_dataframe(
        prov, striped=True, bordered=True,
        hover=True, responsive=True, size="sm"
    )
    return fig_tren, fig_pie, tabel


# ═════════════════════════════════════════════
# CALLBACKS — HALAMAN 2: PETA INTERAKTIF
# ═════════════════════════════════════════════
@app.callback(
    Output("peta-map",  "figure"),   # objek peta Mapbox
    Output("peta-info", "children"), # teks info jumlah titik
    Input("peta-slider-tahun", "value"),
    Input("peta-filter-mag",   "value"),
    Input("peta-filter-ked",   "value"),
    Input("peta-layer",        "value"),
)
def update_peta(tahun_range, filter_mag, filter_ked, layers):
    """
    Callback reaktif Halaman Peta.
    Dipicu oleh perubahan salah satu dari empat filter.
    Menggabungkan hingga tiga layer pada satu objek go.Figure:
      1. KDE Densitas (Densitymapbox) — heatmap intensitas
      2. Episenter (Scattermapbox per grup kedalaman) — titik berlapis
      3. Patahan Aktif (Scattermapbox mode="lines") — garis patahan
    """
    # ┌─────────────────────────────────────────────────────────────┐
    # │  HALAMAN 2: PETA INTERAKTIF — Sumber Tabel                 │
    # │  Episenter   : FACT_GEMPA + DIM_LOKASI + DIM_KEDALAMAN     │
    # │  Hover popup : FACT_GEMPA + DIM_WAKTU + DIM_LOKASI         │
    # │  KDE Heatmap : FACT_GEMPA + DIM_LOKASI                     │
    # │  Patahan Aktif: GeoJSON PSG/ESDM (bukan dari PostgreSQL)   │
    # └─────────────────────────────────────────────────────────────┘
    layers = layers or []

    # Terapkan semua filter sekaligus menggunakan boolean indexing Pandas
    df = df_gempa[
        (df_gempa["tahun"] >= tahun_range[0]) &
        (df_gempa["tahun"] <= tahun_range[1]) &
        (df_gempa["kategori_magnitude"].isin(filter_mag or [])) &
        (df_gempa["kategori_kedalaman"].isin(filter_ked or []))
    ]

    # Sampling maks 10.000 titik untuk performa render peta di browser
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)

    fig = go.Figure()

    # ── Layer 1: KDE (Kernel Density Estimation / Heatmap Densitas) ──────
    # Menampilkan konsentrasi episenter sebagai gradasi warna merah
    if "kde" in layers and len(df) > 0:
        fig.add_trace(go.Densitymapbox(
            lat=df["lat"], lon=df["lon"],
            z=df["nilai_magnitude"],   # intensitas warna berdasarkan magnitudo
            radius=15, opacity=0.6,
            colorscale="Reds",
            name="KDE Densitas",
            showscale=True,
            colorbar=dict(
                orientation="h",
                title=dict(text="Densitas Magnitudo", side="top", font=dict(size=11)),
                thickness=12,
                len=0.25,
                x=0.03,
                xanchor="left",
                y=0.03,
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="gray",
                borderwidth=1,
                tickfont=dict(size=10),
            )
        ))

    # ── Layer 2: Episenter — diplot per grup kedalaman ───────────────────
    # Setiap grup (Dangkal/Menengah/Dalam) menjadi trace terpisah
    # sehingga muncul di legenda dan bisa dimatikan secara individual
    if "episenter" in layers and len(df) > 0:
        # Palet cerah & kontras tinggi, mengikuti gradien kedalaman
        # (dangkal = merah/panas → dalam = ungu/dingin) agar mudah dibedakan
        # di atas peta dasar yang terang.
        warna_ked = {"Dangkal": "#FF1744", "Menengah": "#FF9100", "Dalam": "#6200EA"}
        for ked, grp in df.groupby("kategori_kedalaman"):
            fig.add_trace(go.Scattermapbox(
                lat=grp["lat"], lon=grp["lon"],
                mode="markers",
                marker=dict(
                    # Ukuran marker proporsional dengan magnitudo (clip 2–9)
                    size=grp["nilai_magnitude"].clip(2, 9) * 1.5,
                    color=warna_ked.get(ked, "gray"),
                    opacity=0.85
                ),
                name=f"Kedalaman: {ked}",
                customdata=grp[["tgl","ot","nilai_magnitude",
                                "nilai_kedalaman","remark"]].values,
                # hovertemplate: format tooltip saat kursor hover di titik
                hovertemplate=(
                    "<b>%{customdata[4]}</b><br>"
                    "Tanggal: %{customdata[0]} %{customdata[1]}<br>"
                    "Magnitudo: %{customdata[2]} M<br>"
                    "Kedalaman: %{customdata[3]} km<br>"
                    "Lat/Lon: %{lat:.3f}, %{lon:.3f}<extra></extra>"
                )
            ))

    # ── Layer 3: Patahan Aktif — iterasi setiap fitur GeoJSON ────────────
    if "patahan" in layers:
        lats_all, lons_all, names_all = [], [], []

        for _, row in gdf_aktif.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            nama = str(row.get("namobj", "Patahan Aktif"))

            # Konversi geometri ke daftar koordinat (lat/lon terpisah)
            # Plotly memerlukan None sebagai pemisah antar segmen garis
            if geom.geom_type == "LineString":
                segmen_list = [list(geom.coords)]
            elif geom.geom_type == "MultiLineString":
                segmen_list = [list(seg.coords) for seg in geom.geoms]
            else:
                continue  # lewati tipe geometri lain (Point, Polygon, dst.)

            for segmen in segmen_list:
                lats_all  += [c[1] for c in segmen] + [None]   # None = angkat pena
                lons_all  += [c[0] for c in segmen] + [None]
                names_all += [nama] * len(segmen)   + [None]

        if lats_all:
            fig.add_trace(go.Scattermapbox(
                lat=lats_all,
                lon=lons_all,
                mode="lines",
                line=dict(width=1.5, color="#D55E00"),
                name="Patahan Aktif",
                hoverinfo="text",
                text=names_all,
                showlegend=True,
                opacity=0.8
            ))

    # Pengaturan tampilan peta: pusat Indonesia, zoom level 4
    fig.update_layout(
        mapbox=dict(style=MAPBOX_STYLE, center=dict(lat=-2.5, lon=118), zoom=4),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray", borderwidth=1,
            x=0.01, y=0.99, font=dict(size=11)
        ),
        paper_bgcolor="white"
    )

    info = f"Menampilkan {len(df):,} titik dari {tahun_range[0]}–{tahun_range[1]}"
    return fig, info


# ═════════════════════════════════════════════
# CALLBACKS — HALAMAN 3: ANALISIS KLUSTER
# ═════════════════════════════════════════════
@app.callback(
    Output("kluster-map",     "figure"),   # peta sebaran kluster
    Output("kluster-metrik",  "children"), # kartu metrik evaluasi
    Output("kluster-bar",     "figure"),   # bar chart jumlah per kluster
    Output("kluster-scatter", "figure"),   # scatter magnitudo vs kedalaman
    Input("btn-cluster",      "n_clicks"), # tombol 'Jalankan' sebagai trigger
    State("kluster-eps",          "value"),          # nilai ε (km)
    State("kluster-min-samples",  "value"),          # nilai min_samples
    prevent_initial_call=False   # tetap jalankan saat halaman pertama dibuka
)
def update_kluster(n_clicks, eps_km, min_samp):
    """
    Callback utama Halaman Analisis Kluster.
    Alur komputasi:
      1. Konversi koordinat WGS84 → UTM Zone 50S (EPSG:32750) dalam meter
         agar jarak Euclidean setara dengan jarak geografis sesungguhnya
      2. Jalankan DBSCAN dengan parameter dari slider
      3. Hitung metrik: Silhouette Score & Davies-Bouldin Index pada sampel 10k
      4. Buat empat visualisasi output

    Catatan EPSG:32750 — UTM Zone 50S dipilih karena mencakup sebagian besar
    kepulauan Indonesia (95°E–105°E), meminimalkan distorsi proyeksi.
    """
    # ┌─────────────────────────────────────────────────────────────┐
    # │  HALAMAN 3: ANALISIS KLUSTER — Sumber Tabel                │
    # │  Peta Kluster : FACT_GEMPA + DIM_LOKASI + DIM_KLUSTER      │
    # │  Metrik Eval  : Dihitung real-time dari FACT_GEMPA +        │
    # │                 DIM_LOKASI (proyeksi EPSG:32750)            │
    # │  Bar Chart    : FACT_GEMPA + DIM_KLUSTER                   │
    # │  Scatter Plot : FACT_GEMPA + DIM_MAGNITUDE + DIM_KEDALAMAN │
    # │                 + DIM_KLUSTER                               │
    # └─────────────────────────────────────────────────────────────┘
    # Konversi epsilon dari km ke meter (satuan DBSCAN)
    eps_m = (eps_km or 200) * 1000

    # ── Langkah 1: Proyeksi koordinat ke sistem metrik ──────────────────
    import geopandas as gpd
    gdf_pts = gpd.GeoDataFrame(
        df_gempa[["lat","lon"]].copy(),
        geometry=gpd.points_from_xy(df_gempa["lon"], df_gempa["lat"]),
        crs="EPSG:4326"        # koordinat geografis derajat (WGS84)
    ).to_crs("EPSG:32750")     # proyeksikan ke meter (UTM zona Indonesia)

    # X adalah matriks (N × 2) koordinat dalam meter yang menjadi input DBSCAN
    X = np.column_stack([gdf_pts.geometry.x, gdf_pts.geometry.y])

    # ── Langkah 2: Jalankan DBSCAN ──────────────────────────────────────
    # algorithm="ball_tree": struktur data pohon bola — efisien untuk data spasial
    # n_jobs=-1: gunakan semua core CPU secara paralel
    db     = DBSCAN(eps=eps_m, min_samples=min_samp or 1000,
                    algorithm="ball_tree", n_jobs=-1)
    labels = db.fit_predict(X)  # label -1 berarti noise (tidak masuk kluster mana pun)

    # Hitung statistik dasar hasil kluster
    n_kluster = len(set(labels)) - (1 if -1 in labels else 0)  # kecualikan noise
    n_noise   = int(np.sum(labels == -1))
    pct_noise = n_noise / len(labels) * 100

    # ── Langkah 3: Hitung metrik evaluasi pada sampel acak 10k ──────────
    # Metrik dihitung pada sampel agar tidak kehabisan memori/waktu
    np.random.seed(42)
    idx_eval = np.random.choice(len(X), min(10000, len(X)), replace=False)
    if len(set(labels[idx_eval])) >= 2:
        # Silhouette Score: -1 (buruk) → 1 (sempurna), target ≥ 0.40
        sil = silhouette_score(X[idx_eval], labels[idx_eval])
        # Davies-Bouldin Index: makin kecil makin baik, target ≤ 3.0
        dbi = davies_bouldin_score(X[idx_eval], labels[idx_eval])
    else:
        sil, dbi = 0.0, 0.0   # hanya 1 kluster → metrik tidak bermakna

    # Tambahkan kolom label ke salinan dataframe untuk keperluan visualisasi
    df_plot = df_gempa.copy()
    df_plot["label"] = labels
    df_plot["kluster_str"] = df_plot["label"].apply(
        lambda x: "Noise" if x == -1 else f"Kluster {x}"
    )

    # ── Langkah 4a: Peta Kluster ─────────────────────────────────────────
    # Buat pemetaan warna: setiap kluster mendapat warna dari WARNA_KLUSTER,
    # noise selalu abu-abu
    warna_map = {}
    unik = sorted([l for l in df_plot["label"].unique() if l >= 0])
    for i, l in enumerate(unik):
        warna_map[f"Kluster {l}"] = WARNA_KLUSTER[i % len(WARNA_KLUSTER)]
    warna_map["Noise"] = "#cccccc"

    # Sampel untuk performa peta — seluruh data terlalu berat untuk browser
    df_sample = df_plot.sample(min(10000, len(df_plot)), random_state=42)
    fig_map = px.scatter_mapbox(
        df_sample,
        lat="lat", lon="lon",
        color="kluster_str",
        color_discrete_map=warna_map,
        size_max=8,
        zoom=4, center=dict(lat=-2.5, lon=118),
        mapbox_style=MAPBOX_STYLE,
        hover_data={"lat": True, "lon": True,
                    "nilai_magnitude": True, "nilai_kedalaman": True},
        labels={"kluster_str": "Kluster"},
    )
    fig_map.update_traces(marker=dict(size=5, opacity=0.7))
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(size=11))
    )

    # ── Langkah 4b: Kartu Metrik Evaluasi ───────────────────────────────
    # Indikator ✅/⚠️ ditentukan berdasarkan ambang batas dari literatur
    target_sil = sil >= 0.40   # silhouette ≥ 0.40 → pemisahan kluster baik
    target_dbi = dbi <= 3.0    # DBI ≤ 3.0 → kluster kompak & terpisah baik

    metrik = dbc.ListGroup([
        dbc.ListGroupItem([
            html.Span("Jumlah Kluster", className="fw-semibold"),
            dbc.Badge(str(n_kluster), color="primary", className="float-end")
        ]),
        dbc.ListGroupItem([
            html.Span("Noise (-1)", className="fw-semibold"),
            dbc.Badge(f"{n_noise:,} ({pct_noise:.1f}%)",
                      color="secondary", className="float-end")
        ]),
        dbc.ListGroupItem([
            html.Span("Silhouette Score", className="fw-semibold"),
            dbc.Badge(
                f"{sil:.4f} {'✅' if target_sil else '⚠️'}",
                color="success" if target_sil else "warning",
                className="float-end"
            )
        ]),
        dbc.ListGroupItem([
            html.Span("Davies-Bouldin Idx", className="fw-semibold"),
            dbc.Badge(
                f"{dbi:.4f} {'✅' if target_dbi else '⚠️'}",
                color="success" if target_dbi else "warning",
                className="float-end"
            )
        ]),
        dbc.ListGroupItem([
            html.Span("ε (Epsilon)", className="fw-semibold"),
            dbc.Badge(f"{eps_km} km", color="info", className="float-end")
        ]),
        dbc.ListGroupItem([
            html.Span("min_samples", className="fw-semibold"),
            dbc.Badge(str(min_samp), color="info", className="float-end")
        ]),
    ], flush=True, className="small")

    # ── Langkah 4c: Bar Chart jumlah gempa per kluster (tanpa noise) ─────
    cnt = df_plot[df_plot["label"] >= 0]["kluster_str"].value_counts().reset_index()
    cnt.columns = ["Kluster", "Jumlah"]
    fig_bar = px.bar(
        cnt, x="Kluster", y="Jumlah",
        color="Kluster",
        color_discrete_sequence=WARNA_KLUSTER,
        labels={"Jumlah": "N Gempa"}
    )
    fig_bar.update_layout(
        showlegend=False,
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="white", paper_bgcolor="white"
    )

    # ── Langkah 4d: Scatter Plot Magnitudo vs Kedalaman ──────────────────
    # autorange="reversed" pada sumbu Y agar kedalaman 0 km di atas
    # (konvensi geologi: semakin dalam → semakin ke bawah di grafik)
    df_scatter = df_plot[df_plot["label"] >= 0].sample(
        min(5000, len(df_plot)), random_state=42
    )
    fig_scatter = px.scatter(
        df_scatter,
        x="nilai_magnitude", y="nilai_kedalaman",
        color="kluster_str",
        color_discrete_map=warna_map,
        opacity=0.5,
        labels={
            "nilai_magnitude": "Magnitudo (M)",
            "nilai_kedalaman": "Kedalaman (km)",
            "kluster_str": "Kluster"
        }
    )
    fig_scatter.update_yaxes(autorange="reversed")  # kedalaman tumbuh ke bawah
    fig_scatter.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(font=dict(size=10))
    )

    return fig_map, metrik, fig_bar, fig_scatter


# ─────────────────────────────────────────────
# JALANKAN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Blok ini hanya berjalan ketika file dieksekusi langsung (bukan di-import).
    # debug=False: matikan debug mode untuk tampilan yang bersih saat demo.
    # host="0.0.0.0": dengarkan semua interface (aksesibel dari jaringan lokal).
    # port=8050: port default Dash.
    print("\n" + "="*55)
    print("  Dashboard Visual Analytics Seismisitas Indonesia")
    print("  Sri Bintang Pratomo (825220070) — UNTAR Prodi SI")
    print("="*55)
    print("  Buka browser: http://127.0.0.1:8050")
    print("  Tekan Ctrl+C untuk menghentikan server")
    print("="*55 + "\n")
    # port: pakai env var PORT bila ada (kondisi deploy), default 8050 untuk lokal.
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
