"""
Dashboard Visual Analytics Seismisitas Indonesia
=================================================
Skripsi: Perancangan Sistem Visual Analytics untuk Analisis Seismisitas
         Indonesia dengan Metode Clustering Spasial
Peneliti: Sri Bintang Pratomo (825220070) — UNTAR Prodi SI

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

import pandas as pd
import geopandas as gpd
import numpy as np
import json
import psycopg2
from sqlalchemy import create_engine

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host"    : "db.caqlmbjaqfgrqbjznudn.supabase.co",
    "port"    : 5432,
    "dbname"  : "postgres",
    "user"    : "postgres",
    "password": "Seismisitas2026",
}
GEOJSON_PATH = "Patahan Aktif geoportal.esdm.go.id.geojson"
MAPBOX_STYLE = "open-street-map"

# Parameter DBSCAN final dari skripsi
EPS_DEFAULT        = 200_000   # 200 km dalam meter
MIN_SAMPLES_DEFAULT = 1000

# Warna kluster (color-blind safe palette)
WARNA_KLUSTER = [
    "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
]

# ─────────────────────────────────────────────
# KONEKSI DATABASE
# ─────────────────────────────────────────────
def get_engine():
    cfg = DB_CONFIG
    return create_engine(
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )

# ─────────────────────────────────────────────
# LOAD DATA (cache di memori)
# ─────────────────────────────────────────────
print("Memuat data dari PostgreSQL...")
engine = get_engine()

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

df_kluster = pd.read_sql(
    "SELECT * FROM dim_kluster ORDER BY id_kluster", engine
)

# Load GeoJSON patahan aktif
gdf_patahan = gpd.read_file(GEOJSON_PATH)
gdf_aktif   = gdf_patahan[gdf_patahan["klspthn"] == "Aktif"].copy()

print(f"  Data gempa   : {len(df_gempa):,} baris")
print(f"  Kluster      : {len(df_kluster)} kluster")
print(f"  Patahan aktif: {len(gdf_aktif)} segmen")

# ─────────────────────────────────────────────
# INISIALISASI APP
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Visual Analytics Seismisitas Indonesia"
)
server = app.server

# ─────────────────────────────────────────────
# KOMPONEN NAVBAR
# ─────────────────────────────────────────────
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
    tahun_min = int(df_gempa["tahun"].min())
    tahun_max = int(df_gempa["tahun"].max())

    # Hitung KPI
    total        = len(df_gempa)
    rata_mag     = df_gempa["nilai_magnitude"].mean()
    prov_aktif   = df_gempa["provinsi"].value_counts().index[0]
    pct_dangkal  = (df_gempa["kategori_kedalaman"] == "Dangkal").mean() * 100

    return dbc.Container([
        html.H4("📊 Ikhtisar Seismisitas Indonesia",
                className="my-3 fw-bold text-dark"),

        # ── Kartu KPI ──────────────────────────────────────
        dbc.Row([
            dbc.Col(_kpi_card("Total Kejadian",      f"{total:,}",          "gempa (2000–2026)", "primary"), md=3),
            dbc.Col(_kpi_card("Rata-rata Magnitudo", f"{rata_mag:.2f} M",   "skala Richter",     "warning"), md=3),
            dbc.Col(_kpi_card("Provinsi Paling Aktif", prov_aktif,          "kejadian terbanyak","danger"),  md=3),
            dbc.Col(_kpi_card("Gempa Dangkal",       f"{pct_dangkal:.1f}%", "kedalaman 0–70 km", "success"), md=3),
        ], className="mb-4"),

        # ── Filter Tahun ───────────────────────────────────
        dbc.Card([
            dbc.CardBody([
                html.Label("Filter Rentang Tahun:", className="fw-semibold"),
                dcc.RangeSlider(
                    id="slider-tahun",
                    min=tahun_min, max=tahun_max,
                    value=[tahun_min, tahun_max],
                    marks={y: str(y) for y in range(tahun_min, tahun_max+1, 5)},
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ])
        ], className="mb-4"),

        # ── Grafik Tren + Pie Chart ────────────────────────
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

        # ── Tabel 5 Provinsi Teratas ───────────────────────
        dbc.Card([
            dbc.CardHeader("🏆 Lima Provinsi dengan Kejadian Gempa Terbanyak"),
            dbc.CardBody(html.Div(id="tabel-provinsi"))
        ]),

    ], fluid=True, className="py-3")


def _kpi_card(judul, nilai, sub, warna):
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
    tahun_min = int(df_gempa["tahun"].min())
    tahun_max = int(df_gempa["tahun"].max())

    kat_mag = sorted(df_gempa["kategori_magnitude"].unique())
    kat_ked = sorted(df_gempa["kategori_kedalaman"].unique())

    return dbc.Container([
        html.H4("🗺️ Peta Interaktif Episenter Gempa",
                className="my-3 fw-bold text-dark"),

        dbc.Row([
            # ── Panel Filter ──────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("⚙️ Filter Data"),
                    dbc.CardBody([
                        html.Label("Rentang Tahun:", className="fw-semibold small"),
                        dcc.RangeSlider(
                            id="peta-slider-tahun",
                            min=tahun_min, max=tahun_max,
                            value=[2015, tahun_max],
                            marks={y: str(y) for y in range(tahun_min, tahun_max+1, 5)},
                            tooltip={"placement": "bottom"}
                        ),
                        html.Hr(),
                        html.Label("Kategori Magnitudo:", className="fw-semibold small"),
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
                            value=kat_ked,
                            labelStyle={"display": "block", "fontSize": "0.85rem"}
                        ),
                        html.Hr(),
                        html.Label("Layer Tampilan:", className="fw-semibold small"),
                        dcc.Checklist(
                            id="peta-layer",
                            options=[
                                {"label": " Episenter Gempa",  "value": "episenter"},
                                {"label": " Patahan Aktif",    "value": "patahan"},
                                {"label": " Heatmap KDE",      "value": "kde"},
                            ],
                            value=["episenter"],
                            labelStyle={"display": "block", "fontSize": "0.85rem"}
                        ),
                        html.Hr(),
                        html.Div(id="peta-info", className="text-muted small")
                    ])
                ], className="sticky-top", style={"top": "70px"})
            ], md=2),

            # ── Peta ──────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(
                        dcc.Graph(
                            id="peta-map",
                            style={"height": "75vh"},
                            config={"scrollZoom": True}
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
    return dbc.Container([
        html.H4("🔵 Analisis Kluster DBSCAN",
                className="my-3 fw-bold text-dark"),

        # ── Panel Kontrol Parameter ────────────────────────
        dbc.Card([
            dbc.CardHeader("⚙️ Parameter DBSCAN (Real-time)"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("ε / Epsilon (km):", className="fw-semibold small"),
                        dcc.Slider(
                            id="kluster-eps",
                            min=50, max=300, step=25,
                            value=200,
                            marks={v: f"{v}km" for v in range(50, 301, 50)},
                            tooltip={"placement": "bottom"}
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label("min_samples:", className="fw-semibold small"),
                        dcc.Slider(
                            id="kluster-min-samples",
                            min=100, max=3000, step=100,
                            value=1000,
                            marks={v: str(v) for v in [100, 500, 1000, 2000, 3000]},
                            tooltip={"placement": "bottom"}
                        )
                    ], md=5),
                    dbc.Col([
                        html.Br(),
                        dbc.Button(
                            "▶ Jalankan", id="btn-cluster",
                            color="primary", size="sm", className="mt-1"
                        )
                    ], md=1),
                ])
            ])
        ], className="mb-3"),

        # ── Loading wrapper ────────────────────────────────
        dcc.Loading(
            id="loading-kluster",
            type="circle",
            children=[
                dbc.Row([
                    # Peta Kluster
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🗺️ Peta Zona Kluster DBSCAN"),
                            dbc.CardBody(
                                dcc.Graph(id="kluster-map",
                                          style={"height": "55vh"})
                            )
                        ])
                    ], md=8),

                    # Metrik Evaluasi
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
    if pathname == "/peta":
        return layout_peta()
    elif pathname == "/kluster":
        return layout_kluster()
    return layout_ikhtisar()


# ═════════════════════════════════════════════
# CALLBACKS — HALAMAN 1: IKHTISAR
# ═════════════════════════════════════════════
@app.callback(
    Output("chart-tren",      "figure"),
    Output("chart-pie",       "figure"),
    Output("tabel-provinsi",  "children"),
    Input("slider-tahun",     "value")
)
def update_ikhtisar(tahun_range):
    df = df_gempa[
        (df_gempa["tahun"] >= tahun_range[0]) &
        (df_gempa["tahun"] <= tahun_range[1])
    ]

    # Tren tahunan
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

    # Pie chart magnitudo
    mag_dist = df["kategori_magnitude"].value_counts().reset_index()
    mag_dist.columns = ["kategori", "jumlah"]
    urutan = ["Mikro","Minor","Ringan","Sedang","Kuat","Besar","Sangat Besar"]
    mag_dist["urutan"] = mag_dist["kategori"].map(
        {k: i for i, k in enumerate(urutan)}
    )
    mag_dist = mag_dist.sort_values("urutan")
    fig_pie = px.pie(
        mag_dist, names="kategori", values="jumlah",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4
    )
    fig_pie.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(font=dict(size=10)),
        paper_bgcolor="white"
    )

    # Tabel provinsi
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
    Output("peta-map",  "figure"),
    Output("peta-info", "children"),
    Input("peta-slider-tahun", "value"),
    Input("peta-filter-mag",   "value"),
    Input("peta-filter-ked",   "value"),
    Input("peta-layer",        "value"),
)
def update_peta(tahun_range, filter_mag, filter_ked, layers):
    layers = layers or []

    df = df_gempa[
        (df_gempa["tahun"] >= tahun_range[0]) &
        (df_gempa["tahun"] <= tahun_range[1]) &
        (df_gempa["kategori_magnitude"].isin(filter_mag or [])) &
        (df_gempa["kategori_kedalaman"].isin(filter_ked or []))
    ]

    # Sampling maks 10k titik untuk performa
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)

    fig = go.Figure()

    # Layer KDE (heatmap densitas)
    if "kde" in layers and len(df) > 0:
        fig.add_trace(go.Densitymapbox(
            lat=df["lat"], lon=df["lon"],
            z=df["nilai_magnitude"],
            radius=15, opacity=0.6,
            colorscale="Reds",
            name="KDE Densitas",
            showscale=False
        ))

    # Layer Episenter
    if "episenter" in layers and len(df) > 0:
        warna_ked = {"Dangkal": "#E69F00", "Menengah": "#56B4E9", "Dalam": "#009E73"}
        for ked, grp in df.groupby("kategori_kedalaman"):
            fig.add_trace(go.Scattermapbox(
                lat=grp["lat"], lon=grp["lon"],
                mode="markers",
                marker=dict(
                    size=grp["nilai_magnitude"].clip(2, 9) * 1.5,
                    color=warna_ked.get(ked, "gray"),
                    opacity=0.7
                ),
                name=f"Kedalaman: {ked}",
                customdata=grp[["tgl","ot","nilai_magnitude",
                                "nilai_kedalaman","remark"]].values,
                hovertemplate=(
                    "<b>%{customdata[4]}</b><br>"
                    "Tanggal: %{customdata[0]} %{customdata[1]}<br>"
                    "Magnitudo: %{customdata[2]} M<br>"
                    "Kedalaman: %{customdata[3]} km<br>"
                    "Lat/Lon: %{lat:.3f}, %{lon:.3f}<extra></extra>"
                )
            ))

    # Layer Patahan Aktif
    if "patahan" in layers:
        lats_all, lons_all, names_all = [], [], []

        for _, row in gdf_aktif.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            nama = str(row.get("namobj", "Patahan Aktif"))

            # Support LineString dan MultiLineString
            if geom.geom_type == "LineString":
                segmen_list = [list(geom.coords)]
            elif geom.geom_type == "MultiLineString":
                segmen_list = [list(seg.coords) for seg in geom.geoms]
            else:
                continue

            for segmen in segmen_list:
                lats_all  += [c[1] for c in segmen] + [None]
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
    Output("kluster-map",     "figure"),
    Output("kluster-metrik",  "children"),
    Output("kluster-bar",     "figure"),
    Output("kluster-scatter", "figure"),
    Input("btn-cluster",      "n_clicks"),
    State("kluster-eps",          "value"),
    State("kluster-min-samples",  "value"),
    prevent_initial_call=False
)
def update_kluster(n_clicks, eps_km, min_samp):
    eps_m = (eps_km or 200) * 1000

    # Proyeksi ke meter untuk DBSCAN
    import geopandas as gpd
    gdf_pts = gpd.GeoDataFrame(
        df_gempa[["lat","lon"]].copy(),
        geometry=gpd.points_from_xy(df_gempa["lon"], df_gempa["lat"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:32750")

    X = np.column_stack([gdf_pts.geometry.x, gdf_pts.geometry.y])

    # Jalankan DBSCAN
    db     = DBSCAN(eps=eps_m, min_samples=min_samp or 1000,
                    algorithm="ball_tree", n_jobs=-1)
    labels = db.fit_predict(X)

    n_kluster = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise   = int(np.sum(labels == -1))
    pct_noise = n_noise / len(labels) * 100

    # Hitung metrik pada sampel
    np.random.seed(42)
    idx_eval = np.random.choice(len(X), min(10000, len(X)), replace=False)
    if len(set(labels[idx_eval])) >= 2:
        sil = silhouette_score(X[idx_eval], labels[idx_eval])
        dbi = davies_bouldin_score(X[idx_eval], labels[idx_eval])
    else:
        sil, dbi = 0.0, 0.0

    df_plot = df_gempa.copy()
    df_plot["label"] = labels
    df_plot["kluster_str"] = df_plot["label"].apply(
        lambda x: "Noise" if x == -1 else f"Kluster {x}"
    )

    # ── Peta Kluster ──────────────────────────────
    warna_map = {}
    unik = sorted([l for l in df_plot["label"].unique() if l >= 0])
    for i, l in enumerate(unik):
        warna_map[f"Kluster {l}"] = WARNA_KLUSTER[i % len(WARNA_KLUSTER)]
    warna_map["Noise"] = "#cccccc"

    # Sampel untuk performa peta
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

    # ── Kartu Metrik ──────────────────────────────
    target_sil = sil >= 0.40
    target_dbi = dbi <= 3.0

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

    # ── Bar Chart per Kluster ──────────────────────
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

    # ── Scatter Mag vs Depth ───────────────────────
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
    fig_scatter.update_yaxes(autorange="reversed")
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
    print("\n" + "="*55)
    print("  Dashboard Visual Analytics Seismisitas Indonesia")
    print("  Sri Bintang Pratomo (825220070) — UNTAR Prodi SI")
    print("="*55)
    print("  Buka browser: http://127.0.0.1:8050")
    print("  Tekan Ctrl+C untuk menghentikan server")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
