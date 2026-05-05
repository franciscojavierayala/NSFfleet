"""
app.py — Interfaz Streamlit para cNSFfleet Predictor
Ejecutar con: streamlit run app.py
"""

import sys
import math
import time
from pathlib import Path
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import torch
import numpy as np
import folium
from streamlit_folium import st_folium
import pandas as pd
import requests as _requests
from streamlit_searchbox import st_searchbox

from model.nflow_model import ConditionalFlowModel
from inference.predictor import FleetPredictor
from route.route_builder import build_route_context
from data.synthetic import MINS, MAXS

# ── Ciudades frecuentes ───────────────────────────────────────────────────────
CITIES_FALLBACK = [
    "Madrid, España", "Barcelona, España", "Valencia, España",
    "Sevilla, España", "Zaragoza, España", "Málaga, España",
    "Murcia, España", "Palma, España", "Las Palmas, España",
    "Bilbao, España", "Alicante, España", "Córdoba, España",
    "Valladolid, España", "Vigo, España", "Gijón, España",
    "Granada, España", "Pamplona, España", "Santander, España",
    "San Sebastián, España", "Toledo, España", "Burgos, España",
    "Salamanca, España", "Logroño, España", "Albacete, España",
    "Lisboa, Portugal", "Oporto, Portugal",
    "Lyon, Francia", "Burdeos, Francia", "París, Francia",
    "Milán, Italia", "Roma, Italia",
]

_last_nominatim_call: float = 0.0
_NOMINATIM_MIN_INTERVAL = 1.1

N_SAMPLES = 1000

# ── Paleta de colores centralizada ───────────────────────────────────────────
AMBER   = "#f5a623"
BLUE    = "#4f8ef7"
TEAL    = "#2dd4bf"
VIOLET  = "#a78bfa"
ROSE    = "#fb7185"
ORANGE  = "#fb923c"
GREEN   = "#4ade80"

# Colores para tramos del mapa (bien diferenciados)
SEG_COLORS = [AMBER, BLUE, TEAL, VIOLET, ROSE, ORANGE, GREEN, "#e879f9"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def haversine(p1, p2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [p1[0], p1[1], p2[0], p2[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def segment_polyline_by_distance(polyline, n_segs):
    if len(polyline) < 2 or n_segs < 1:
        return [polyline]
    cumulative = [0.0]
    for i in range(1, len(polyline)):
        cumulative.append(cumulative[-1] + haversine(polyline[i-1], polyline[i]))
    total_dist = cumulative[-1]
    if total_dist == 0:
        return [polyline]
    seg_dist = total_dist / n_segs
    segments = []
    for i in range(n_segs):
        target_start = i * seg_dist
        target_end   = (i + 1) * seg_dist
        start_idx = next((j for j, d in enumerate(cumulative) if d >= target_start), 0)
        end_idx   = next((j for j, d in enumerate(cumulative) if d >= target_end), len(polyline)-1)
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, len(polyline)-1)
        segments.append(polyline[start_idx:end_idx+1])
    return segments

def frontal_wind(wind_speed, wind_dir, route_bearing):
    angle_diff = math.radians(abs(route_bearing - wind_dir) % 360)
    if angle_diff > math.pi:
        angle_diff = 2 * math.pi - angle_diff
    return wind_speed * math.cos(angle_diff)

@st.cache_data(ttl=3_600, show_spinner=False)
def _fetch_nominatim(query):
    global _last_nominatim_call
    elapsed = time.monotonic() - _last_nominatim_call
    if elapsed < _NOMINATIM_MIN_INTERVAL:
        time.sleep(_NOMINATIM_MIN_INTERVAL - elapsed)
    resp = _requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 7, "addressdetails": 1, "accept-language": "es"},
        headers={"User-Agent": "cnsffleet_demo/1.0"},
        timeout=4,
    )
    resp.raise_for_status()
    _last_nominatim_call = time.monotonic()
    return resp.json()

def search_cities(query):
    if not query or len(query) < 2:
        return []
    q_lower = query.lower()
    local_matches = [c for c in CITIES_FALLBACK if q_lower in c.lower()]
    try:
        results = _fetch_nominatim(query)
        seen = set(local_matches)
        suggestions = list(local_matches)
        for r in results:
            addr = r.get("address", {})
            city = (addr.get("city") or addr.get("town") or addr.get("village")
                    or addr.get("municipality") or r.get("display_name","").split(",")[0].strip())
            country = addr.get("country", "")
            short = f"{city}, {country}".strip(", ")
            if short and short not in seen:
                seen.add(short)
                suggestions.append(short)
        return suggestions[:8]
    except _requests.exceptions.Timeout:
        st.toast("⏱️ Nominatim tardó demasiado — mostrando ciudades frecuentes.", icon="⚠️")
        return local_matches[:8]
    except _requests.exceptions.ConnectionError:
        st.toast("🌐 Sin conexión a Nominatim — mostrando ciudades frecuentes.", icon="⚠️")
        return local_matches[:8]
    except _requests.exceptions.HTTPError as e:
        st.toast(f"⚠️ Error Nominatim ({e.response.status_code}).", icon="⚠️")
        return local_matches[:8]
    except (ValueError, KeyError):
        return local_matches[:8]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="cNSFfleet", page_icon="🚛", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

:root {
    --bg:        #080a0f;
    --surface:   #0e1118;
    --surface2:  #141820;
    --border:    #1c2030;
    --border2:   #252d40;
    --amber:     #f5a623;
    --amber-dim: #6b4510;
    --amber-glow:rgba(245,166,35,0.15);
    --blue:      #4f8ef7;
    --teal:      #2dd4bf;
    --violet:    #a78bfa;
    --rose:      #fb7185;
    --text:      #eef0f6;
    --muted:     #5c6680;
    --muted2:    #8892a8;
    --mono:      'DM Mono', monospace;
    --display:   'Syne', sans-serif;
    --body:      'DM Sans', sans-serif;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: var(--body);
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0f18 0%, #080a0f 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

/* ── Hero ── */
.hero {
    padding: 2.8rem 0 1.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: -2rem;
    width: 400px; height: 200px;
    background: radial-gradient(ellipse at 0% 0%, rgba(245,166,35,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245,166,35,0.08);
    border: 1px solid var(--amber-dim);
    border-radius: 99px;
    padding: 5px 16px;
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--amber);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: var(--display);
    font-size: clamp(2.4rem, 4.5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    color: var(--text);
    margin: 0 0 0.6rem;
    letter-spacing: -0.03em;
}
.hero-title span {
    color: var(--amber);
    text-shadow: 0 0 40px rgba(245,166,35,0.4);
}
.hero-sub {
    font-size: 0.93rem;
    color: var(--muted2);
    max-width: 580px;
    line-height: 1.7;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stDateInput label {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stSidebar"] h1 {
    font-family: var(--display) !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.4rem !important;
}

/* ── Inputs ── */
[data-testid="stSidebar"] .stDateInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    transition: border-color 0.2s;
}
[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumb"] {
    background: var(--amber) !important;
    box-shadow: 0 0 8px rgba(245,166,35,0.5) !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] { background: var(--amber) !important; }
[data-testid="stSidebar"] .stDateInput input {
    border: 1px solid #eef0f6 !important;
    color: var(--text) !important;
}
/* ── Searchbox ── */
.stSearchbox input {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--body) !important;
    font-size: 14px !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stSearchbox input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.14) !important;
    outline: none !important;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #f5a623 0%, #e8920f 100%) !important;
    color: #07090e !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--display) !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.04em !important;
    padding: 0.7rem 1.6rem !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 4px 24px rgba(245,166,35,0.3), 0 1px 0 rgba(255,255,255,0.1) inset !important;
}
[data-testid="stButton"] > button:hover {
    box-shadow: 0 6px 32px rgba(245,166,35,0.5) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }
[data-testid="stButton"] > button:disabled {
    background: var(--border) !important;
    color: var(--muted) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.3rem !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--amber-dim) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted2) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.55rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--mono) !important;
    font-size: 11px !important;
}

/* ── Headings ── */
h2, h3 { font-family: var(--display) !important; letter-spacing: -0.015em !important; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; margin: 1.8rem 0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border2) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
.stDataFrame th {
    background: var(--surface2) !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted2) !important;
    padding: 0.6rem 1rem !important;
}
.stDataFrame td {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* ── Status / Alert ── */
[data-testid="stStatus"],
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 12px !important;
    font-family: var(--body) !important;
}

/* ── Captions ── */
[data-testid="stCaptionContainer"], .stCaption {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--muted) !important;
}

/* ── Sidebar markdown ── */
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 12px !important;
    color: var(--muted2) !important;
    line-height: 1.8 !important;
}
[data-testid="stSidebar"] .stMarkdown strong { color: var(--text) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #303550; }
</style>
""", unsafe_allow_html=True)

VEHICLE_NAMES = ["Tractor", "Rígido", "Cisterna"]
DAY_NAMES     = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0e1118",
    "axes.facecolor":    "#0e1118",
    "axes.edgecolor":    "#1c2030",
    "axes.labelcolor":   "#8892a8",
    "axes.titlecolor":   "#eef0f6",
    "axes.titleweight":  "bold",
    "axes.titlesize":    10,
    "axes.labelsize":    8.5,
    "xtick.color":       "#5c6680",
    "ytick.color":       "#5c6680",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "grid.color":        "#141820",
    "grid.linewidth":    1.2,
    "text.color":        "#eef0f6",
    "legend.facecolor":  "#0e1118",
    "legend.edgecolor":  "#1c2030",
    "legend.fontsize":   8.5,
    "font.family":       "monospace",
    "figure.dpi":        140,
    "patch.linewidth":   0,
})

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model():
    checkpoint_path = Path("checkpoints/best_model.pt")
    model = ConditionalFlowModel()
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, True
    return model, False

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⬡ Neural Spline Flow · Probabilistic</div>
    <div class="hero-title">cNSF<span>fleet</span></div>
    <p class="hero-sub">Predictor probabilístico de consumo para transporte pesado.
    Intervalos P5/P50/P95 por tramo usando física real, meteorología en vivo y rutas OSRM.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parámetros")

    vehicle_type = st.selectbox(
        "Tipo de vehículo",
        options=[0, 1, 2],
        format_func=lambda x: VEHICLE_NAMES[x],
    )
    load_pct = st.slider("Carga (%)", 0, 100, 75) / 100.0
    departure_date = st.date_input(
        "Fecha de salida",
        value=date.today() + timedelta(days=1),
        min_value=date.today() - timedelta(days=365),
        max_value=date.today() + timedelta(days=16),
        format="DD/MM/YYYY",
    )
    day_of_week = departure_date.weekday()

    st.divider()
    st.markdown(
        "**APIs utilizadas:**\n"
        "- 🗺️ Nominatim (OSM) — geocodificación\n"
        "- 🛣️ OSRM — routing real\n"
        "- ⛰️ Open-Topo-Data — elevación\n"
        "- 🌤️ Open-Meteo — meteorología\n\n"
        "_Todas gratuitas, sin API key._"
    )

# ── Model state ───────────────────────────────────────────────────────────────
model, model_ready = load_model()
if not model_ready:
    st.warning("⚠️ Modelo no entrenado. Ejecuta `python main.py` primero para generar `checkpoints/best_model.pt`.")

# ── Route form ────────────────────────────────────────────────────────────────
col_in1, col_in2, col_btn = st.columns([2, 2, 1])
with col_in1:
    origin = st_searchbox(
        search_cities,
        placeholder="📍 Ciudad de origen...",
        label="Origen",
        key="searchbox_origin",
        debounce=300,
        clear_on_submit=False,
    )
with col_in2:
    destination = st_searchbox(
        search_cities,
        placeholder="🏁 Ciudad de destino...",
        label="Destino",
        key="searchbox_destination",
        debounce=300,
        clear_on_submit=False,
    )
with col_btn:
    st.write("")
    predict_btn = st.button(
        "⚡ Predecir",
        type="primary",
        use_container_width=True,
        disabled=not model_ready,
    )

# ── Execution ─────────────────────────────────────────────────────────────────
if predict_btn and origin and destination:
    if origin.strip().lower() == destination.strip().lower():
        st.error("⚠️ El origen y el destino no pueden ser iguales. Por favor, selecciona ciudades distintas.")
    else:
        st.session_state.pop("result", None)
        st.session_state.pop("context", None)

    with st.status("Calculando ruta...", expanded=True) as status_box:
        st.write("🗺️ Geocodificando origen y destino...")
        try:
            context = build_route_context(
                origin=origin, destination=destination,
                vehicle_type=vehicle_type, load_pct=load_pct,
                day_of_week=day_of_week, departure_date=departure_date,
            )
        except _requests.exceptions.Timeout:
            st.error("⏱️ La API de rutas tardó demasiado. Comprueba tu conexión e inténtalo de nuevo.")
            st.stop()
        except _requests.exceptions.ConnectionError:
            st.error("🌐 No se pudo conectar con OSRM. Verifica tu conexión a Internet.")
            st.stop()
        except KeyError as e:
            st.error(f"⚠️ Campo inesperado en la respuesta: `{e}`. Prueba con un nombre más específico (ej. 'Sevilla, España').")
            st.stop()
        except ValueError as e:
            st.error(f"⚠️ Datos de ruta inválidos: {e}")
            st.stop()

        st.session_state["context"] = context

        wind_speed    = context["weather"].get("wind_speed", 0.0)
        wind_dir      = context["weather"].get("wind_direction", 0.0)
        route_bearing = context.get("route_bearing", 0.0)
        wind_frontal  = frontal_wind(wind_speed, wind_dir, route_bearing)
        weather_date  = context["weather"].get("date_used", "hoy")

        st.write(f"✅ Ruta: **{context['route_info']['distance_km']:.0f} km**")
        st.write(f"⛰️ Pendiente media: **{context['avg_slope']:.1f}%**")
        st.write(f"🌡️ Temperatura: **{context['weather']['temperature']:.1f} °C** _(fecha: {weather_date})_")
        st.write(f"🌧️ Precipitación: **{context['weather']['precipitation']:.1f} mm/h**")
        st.write(f"💨 Viento: **{wind_speed:.1f} km/h** dir {wind_dir:.0f}° → componente frontal: **{wind_frontal:+.1f} km/h**")
        st.write(f"🤖 Generando {N_SAMPLES} viajes sintéticos con el NSF...")

        predictor = FleetPredictor(model)
        result = predictor.predict_route(
            avg_slope=context["avg_slope"],
            avg_temp=context["weather"]["temperature"],
            precipitation=context["weather"]["precipitation"],
            load_pct=load_pct,
            vehicle_type=vehicle_type,
            day_of_week=day_of_week,
            n_samples=N_SAMPLES,
            wind_kmh=wind_frontal,
        )
        st.session_state["result"] = result
        status_box.update(label="✅ Predicción completada", state="complete")

# ── Results ───────────────────────────────────────────────────────────────────
if "result" in st.session_state and "context" in st.session_state:
    result  = st.session_state["result"]
    context = st.session_state["context"]
    ri      = context["route_info"]
    s       = result["summary"]

    st.divider()

    # ── KPI strip ────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📏 Distancia",    f"{ri['distance_km']:.0f} km")
    m2.metric("⏱️ Duración",      f"{ri['duration_min']:.0f} min")
    m3.metric("⛽ Consumo P50",   f"{s['consumo_medio_l100km']['p50']:.1f} l/100km")
    m4.metric("🚀 Velocidad P50", f"{s['velocidad_media_kmh']['p50']:.1f} km/h")
    m5.metric("💨 Viento",        f"{context['weather'].get('wind_speed', 0.0):.1f} km/h")

    st.divider()

    # ── Map + confidence intervals ────────────────────────────────────────────
    col_map, col_ic = st.columns([1.3, 0.7])

    with col_map:
        st.subheader("🗺️ Ruta calculada")
        polyline = ri["polyline"]
        mid_pt   = polyline[len(polyline) // 2]
        n_segs   = len(result["segments"])

        m = folium.Map(location=mid_pt, zoom_start=7, tiles="CartoDB darkmatter")

        seg_polys = segment_polyline_by_distance(polyline, n_segs)
        for i, seg_pts in enumerate(seg_polys):
            seg_data = result["segments"][i]
            color    = SEG_COLORS[i % len(SEG_COLORS)]
            tooltip  = (
                f"<b style='font-family:monospace'>Tramo {i+1}</b><br>"
                f"Consumo P50: <b>{seg_data['consumo']['p50']:.1f}</b> l/100km<br>"
                f"IC 90%: [{seg_data['consumo']['p5']:.1f} – {seg_data['consumo']['p95']:.1f}]<br>"
                f"Velocidad P50: <b>{seg_data['velocidad']['p50']:.1f}</b> km/h"
            )
            folium.PolyLine(
                seg_pts, color=color, weight=6, opacity=0.92,
                tooltip=folium.Tooltip(tooltip, sticky=True),
            ).add_to(m)
            if i > 0:
                folium.CircleMarker(
                    location=seg_pts[0], radius=6,
                    color=color, fill=True, fill_color=color, fill_opacity=1.0,
                    tooltip=f"Inicio T{i+1}",
                ).add_to(m)

        folium.Marker(
            polyline[0], tooltip=f"🟢 {origin}",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)
        folium.Marker(
            polyline[-1], tooltip=f"🔴 {destination}",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

        st_folium(m, width=560, height=400)

        legend_html = "".join(
            f'<span style="background:{SEG_COLORS[i % len(SEG_COLORS)]};padding:4px 13px;'
            f'border-radius:99px;margin-right:5px;color:#07090e;'
            f'font-size:11px;font-family:monospace;font-weight:700;letter-spacing:0.05em">T{i+1}</span>'
            for i in range(n_segs)
        )
        st.markdown(f'<div style="margin-top:0.6rem">{legend_html}</div>', unsafe_allow_html=True)

    with col_ic:
        st.subheader("📊 Intervalos de confianza")

        c_p5  = s["consumo_medio_l100km"]["p5"]
        c_p50 = s["consumo_medio_l100km"]["p50"]
        c_p95 = s["consumo_medio_l100km"]["p95"]
        v_p5  = s["velocidad_media_kmh"]["p5"]
        v_p50 = s["velocidad_media_kmh"]["p50"]
        v_p95 = s["velocidad_media_kmh"]["p95"]

        st.markdown("**Consumo (l/100km)**")
        cola, colb, colc = st.columns(3)
        cola.metric("P05", f"{c_p5:.1f}", delta=f"{c_p5 - c_p50:.1f}", delta_color="inverse")
        colb.metric("P50", f"{c_p50:.1f}")
        colc.metric("P95", f"{c_p95:.1f}", delta=f"+{c_p95 - c_p50:.1f}", delta_color="inverse")

        st.markdown("**Velocidad (km/h)**")
        cold, cole, colf = st.columns(3)
        cold.metric("P05", f"{v_p5:.1f}")
        cole.metric("P50", f"{v_p50:.1f}")
        colf.metric("P95", f"{v_p95:.1f}")

        st.divider()
        st.markdown("**Condiciones de la ruta**")

        _wspd    = context['weather'].get('wind_speed', 0.0)
        _wdir    = context['weather'].get('wind_direction', 0.0)
        _bearing = context.get('route_bearing', 0.0)
        _wfront  = frontal_wind(_wspd, _wdir, _bearing)
        _label   = "frontal 🔴" if _wfront > 0 else "trasero 🟢"

        info_rows = [
            ("🏔️ Pendiente media", f"{context['avg_slope']:.1f}%"),
            ("🚛 Vehículo",        VEHICLE_NAMES[vehicle_type]),
            ("📦 Carga",           f"{load_pct*100:.0f}%"),
            ("📅 Fecha",           f"{departure_date.strftime('%d/%m/%Y')} ({DAY_NAMES[day_of_week]})"),
            ("🌡️ Temperatura",     f"{context['weather']['temperature']:.1f} °C"),
            ("💨 Viento",          f"{_wspd:.1f} km/h {_label} {abs(_wfront):.1f} km/h"),
        ]
        for label, value in info_rows:
            ca, cb = st.columns([1, 1])
            ca.markdown(f"<span style='font-family:monospace;font-size:10px;color:#5c6680;text-transform:uppercase;letter-spacing:0.1em'>{label}</span>", unsafe_allow_html=True)
            cb.markdown(f"<span style='font-family:monospace;font-size:13px;font-weight:500;color:#eef0f6'>{value}</span>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Análisis por tramos")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        segs_sorted = sorted(result["segments"], key=lambda x: x["tramo"])
        tramos = [f"T{seg['tramo']}" for seg in segs_sorted]
        x = np.arange(len(tramos))

        fig, (ax_c, ax_v) = plt.subplots(1, 2, figsize=(7.5, 3.6),
                                          gridspec_kw={"wspace": 0.38})
        fig.patch.set_facecolor("#0e1118")

        p50s_c = [seg["consumo"]["p50"] for seg in segs_sorted]
        p5s_c  = [seg["consumo"]["p5"]  for seg in segs_sorted]
        p95s_c = [seg["consumo"]["p95"] for seg in segs_sorted]
        p50s_v = [seg["velocidad"]["p50"] for seg in segs_sorted]
        p5s_v  = [seg["velocidad"]["p5"]  for seg in segs_sorted]
        p95s_v = [seg["velocidad"]["p95"] for seg in segs_sorted]

        BG = "#1a2035"
        fig, (ax_c, ax_v) = plt.subplots(1, 2, figsize=(7.5, 3.8),
                                          gridspec_kw={"wspace": 0.42})
        fig.patch.set_facecolor(BG)
        for ax in (ax_c, ax_v):
            ax.set_facecolor(BG)

        def draw_bars(ax, p50s, p5s, p95s, color, ylabel, title):
            yerr = [
                [p50 - p5  for p50, p5  in zip(p50s, p5s)],
                [p95 - p50 for p95, p50 in zip(p95s, p50s)],
            ]
            bars = ax.bar(x, p50s, color=color, alpha=0.75, width=0.52,
                          label="P50 (mediana)", zorder=3)
            ax.errorbar(x, p50s, yerr=yerr, fmt="none",
                        ecolor="#ffffff", elinewidth=2.2, capsize=8, capthick=2.2,
                        label="IC 90% (P5–P95)", zorder=4)
            for xi, val in zip(x, p50s):
                ax.text(xi, ax.get_ylim()[1] * 0.01 if ax.get_ylim()[1] else 0,
                        f"{val:.1f}", ha="center", va="bottom",
                        fontsize=7.5, color="#ffffff", fontfamily="monospace", zorder=5)
            ax.set_title(title, pad=10, color="#e8eaf0")
            ax.set_ylabel(ylabel, color="#9ca3af")
            ax.set_xticks(x); ax.set_xticklabels(tramos)
            ax.grid(alpha=0.15, axis="y", linestyle="--", color="#ffffff")
            ax.spines[:].set_visible(False)
            ax.tick_params(length=0, colors="#9ca3af")
            ax.legend(fontsize=7.5, framealpha=0.2, labelcolor="#e8eaf0",
                      edgecolor="#ffffff30", loc="upper right")

        draw_bars(ax_c, p50s_c, p5s_c, p95s_c, BLUE,  "l / 100 km", "Consumo P50 · IC 90%")
        draw_bars(ax_v, p50s_v, p5s_v, p95s_v, TEAL,  "km / h",     "Velocidad P50 · IC 90%")

        # Ajuste del ylim para que el texto encima de las barras no se corte
        for ax, p50s, p95s in [(ax_c, p50s_c, p95s_c), (ax_v, p50s_v, p95s_v)]:
            margin = (max(p95s) - min(p50s)) * 0.18
            ax.set_ylim(min(p50s) * 0.88, max(p95s) + margin)
            for xi, val, p50 in zip(x, p95s, p50s):
                ax.texts[xi].set_y(p50 + (p95s[xi] - p50) + margin * 0.15)

        st.pyplot(fig)
        plt.close()

    with col_g2:
        fig2, ax2 = plt.subplots(figsize=(6.2, 3.6))
        fig2.patch.set_facecolor("#0e1118")

        v_min, v_max_val = MINS[0], MAXS[0]
        trips_v = (result["trips_raw"][:, :, 0] + 1) / 2 * (v_max_val - v_min) + v_min

        t = np.arange(trips_v.shape[1])
        p5  = np.percentile(trips_v, 5,  axis=0)
        p50 = np.percentile(trips_v, 50, axis=0)
        p95 = np.percentile(trips_v, 95, axis=0)

        # Muestra hasta 8 trips individuales en gris muy suave
        for i in range(min(8, len(trips_v))):
            ax2.plot(t, trips_v[i], alpha=0.13, linewidth=0.6, color="#8892a8")


        # Percentiles P5 / P95 como líneas punteadas
        ax2.plot(t, p5,  color=BLUE,  linewidth=1.0, linestyle="--", alpha=0.7, label="P05 / P95")
        ax2.plot(t, p95, color=BLUE,  linewidth=1.0, linestyle="--", alpha=0.7)

        # Mediana destacada en amber
        ax2.plot(t, p50, color=AMBER, linewidth=2.2, label="Mediana P50", zorder=5)

        ax2.set_title("Perfiles de velocidad sintéticos", pad=12)
        ax2.set_xlabel("Tiempo (min)")
        ax2.set_ylabel("km / h")
        ax2.legend(frameon=True, loc="upper right")
        ax2.grid(alpha=0.2, linestyle="--")
        ax2.spines[:].set_visible(False)
        ax2.tick_params(length=0)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Tabla por tramos")

    rows = []
    for seg in sorted(result["segments"], key=lambda x: x["tramo"]):
        rows.append({
            "Tramo":         f"T{seg['tramo']}",
            "Consumo P05":   f"{seg['consumo']['p5']:.1f}",
            "Consumo P50 ←": f"{seg['consumo']['p50']:.1f}",
            "Consumo P95":   f"{seg['consumo']['p95']:.1f}",
            "IC Consumo":    f"±{(seg['consumo']['p95']-seg['consumo']['p5'])/2:.1f}",
            "Vel P05":       f"{seg['velocidad']['p5']:.1f}",
            "Vel P50":       f"{seg['velocidad']['p50']:.1f}",
            "Vel P95":       f"{seg['velocidad']['p95']:.1f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div style="
        margin-top: 3rem;
        padding: 3.5rem 2rem;
        text-align: center;
        border: 1px dashed #1c2030;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(14,17,24,0.6) 0%, rgba(8,10,15,0.8) 100%);
        color: #5c6680;
        font-family: 'DM Mono', monospace;
        font-size: 13px;
        letter-spacing: 0.05em;
    ">
        ↑ Introduce origen y destino arriba y pulsa&nbsp;
        <strong style="color:#f5a623">⚡ Predecir</strong>
    </div>
    """, unsafe_allow_html=True)