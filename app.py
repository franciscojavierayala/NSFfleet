"""
app.py — Interfaz Streamlit para NSFfleet Predictor
Introduce origen y destino (en texto), selecciona tipo de camión y carga,
y obtén predicciones de consumo y velocidad con intervalos de confianza.

Ejecutar con:
    streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import torch
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests as _requests
from streamlit_searchbox import st_searchbox

from model.nflow_model import ConditionalFlowModel
from inference.predictor import FleetPredictor
from route.route_builder import build_route_context


# ── Lista de ciudades frecuentes para fallback rápido ────────────────────────
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


# ── Autocompletado de ciudades con Nominatim ─────────────────────────────────
def search_cities(query: str) -> list[str]:
    """Devuelve sugerencias de ciudades mientras el usuario escribe."""
    if not query or len(query) < 2:
        return []

    # Filtro rápido local primero (instantáneo)
    q_lower = query.lower()
    local_matches = [c for c in CITIES_FALLBACK if q_lower in c.lower()]

    # Luego intenta Nominatim para resultados más completos
    try:
        resp = _requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "limit": 7,
                "addressdetails": 1,
                "accept-language": "es",
            },
            headers={"User-Agent": "cvae_fleet_demo/1.0"},
            timeout=4,
        )
        results = resp.json()
        seen = set(local_matches)
        suggestions = list(local_matches)
        for r in results:
            addr = r.get("address", {})
            city = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("municipality")
                or r.get("display_name", "").split(",")[0].strip()
            )
            country = addr.get("country", "")
            short = f"{city}, {country}".strip(", ")
            if short and short not in seen:
                seen.add(short)
                suggestions.append(short)
        return suggestions[:8]
    except Exception:
        return local_matches[:8]


# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSFfleet Predictor",
    page_icon="🚛",
    layout="wide",
)

VEHICLE_NAMES = ["Tractor", "Rígido", "Cisterna"]
DAY_NAMES     = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


# ── Cargar modelo ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model():
    checkpoint_path = Path("checkpoints/best_model.pt")
    model = ConditionalFlowModel()
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, True
    return model, False


# ── Layout principal ───────────────────────────────────────────────────────────
st.title("🚛 NSFfleet — Predictor de rutas")
st.caption(
    "Introduce origen y destino. El sistema calcula la ruta real, "
    "obtiene pendientes y meteorología, y genera viajes sintéticos "
    "para estimar consumo y velocidad con intervalos de confianza P5/P50/P95."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parámetros del vehículo")

    vehicle_type = st.selectbox(
        "Tipo de vehículo",
        options=[0, 1, 2],
        format_func=lambda x: VEHICLE_NAMES[x],
    )
    load_pct = st.slider("Carga (%)", 0, 100, 75) / 100.0
    from datetime import date, timedelta
    departure_date = st.date_input(
        "Fecha de salida",
        value=date.today() + timedelta(days=1),
        min_value=date.today() - timedelta(days=365),
        max_value=date.today() + timedelta(days=16),
        format="DD/MM/YYYY",
    )
    day_of_week = departure_date.weekday()  # 0=lunes ... 6=domingo
    st.caption(f"📅 {DAY_NAMES[day_of_week]}")
    n_samples = st.select_slider(
        "Viajes sintéticos a generar",
        options=[50, 100, 150, 200],
        value=100,
    )

    st.divider()
    st.markdown(
        "**APIs utilizadas:**\n"
        "- 🗺️ Nominatim (OSM) — geocodificación\n"
        "- 🛣️ OSRM — routing real\n"
        "- ⛰️ Open-Topo-Data — elevación\n"
        "- 🌤️ Open-Meteo — meteorología\n\n"
        "_Todas gratuitas, sin API key._"
    )

# ── Estado del modelo ──────────────────────────────────────────────────────────
model, model_ready = load_model()
if not model_ready:
    st.warning(
        "⚠️ Modelo no entrenado. Ejecuta `python main.py` primero para generar "
        "`checkpoints/best_model.pt`, luego vuelve aquí."
    )

# ── Formulario de ruta ─────────────────────────────────────────────────────────
col_in1, col_in2, col_btn = st.columns([2, 2, 1])
with col_in1:
    origin = st_searchbox(
        search_cities,
        placeholder="📍 Escribe una ciudad de origen...",
        label="Origen",
        key="searchbox_origin",
        debounce=300,
        clear_on_submit=False,
    )
with col_in2:
    destination = st_searchbox(
        search_cities,
        placeholder="🏁 Escribe una ciudad de destino...",
        label="Destino",
        key="searchbox_destination",
        debounce=300,
        clear_on_submit=False,
    )
with col_btn:
    st.write("")
    predict_btn = st.button(
        "🔮 Predecir",
        type="primary",
        use_container_width=True,
        disabled=not model_ready,
    )

# ── Ejecución ──────────────────────────────────────────────────────────────────
if predict_btn and origin and destination:

    # Limpiar estado anterior
    st.session_state.pop("result", None)
    st.session_state.pop("context", None)

    with st.status("Calculando ruta...", expanded=True) as status_box:

        st.write("🗺️ Geocodificando origen y destino...")
        try:
            context = build_route_context(
                origin=origin,
                destination=destination,
                vehicle_type=vehicle_type,
                load_pct=load_pct,
                day_of_week=day_of_week,
                departure_date=departure_date,
            )
            st.session_state["context"] = context
        except Exception as e:
            st.error(f"Error al obtener la ruta: {e}")
            st.stop()

        st.write(f"✅ Ruta obtenida: {context['route_info']['distance_km']:.0f} km")
        st.write(f"⛰️ Pendiente media: {context['avg_slope']:.1f}%")
        weather_date = context["weather"].get("date_used", "hoy")
        st.write(f"🌡️ Temperatura: {context['weather']['temperature']:.1f} °C  _(fecha: {weather_date})_")
        st.write(f"🌧️ Precipitación: {context['weather']['precipitation']:.1f} mm/h")
        wind_speed = context["weather"].get("wind_speed", 0.0)
        wind_dir   = context["weather"].get("wind_direction", 0.0)
        route_bearing = context.get("route_bearing", 0.0)

        # Componente frontal real: cos(ángulo entre viento y dirección de marcha)
        # Si el viento viene exactamente de frente → cos(0°)=1.0 → máxima penalización
        # Si viene de atrás → cos(180°)=-1.0 → ayuda (reduce consumo)
        import math
        angle_diff = math.radians(abs(route_bearing - wind_dir) % 360)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        wind_frontal = wind_speed * math.cos(angle_diff)

        st.write(f"💨 Viento: {wind_speed:.1f} km/h  _(dirección: {wind_dir:.0f}°, componente frontal: {wind_frontal:+.1f} km/h)_")

        st.write(f"🤖 Generando {n_samples} viajes sintéticos con el cVAE...")
        predictor = FleetPredictor(model)
        result = predictor.predict_route(
            avg_slope=context["avg_slope"],
            avg_temp=context["weather"]["temperature"],
            precipitation=context["weather"]["precipitation"],
            load_pct=load_pct,
            vehicle_type=vehicle_type,
            day_of_week=day_of_week,
            n_samples=n_samples,
            wind_kmh=wind_frontal,
        )
        st.session_state["result"] = result

        status_box.update(label="✅ Predicción completada", state="complete")

# ── Resultados ─────────────────────────────────────────────────────────────────
if "result" in st.session_state and "context" in st.session_state:
    result  = st.session_state["result"]
    context = st.session_state["context"]
    ri      = context["route_info"]
    s       = result["summary"]

    st.divider()

    # ── Métricas rápidas ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📏 Distancia",        f"{ri['distance_km']:.0f} km")
    m2.metric("⏱️ Duración",          f"{ri['duration_min']:.0f} min")
    m3.metric("⛽ Consumo P50",       f"{s['consumo_medio_l100km']['p50']:.1f} l/100km")
    m4.metric("🚀 Velocidad P50",     f"{s['velocidad_media_kmh']['p50']:.1f} km/h")
    m5.metric("💨 Viento",            f"{context['weather'].get('wind_speed', 0.0):.1f} km/h")

    st.divider()

    # ── Mapa + intervalos de confianza ─────────────────────────────────────────
    col_map, col_ic = st.columns([1.3, 0.7])

    with col_map:
        st.subheader("🗺️ Ruta calculada")
        polyline = ri["polyline"]
        mid_pt   = polyline[len(polyline) // 2]
        n_segs   = len(result["segments"])

        # Colores por tramo
        SEG_COLORS = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#9B5DE5", "#F77F00"]

        m = folium.Map(location=mid_pt, zoom_start=7, tiles="CartoDB positron")

        # Dividir polyline en N tramos y pintarlos con color distinto
        seg_size = max(1, len(polyline) // n_segs)
        for i in range(n_segs):
            start_i = i * seg_size
            end_i   = (i + 1) * seg_size if i < n_segs - 1 else len(polyline)
            seg_pts  = polyline[start_i:end_i + 1]
            seg_data = result["segments"][i]
            color    = SEG_COLORS[i % len(SEG_COLORS)]
            tooltip  = (
                f"<b>Tramo {i+1}</b><br>"
                f"Consumo P50: {seg_data['consumo']['p50']:.1f} l/100km<br>"
                f"IC 90%: [{seg_data['consumo']['p5']:.1f} – {seg_data['consumo']['p95']:.1f}]<br>"
                f"Velocidad P50: {seg_data['velocidad']['p50']:.1f} km/h"
            )
            folium.PolyLine(
                seg_pts, color=color, weight=6, opacity=0.9,
                tooltip=folium.Tooltip(tooltip, sticky=True),
            ).add_to(m)

            # Punto de inicio de cada tramo (excepto el origen)
            if i > 0:
                folium.CircleMarker(
                    location=seg_pts[0],
                    radius=5, color=color, fill=True, fill_opacity=1.0,
                    tooltip=f"Inicio T{i+1}",
                ).add_to(m)

        # Marcadores origen / destino
        folium.Marker(
            polyline[0], tooltip=f"🟢 Origen: {origin}",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)
        folium.Marker(
            polyline[-1], tooltip=f"🔴 Destino: {destination}",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

        st_folium(m, width=560, height=400)

        # Leyenda de colores de tramos
        legend_html = "".join(
            f'<span style="background:{SEG_COLORS[i]};padding:2px 10px;border-radius:4px;'
            f'margin-right:6px;color:white;font-size:12px">T{i+1}</span>'
            for i in range(n_segs)
        )
        st.markdown(legend_html, unsafe_allow_html=True)

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
        st.write(f"🏔️ Pendiente media: **{context['avg_slope']:.1f}%**")
        st.write(f"🚛 Vehículo: **{VEHICLE_NAMES[vehicle_type]}**")
        st.write(f"📦 Carga: **{load_pct*100:.0f}%**")
        st.write(f"📅 Fecha: **{departure_date.strftime('%d/%m/%Y')}** ({DAY_NAMES[day_of_week]})")
        st.write(f"🌡️ Temperatura: **{context['weather']['temperature']:.1f} °C**")
        _wspd = context['weather'].get('wind_speed', 0.0)
        _wdir = context['weather'].get('wind_direction', 0.0)
        _bearing = context.get('route_bearing', 0.0)
        import math as _math
        _adiff = _math.radians(abs(_bearing - _wdir) % 360)
        if _adiff > _math.pi: _adiff = 2*_math.pi - _adiff
        _wfront = _wspd * _math.cos(_adiff)
        _label = "frontal 🔴" if _wfront > 0 else "trasero 🟢"
        st.write(f"💨 Viento: **{_wspd:.1f} km/h** dir {_wdir:.0f}° → componente {_label} **{abs(_wfront):.1f} km/h**")


    # ── Gráficos ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Análisis por tramos")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        segs_sorted = sorted(result["segments"], key=lambda x: x["tramo"])
        tramos = [f"T{seg['tramo']}" for seg in segs_sorted]

        fig, (ax_c, ax_v) = plt.subplots(1, 2, figsize=(7, 3.2))

        # Consumo
        p50s_c = [seg["consumo"]["p50"] for seg in segs_sorted]
        p5s_c  = [seg["consumo"]["p5"]  for seg in segs_sorted]
        p95s_c = [seg["consumo"]["p95"] for seg in segs_sorted]
        yerr_c = [[p50-p5 for p50,p5 in zip(p50s_c,p5s_c)],
                  [p95-p50 for p95,p50 in zip(p95s_c,p50s_c)]]
        ax_c.bar(tramos, p50s_c, color="#1F5C99", alpha=0.85,
                 yerr=yerr_c, capsize=5, error_kw={"color":"#E07B39","linewidth":1.8})
        ax_c.set_title("Consumo P50 (IC 90%)", fontsize=10)
        ax_c.set_ylabel("l/100km")
        ax_c.grid(alpha=0.3, axis="y")

        # Velocidad
        p50s_v = [seg["velocidad"]["p50"] for seg in segs_sorted]
        p5s_v  = [seg["velocidad"]["p5"]  for seg in segs_sorted]
        p95s_v = [seg["velocidad"]["p95"] for seg in segs_sorted]
        yerr_v = [[p50-p5 for p50,p5 in zip(p50s_v,p5s_v)],
                  [p95-p50 for p95,p50 in zip(p95s_v,p50s_v)]]
        ax_v.bar(tramos, p50s_v, color="#2A9D8F", alpha=0.85,
                 yerr=yerr_v, capsize=5, error_kw={"color":"#E07B39","linewidth":1.8})
        ax_v.set_title("Velocidad P50 (IC 90%)", fontsize=10)
        ax_v.set_ylabel("km/h")
        ax_v.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_g2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
        trips_v = (result["trips_raw"][:, :, 0] + 1) / 2 * 130
        for i in range(min(5, len(trips_v))):
            ax2.plot(trips_v[i], alpha=0.25, linewidth=0.8)
        ax2.plot(np.percentile(trips_v, 50, axis=0),
                 color="black", linewidth=1.8, label="Mediana P50")
        ax2.fill_between(
            range(trips_v.shape[1]),
            np.percentile(trips_v, 5,  axis=0),
            np.percentile(trips_v, 95, axis=0),
            alpha=0.15, color="#1F5C99", label="IC 90%",
        )
        ax2.set_title("Perfiles de velocidad sintéticos", fontsize=11)
        ax2.set_xlabel("Tiempo (min)")
        ax2.set_ylabel("km/h")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Tabla resumen ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Tabla por tramos")

    rows = []
    for seg in sorted(result["segments"], key=lambda x: x["tramo"]):
        rows.append({
            "Tramo":          f"T{seg['tramo']}",
            "Consumo P05":    f"{seg['consumo']['p5']:.1f}",
            "Consumo P50 ←":  f"{seg['consumo']['p50']:.1f}",
            "Consumo P95":    f"{seg['consumo']['p95']:.1f}",
            "IC Consumo":     f"±{(seg['consumo']['p95']-seg['consumo']['p5'])/2:.1f}",
            "Vel P05":        f"{seg['velocidad']['p5']:.1f}",
            "Vel P50":        f"{seg['velocidad']['p50']:.1f}",
            "Vel P95":        f"{seg['velocidad']['p95']:.1f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

else:
    # Estado inicial
    st.info(
        "👆 Introduce origen y destino en los campos de arriba y pulsa **Predecir** "
        "para obtener las métricas de la ruta."
    )
