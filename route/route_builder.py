"""
route/route_builder.py
Construye el vector de condicionamiento c a partir de una ruta real.

APIs gratuitas, sin necesidad de API key:
  - Nominatim  (OSM)          → geocodificación nombre → coordenadas
  - OSRM public               → polyline de ruta real + distancia + duración
  - Open-Topo-Data (SRTM90m)  → perfil de elevación
  - Open-Meteo                → temperatura y precipitación actuales
"""

import math
import requests
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


NOMINATIM_UA = "cnsffleet_demo/1.0"


# ── Geocodificación ────────────────────────────────────────────────────────────
def geocode(place: str) -> tuple[float, float]:
    """Convierte un nombre de lugar a (lat, lon)."""
    geolocator = Nominatim(user_agent=NOMINATIM_UA)
    try:
        location = geolocator.geocode(place, timeout=10)
        if location is None:
            raise ValueError(f"No se encontró la ubicación: '{place}'")
        return location.latitude, location.longitude
    except GeocoderTimedOut:
        raise ValueError(f"Timeout al geocodificar: '{place}'")


# ── Routing con OSRM ──────────────────────────────────────────────────────────
def get_route(origin_latlon: tuple, dest_latlon: tuple) -> dict:
    """
    Obtiene la ruta real entre dos puntos usando el servidor público de OSRM.
    Devuelve polyline, distancia (km) y duración (min).
    """
    lat1, lon1 = origin_latlon
    lat2, lon2 = dest_latlon

    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}"
        f"?overview=full&geometries=geojson"
    )

    resp = requests.get(url, timeout=20)
    if resp.status_code == 400:
        raise ValueError("OSRM no pudo calcular la ruta. ¿Origen y destino en continentes distintos o sin carretera entre ellos?")
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "Ok":
        raise ValueError(f"OSRM no pudo calcular la ruta: {data.get('code')}")

    route = data["routes"][0]
    coords = route["geometry"]["coordinates"]  # [[lon, lat], ...]

    return {
        "coordinates": coords,                         # [[lon, lat], ...]
        "polyline":    [(c[1], c[0]) for c in coords], # [(lat, lon), ...] para folium
        "distance_km": route["distance"] / 1000,
        "duration_min": route["duration"] / 60,
    }


# ── Perfil de elevación ────────────────────────────────────────────────────────
def get_elevation_profile_with_source(coords: list, n_samples: int = 20):
    """Igual que get_elevation_profile pero devuelve (elevations, source_name)."""
    step = max(1, len(coords) // n_samples)
    sampled = coords[::step][:n_samples]
    locations = [{"latitude": c[1], "longitude": c[0]} for c in sampled]

    try:
        resp = requests.post(
            "https://api.opentopodata.org/v1/srtm90m",
            json={"locations": locations}, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "OK":
            elevs = [r["elevation"] or 0.0 for r in data["results"]]
            if any(e != 0.0 for e in elevs):
                return np.array(elevs, dtype=np.float32), "Open-Topo-Data (SRTM 90m)"
    except Exception:
        pass

    try:
        loc_str = "|".join(f"{c[1]},{c[0]}" for c in sampled)
        resp = requests.get(
            f"https://api.open-elevation.com/api/v1/lookup?locations={loc_str}", timeout=10,
        )
        resp.raise_for_status()
        elevs = [r["elevation"] for r in resp.json()["results"]]
        if any(e != 0.0 for e in elevs):
            return np.array(elevs, dtype=np.float32), "Open-Elevation"
    except Exception:
        pass

    try:
        lats = ",".join(str(c[1]) for c in sampled)
        lons = ",".join(str(c[0]) for c in sampled)
        resp = requests.get(
            f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}", timeout=8,
        )
        resp.raise_for_status()
        elevs = resp.json()["elevation"]
        return np.array(elevs, dtype=np.float32), "Open-Meteo Elevation"
    except Exception:
        pass

    return np.zeros(len(sampled), dtype=np.float32), "Fallback (terreno plano)"


def get_elevation_profile(coords: list, n_samples: int = 20) -> np.ndarray:
    """
    Obtiene el perfil de elevación (metros) para puntos a lo largo de la ruta.
    coords: lista de [lon, lat] procedente de OSRM.
    Intenta Open-Topo-Data primero, luego Open-Elevation como respaldo.
    """
    step = max(1, len(coords) // n_samples)
    sampled = coords[::step][:n_samples]
    locations = [{"latitude": c[1], "longitude": c[0]} for c in sampled]

    # Intento 1: Open-Topo-Data
    try:
        resp = requests.post(
            "https://api.opentopodata.org/v1/srtm90m",
            json={"locations": locations},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "OK":
            elevations = [r["elevation"] or 0.0 for r in data["results"]]
            if any(e != 0.0 for e in elevations):
                return np.array(elevations, dtype=np.float32)
    except Exception:
        pass

    # Intento 2: Open-Elevation (API alternativa gratuita)
    try:
        loc_str = "|".join(f"{c[1]},{c[0]}" for c in sampled)
        resp = requests.get(
            f"https://api.open-elevation.com/api/v1/lookup?locations={loc_str}",
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        elevations = [r["elevation"] for r in results]
        if any(e != 0.0 for e in elevations):
            return np.array(elevations, dtype=np.float32)
    except Exception:
        pass

    # Intento 3: Open-Meteo elevation (muy fiable, parte del forecast)
    try:
        lats = ",".join(str(c[1]) for c in sampled)
        lons = ",".join(str(c[0]) for c in sampled)
        resp = requests.get(
            f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}",
            timeout=8,
        )
        resp.raise_for_status()
        elevations = resp.json()["elevation"]
        return np.array(elevations, dtype=np.float32)
    except Exception:
        pass

    # Fallback final: terreno plano
    return np.zeros(len(sampled), dtype=np.float32)


def compute_avg_slope(elevations: np.ndarray, total_distance_km: float) -> float:
    """
    Calcula la pendiente media absoluta de la ruta (%).
    Usa la distancia real entre puntos de muestreo, no la media.
    Filtra cambios de elevación imposibles (errores de la API).
    """
    if len(elevations) < 2 or total_distance_km <= 0:
        return 0.0
    segment_m = (total_distance_km * 1000) / (len(elevations) - 1)
    if segment_m <= 0:
        return 0.0
    diffs = np.diff(elevations.astype(np.float32))
    # Filtrar saltos imposibles (>500m entre puntos consecutivos = error API)
    max_realistic = segment_m * 0.30  # max 30% de pendiente fisica
    diffs_clean = diffs[np.abs(diffs) < max_realistic]
    if len(diffs_clean) == 0:
        diffs_clean = diffs
    slopes_pct = np.abs(diffs_clean) / segment_m * 100
    # Usar percentil 75 en vez de media para capturar tramos exigentes
    return float(np.percentile(slopes_pct, 75))


# ── Meteorología ──────────────────────────────────────────────────────────────
def get_weather(lat: float, lon: float, target_date=None) -> dict:
    """
    Temperatura (°C) y precipitación media diaria para una fecha concreta.
    - Fechas futuras (<=16 días): forecast de Open-Meteo
    - Fechas pasadas: historical API de Open-Meteo
    - Sin fecha: condiciones actuales
    """
    from datetime import date as _date, timedelta

    today = _date.today()

    if target_date is None or target_date == today:
        # Condiciones actuales
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,precipitation,wind_speed_10m,wind_direction_10m"
            f"&timezone=auto"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            current = resp.json()["current"]
            return {
                "temperature":   float(current["temperature_2m"]),
                "precipitation": float(current["precipitation"]),
                "wind_speed":    float(current.get("wind_speed_10m", 0.0)),
                "wind_direction": float(current.get("wind_direction_10m", 0.0)),
                "date_used":     str(today),
            }
        except Exception:
            return {"temperature": 15.0, "precipitation": 0.0, "wind_speed": 0.0, "wind_direction": 0.0, "date_used": "fallback"}

    date_str = str(target_date)

    if target_date > today:
        # Forecast (hasta 16 días vista)
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant"
            f"&timezone=auto"
            f"&start_date={date_str}&end_date={date_str}"
        )
    else:
        # Histórico
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max"
            f"&timezone=auto"
            f"&start_date={date_str}&end_date={date_str}"
        )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        daily = resp.json()["daily"]
        temp  = daily["temperature_2m_mean"][0]
        prec  = daily["precipitation_sum"][0]
        wind  = daily.get("wind_speed_10m_max", [0.0])[0]
        wdir  = daily.get("wind_direction_10m_dominant", [0.0])[0]
        return {
            "temperature":    float(temp) if temp is not None else 15.0,
            "precipitation":  float(prec) if prec is not None else 0.0,
            "wind_speed":     float(wind) if wind is not None else 0.0,
            "wind_direction": float(wdir) if wdir is not None else 0.0,
            "date_used":      date_str,
        }
    except Exception:
        return {"temperature": 15.0, "precipitation": 0.0, "wind_speed": 0.0, "wind_direction": 0.0, "date_used": "fallback"}


def compute_route_bearing(origin_ll: tuple, dest_ll: tuple) -> float:
    """
    Calcula el ángulo medio de la ruta en grados (0=Norte, 90=Este, 180=Sur, 270=Oeste).
    Usa la fórmula de azimut entre origen y destino.
    """
    lat1 = math.radians(origin_ll[0])
    lat2 = math.radians(dest_ll[0])
    dlon = math.radians(dest_ll[1] - origin_ll[1])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360  # normalizar a [0, 360]

def haversine_km(ll1, ll2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [ll1[0], ll1[1], ll2[0], ll2[1]])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))
# ── Pipeline completo ─────────────────────────────────────────────────────────
def build_route_context(
    origin: str,
    destination: str,
    vehicle_type: int = 0,
    load_pct: float = 0.7,
    day_of_week: int = 0,
    departure_date=None,
    
) -> dict:
    """
    Pipeline completo: nombre de ciudad → vector de condicionamiento c.

    Pasos:
      1. Geocodificar origen y destino
      2. Calcular ruta real con OSRM
      3. Obtener perfil de elevación y calcular pendiente media
      4. Obtener meteorología en el punto medio de la ruta
      5. Construir vector c

    Returns:
        dict con condition_vector, route_info, weather, avg_slope, coordenadas
    """
    from data.synthetic import generate_conditioning_vector

    # 1. Geocodificación
    origin_ll = geocode(origin)
    dest_ll   = geocode(destination)

    # 2. Ruta real
    route_info = get_route(origin_ll, dest_ll)

    # 3. Elevación y pendiente (con log de qué API funcionó)
    n_elev = min(50, max(20, len(route_info["coordinates"]) // 10))
    elevations, elev_source = get_elevation_profile_with_source(route_info["coordinates"], n_samples=n_elev)
    avg_slope  = compute_avg_slope(elevations, route_info["distance_km"])
    route_info["elevation_source"] = elev_source

    # 4. Meteorología en el punto medio
    coords   = route_info["coordinates"]
    mid      = coords[len(coords) // 2]
    weather  = get_weather(mid[1], mid[0], target_date=departure_date)

    # 5. Ángulo medio de la ruta
    route_bearing = compute_route_bearing(origin_ll, dest_ll)

    # 6. Vector de condicionamiento
    c = generate_conditioning_vector(
        avg_slope=avg_slope,
        avg_temp=weather["temperature"],
        precipitation=weather["precipitation"],
        load_pct=load_pct,
        vehicle_type=vehicle_type,
        day_of_week=day_of_week,
    )

    return {
        "condition_vector": c,
        "route_info":       route_info,
        "weather":          weather,
        "avg_slope":        avg_slope,
        "elevations":       elevations,
        "origin_ll":        origin_ll,
        "dest_ll":          dest_ll,
        "route_bearing":    route_bearing,
        "haversine_km":     haversine_km(origin_ll, dest_ll),
    }