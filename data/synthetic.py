"""
data/synthetic.py
Generador de viajes sintéticos con física real de vehículos pesados.

Mejoras respecto a la versión anterior:
  - Aerodinámica real: F_drag = 0.5 * Cd * A * rho * v²
  - Resistencia a la rodadura: F_roll = Cr * m * g * cos(θ)
  - Resistencia a la pendiente: F_grade = m * g * sin(θ)
  - Consumo desde BSFC (Brake Specific Fuel Consumption map real)
  - Caja de cambios de 12 velocidades con cambios realistas
  - Modelo térmico del motor con inercia (calentamiento en frío)
  - Perfiles de conducción: autopista, secundaria, retenciones, paradas
  - Física diferenciada por tipo de vehículo (tractor, rígido, cisterna)
  - Efecto sloshing en cisterna según nivel de carga
  - Ruido de sensor modelado (deriva, cuantización, picos)
  - Penalización de velocidad en pendiente corregida (fix clip 0.65→0.35,
    eliminado suelo 0.70 que anulaba grade_penalty en subidas pronunciadas)

Variables (F=8):
  0 - velocidad (km/h)
  1 - consumo instantáneo (l/100km)
  2 - RPM
  3 - temperatura motor (°C)
  4 - pendiente (%)
  5 - temperatura exterior (°C)
  6 - precipitación (mm/h)
  7 - porcentaje de carga (0-1)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── Constantes ────────────────────────────────────────────────────────────────
T     = 480    # pasos temporales (1 min/paso → 8 h de viaje)
F     = 8      # variables de telemetría
C_DIM = 16     # dimensión vector de condicionamiento

# Rangos para normalización a [-1, 1]
MINS = np.array([0,    5,   500,  50,  -25, -40,  0,  0  ], dtype=np.float32)
MAXS = np.array([130, 120, 3500, 115,   25,  60, 100,  1  ], dtype=np.float32)

VAR_NAMES = [
    "velocidad", "consumo", "RPM", "temp_motor",
    "pendiente", "temp_ext", "precipitacion", "carga",
]

VEHICLE_NAMES = {0: "Tractor", 1: "Rígido", 2: "Cisterna"}


# ── Perfiles físicos por tipo de vehículo ─────────────────────────────────────
VEHICLE_PROFILES = {
    0: {  # Tractor (semitrailer, Class 8)
        "name":             "Tractor",
        "mass_empty":       8_500,
        "mass_trailer":     13_500,
        "payload_max":      24_000,
        "Cd":               0.65,
        "A_frontal":        9.5,
        "Cr":               0.006,
        "P_engine_kW":      420,
        "rpm_idle":         650,
        "rpm_rated":        1800,
        "rpm_max":          2200,
        "gear_ratios":      [14.93, 11.48, 8.83, 6.79, 5.22, 4.02,
                             3.09,  2.38,  1.83, 1.41, 1.0,  0.78],
        "axle_ratio":       3.08,
        "tire_radius":      0.508,
        "v_max":            90,
        "temp_thermostat":  88,
        "temp_nominal":     93,
        "eta_transmission": 0.94,
        "sloshing":         False,
    },
    1: {  # Rígido (Class 6-7)
        "name":             "Rígido",
        "mass_empty":       7_500,
        "mass_trailer":     0,
        "payload_max":      12_000,
        "Cd":               0.72,
        "A_frontal":        8.0,
        "Cr":               0.007,
        "P_engine_kW":      250,
        "rpm_idle":         700,
        "rpm_rated":        2000,
        "rpm_max":          2600,
        "gear_ratios":      [11.73, 8.51, 6.18, 4.49, 3.26, 2.37,
                             1.72,  1.25,  0.91, 0.66],
        "axle_ratio":       4.11,
        "tire_radius":      0.457,
        "v_max":            90,
        "temp_thermostat":  85,
        "temp_nominal":     90,
        "eta_transmission": 0.93,
        "sloshing":         False,
    },
    2: {  # Cisterna (tanker, ADR)
        "name":             "Cisterna",
        "mass_empty":       9_500,
        "mass_trailer":     15_000,
        "payload_max":      21_000,
        "Cd":               0.70,
        "A_frontal":        9.2,
        "Cr":               0.0065,
        "P_engine_kW":      400,
        "rpm_idle":         650,
        "rpm_rated":        1700,
        "rpm_max":          2100,
        "gear_ratios":      [14.93, 11.48, 8.83, 6.79, 5.22, 4.02,
                             3.09,  2.38,  1.83, 1.41, 1.0,  0.78],
        "axle_ratio":       3.36,
        "tire_radius":      0.508,
        "v_max":            85,
        "temp_thermostat":  88,
        "temp_nominal":     92,
        "eta_transmission": 0.94,
        "sloshing":         True,
    },
}

# ── Constantes físicas ────────────────────────────────────────────────────────
RHO_AIR        = 1.225   # kg/m³
G              = 9.81    # m/s²
DIESEL_DENSITY = 0.835   # kg/l

# ── BSFC map (g/kWh) — motor diesel Euro VI pesado (Volvo/Scania/DAF) ────────
# Filas: carga 0/25/50/75/100%  |  Columnas: RPM ralentí→máximo
BSFC_MAP = np.array([
    [340, 310, 300, 305, 320],  # carga   0% — ralentí
    [210, 195, 185, 190, 200],  # carga  25% — carga ligera
    [192, 180, 172, 176, 188],  # carga  50% — zona habitual autopista
    [188, 178, 170, 174, 185],  # carga  75% — carga alta
    [195, 185, 178, 183, 198],  # carga 100% — plena carga
], dtype=np.float32)


# ── Helpers físicos ───────────────────────────────────────────────────────────
def _total_mass(p, load_pct):
    return p["mass_empty"] + p["mass_trailer"] + load_pct * p["payload_max"]

def _aero_drag(p, v_ms, wind_ms=0.0):
    v_eff = v_ms + wind_ms   # positivo = viento frontal, negativo = trasero
    return 0.5 * p["Cd"] * p["A_frontal"] * RHO_AIR * v_eff ** 2

def _rolling_resistance(p, mass_kg, slope_pct):
    theta = np.arctan(slope_pct / 100.0)
    return p["Cr"] * mass_kg * G * np.cos(theta)

def _grade_force(mass_kg, slope_pct):
    theta = np.arctan(slope_pct / 100.0)
    return mass_kg * G * np.sin(theta)

def _bsfc_lookup(load_fraction, rpm_fraction):
    li = np.clip(load_fraction * 4, 0, 4)
    ri = np.clip(rpm_fraction  * 4, 0, 4)
    lo_l, lo_r = int(np.floor(li)), int(np.floor(ri))
    hi_l, hi_r = min(lo_l + 1, 4), min(lo_r + 1, 4)
    fl, fr = li - lo_l, ri - lo_r
    return float(
        BSFC_MAP[lo_l, lo_r] * (1-fl) * (1-fr) + BSFC_MAP[hi_l, lo_r] * fl * (1-fr) +
        BSFC_MAP[lo_l, hi_r] * (1-fl) * fr     + BSFC_MAP[hi_l, hi_r] * fl * fr
    )

def _select_gear(p, v_kmh):
    if v_kmh < 5:
        return p["rpm_idle"]
    v_ms   = v_kmh / 3.6
    rpm_lo = p["rpm_rated"] * 0.55
    rpm_hi = p["rpm_rated"] * 0.85
    best_rpm = p["rpm_idle"]
    for ratio in p["gear_ratios"]:
        rpm = v_ms * ratio * p["axle_ratio"] / p["tire_radius"] * 60 / (2 * np.pi)
        if rpm < p["rpm_max"] * 0.98:
            best_rpm = rpm
            if rpm_lo <= rpm <= rpm_hi:
                break
    return max(best_rpm, p["rpm_idle"])

def _engine_power_kW(p, v_kmh, slope_pct, mass_kg, a_ms2=0.0, wind_ms=0.0):
    v_ms  = v_kmh / 3.6
    F_net = (_aero_drag(p, v_ms, wind_ms) + _rolling_resistance(p, mass_kg, slope_pct)
             + _grade_force(mass_kg, slope_pct) + mass_kg * a_ms2)
    return max(F_net * v_ms / p["eta_transmission"] / 1000, 0.0)

def _fuel_l100(p, v_kmh, P_kW, rpm):
    lf = np.clip(P_kW / p["P_engine_kW"], 0, 1)
    rf = np.clip((rpm - p["rpm_idle"]) / (p["rpm_max"] - p["rpm_idle"]), 0, 1)
    fuel_lh = P_kW * _bsfc_lookup(lf, rf) / 1000 / DIESEL_DENSITY   # l/h

    if v_kmh < 5:
        # Parado o casi parado: convertir a l/100km equivalente a 30 km/h
        # para no contaminar la media de consumo en movimiento
        return float(np.clip(fuel_lh / 30 * 100, 5, 40))

    return float(np.clip(fuel_lh / v_kmh * 100, 5, 120))

def _engine_temp(T_prev, P_kW, P_max, T_ext, dt_min, p):
    lf       = np.clip(P_kW / P_max, 0, 1)
    T_target = p["temp_thermostat"] + lf * (p["temp_nominal"] - p["temp_thermostat"] + 5)
    tau      = 20.0 if T_prev < p["temp_thermostat"] else 8.0
    T_new    = T_prev + (dt_min / tau) * (T_target - T_prev)
    T_new   -= max(0, (10 - T_ext) * 0.02) * dt_min
    return float(np.clip(T_new, 40, 115))


# ── Perfil de velocidad realista ──────────────────────────────────────────────
def _speed_profile(T, p, slope, driving_style, rng):
    v_max  = p["v_max"]
    speed  = np.zeros(T, dtype=np.float32)
    t_cur  = 0
    consec = 0

    while t_cur < T:
        remaining = T - t_cur

        # Parada reglamentaria cada ~230-280 min (reglamento 561/2006)
        if consec >= rng.integers(230, 280):
            pause  = min(rng.integers(20, 35), remaining)
            speed[t_cur: t_cur + pause] = 0.0
            t_cur  += pause
            consec  = 0
            continue

        # Trucks larga distancia: 90% autopista
        r = rng.random()
        if r < 0.90:
            stype = "hw";  slen = rng.integers(50, 120)
        elif r < 0.97:
            stype = "sec"; slen = rng.integers(10, 30)
        else:
            stype = "urb"; slen = rng.integers(5, 12)
        slen = min(slen, remaining)

        sl_mean = slope[t_cur: t_cur + slen].mean() if slen > 0 else 0.0

        if stype == "hw":
            # Penalización de velocidad proporcional a la pendiente.
            # Para slope=6: grade_penalty = 6*5 + (6-3)*4 = 30+12 = 42 km/h
            # → v_c ≈ 83*0.93 - 42 ≈ 35 km/h (físicamente correcto en subida 6%)
            grade_penalty = abs(sl_mean) * 5.0 + max(0, abs(sl_mean) - 3) * 4.0
            v_c = v_max * rng.uniform(0.88, 0.98) - grade_penalty
            # Suelo: 40% de v_max (~36 km/h) — permite velocidades bajas en montaña.
            # NOTA: la línea antigua "max(v_c, v_max * 0.70)" ha sido eliminada
            # porque sobreescribía la penalización y causaba consumos de 120 l/100km.
            v_c = max(v_c, v_max * 0.40)

            seg = np.clip(
                v_c + rng.standard_normal(slen) * 3.0 * (0.5 + driving_style * 0.5),
                v_max * 0.35,   # antes 0.65 — reducido para permitir velocidades bajas en pendiente
                v_max,
            )
            ramp = np.minimum(np.arange(slen) / 8.0, 1.0)
            speed[t_cur: t_cur + slen] = (seg * ramp).astype(np.float32)

        elif stype == "sec":
            v_c = max(v_max * rng.uniform(0.55, 0.70) - abs(sl_mean) * 2, 25.0)
            seg = np.clip(v_c + rng.standard_normal(slen) * 3, 25, v_max * 0.72)
            speed[t_cur: t_cur + slen] = seg.astype(np.float32)

        else:  # urbano
            v_c = rng.uniform(25, 40)
            seg = np.clip(v_c + rng.standard_normal(slen) * 4, 0, 50)
            speed[t_cur: t_cur + slen] = seg.astype(np.float32)

        t_cur  += slen
        consec += slen

    # Suavizado kernel 7 — elimina picos de aceleración bruscos
    kernel = np.ones(3) / 3
    speed  = np.convolve(speed, kernel, mode="same")
    return np.clip(speed, 0, v_max).astype(np.float32)


# ── Generador principal ───────────────────────────────────────────────────────
def generate_trip(
    avg_slope:     float = 0.0,
    avg_temp:      float = 15.0,
    precipitation: float = 0.0,
    load_pct:      float = 0.7,
    driving_style: float = 0.5,
    vehicle_type:  int   = 0,
    noise_level:   float = 0.02,
    cold_start:    bool  = False,
    wind_kmh:      float = 0.0,   # positivo=frontal, negativo=trasero
    T:             int   = T,
    seed:          int   = None,
) -> np.ndarray:
    """
    Genera un viaje simulado con física real de vehículo pesado.
    Devuelve array (T, F) normalizado en [-1, 1].
    """
    rng = np.random.default_rng(seed)
    p   = VEHICLE_PROFILES[vehicle_type]

    # Masa total (con sloshing en cisterna)
    eff_load = load_pct
    if p["sloshing"] and 0.30 < load_pct < 0.80:
        eff_load = min(load_pct * (1 + 0.08 * np.sin(np.pi * (load_pct - 0.30) / 0.50)), 1.0)
    mass_kg = _total_mass(p, eff_load)

    # Pendiente: componente media + variación sinusoidal + ruido
    t_arr = np.linspace(0, 1, T)
    slope = (avg_slope
             + 1.2 * np.sin(2 * np.pi * t_arr * rng.uniform(0.8, 1.5))
             + 0.6 * rng.standard_normal(T)).astype(np.float32)
    slope = np.clip(slope, -20, 20)

    # Velocidad
    speed = _speed_profile(T, p, slope, driving_style, rng)

    # Temperatura exterior (curva diurna)
    hour     = np.linspace(7, 15, T)
    ext_temp = (avg_temp + 3.0 * np.sin(np.pi * (hour - 6) / 12)
                + rng.standard_normal(T) * 1.5).astype(np.float32)

    # Lluvia en rachas
    rain = np.zeros(T, dtype=np.float32)
    if precipitation > 0:
        for _ in range(rng.integers(1, 4)):
            s = rng.integers(0, T)
            l = rng.integers(20, 80)
            rain[s: s + l] = float(np.clip(precipitation * rng.uniform(0.5, 1.5), 0, 100))

    # Simulación paso a paso
    rpm_arr  = np.zeros(T, dtype=np.float32)
    fuel_arr = np.zeros(T, dtype=np.float32)
    temp_arr = np.zeros(T, dtype=np.float32)
    T_motor  = 40.0 if cold_start else p["temp_thermostat"] - 5.0

    for i in range(T):
        v = float(speed[i])
        s = float(slope[i])
        a = float((speed[i] - speed[i-1]) / 3.6 / 60) if i > 0 else 0.0

        P_kW    = _engine_power_kW(p, v, s, mass_kg, a, wind_ms=wind_kmh / 3.6)
        rpm     = _select_gear(p, v) * (1.0 + driving_style * 0.12)
        rpm     = float(np.clip(rpm, p["rpm_idle"], p["rpm_max"]))

        temp_factor = 1.0
        if ext_temp[i] > 25:
            temp_factor += (float(ext_temp[i]) - 25) * 0.004
        elif ext_temp[i] < 5:
            temp_factor += (5 - float(ext_temp[i])) * 0.006

        fuel    = _fuel_l100(p, v, P_kW, rpm) * temp_factor
        T_motor = _engine_temp(T_motor, P_kW, p["P_engine_kW"], float(ext_temp[i]), 1.0, p)

        # Ruido sensor + glitch CAN (0.5%)
        glitch      = 3.0 if rng.random() < 0.005 else 1.0
        rpm_arr[i]  = np.clip(rpm   + rng.standard_normal() * noise_level * 40  * glitch,
                               p["rpm_idle"], p["rpm_max"])
        fuel_arr[i] = np.clip(fuel  + rng.standard_normal() * noise_level * 1.0 * glitch, 5, 120)
        temp_arr[i] = np.clip(T_motor + rng.standard_normal() * noise_level * 0.8,        40, 115)

    # Media móvil de 5 pasos para suavizar picos de consumo en aceleración
    fuel_kernel = np.ones(5) / 5
    fuel_arr    = np.convolve(fuel_arr, fuel_kernel, mode="same").astype(np.float32)

    load_arr = np.clip(
        eff_load - np.linspace(0, eff_load * 0.02, T) + rng.standard_normal(T) * 0.005,
        0, 1,
    ).astype(np.float32)

    trip = np.stack([speed, fuel_arr, rpm_arr, temp_arr,
                     slope, ext_temp, rain, load_arr], axis=1)
    trip = np.clip(2 * (trip - MINS) / (MAXS - MINS) - 1, -1, 1)
    return trip.astype(np.float32)


def denormalize(trips: np.ndarray) -> np.ndarray:
    """Desnormaliza de [-1, 1] a valores reales. trips: (N, T, F) o (T, F)."""
    return (trips + 1) / 2 * (MAXS - MINS) + MINS


def generate_conditioning_vector(
    avg_slope:     float = 0.0,
    avg_temp:      float = 15.0,
    precipitation: float = 0.0,
    load_pct:      float = 0.7,
    vehicle_type:  int   = 0,
    day_of_week:   int   = 0,
    driving_style: float = 0.5,
) -> np.ndarray:
    vehicle_onehot               = np.zeros(3, dtype=np.float32)
    vehicle_onehot[vehicle_type] = 1.0
    dow_onehot                   = np.zeros(7, dtype=np.float32)
    dow_onehot[day_of_week]      = 1.0
    c = np.array([
        np.clip(avg_slope    / 10.0, -1, 1),
        np.clip(avg_temp     / 40.0, -1, 1),
        np.clip(precipitation / 20.0, 0, 1),
        np.clip(load_pct,             0, 1),
        np.clip(driving_style,        0, 1),
    ], dtype=np.float32)
    c = np.concatenate([c, vehicle_onehot, dow_onehot])
    c = np.pad(c, (0, C_DIM - len(c)))
    return c.astype(np.float32)


# ── Dataset PyTorch ────────────────────────────────────────────────────────────
class TripDataset(Dataset):
    def __init__(self, n_trips: int = 2000, seed: int = 42):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.trips, self.conditions = [], []

        for _ in range(n_trips):
            avg_slope     = rng.uniform(-6,  6)
            avg_temp      = rng.uniform(-15, 38)
            precipitation = rng.uniform(0,   15)
            load_pct      = rng.uniform(0.2, 1.0)
            driving_style = rng.uniform(0,   1)
            vehicle_type  = int(rng.integers(0, 3))
            day_of_week   = int(rng.integers(0, 7))
            cold_start    = rng.random() < 0.15

            trip = generate_trip(
                avg_slope=avg_slope, avg_temp=avg_temp,
                precipitation=precipitation, load_pct=load_pct,
                driving_style=driving_style, vehicle_type=vehicle_type,
                cold_start=cold_start, seed=int(rng.integers(0, 2**31)),
            )
            c = generate_conditioning_vector(
                avg_slope=avg_slope, avg_temp=avg_temp,
                precipitation=precipitation, load_pct=load_pct,
                vehicle_type=vehicle_type, day_of_week=day_of_week,
                driving_style=driving_style,
            )

            self.trips.append(torch.tensor(trip))
            self.conditions.append(torch.tensor(c))

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        return self.trips[idx], self.conditions[idx]


def get_dataloaders(n_trips: int = 2000, batch_size: int = 64, seed: int = 42):
    dataset  = TripDataset(n_trips=n_trips, seed=seed)
    n_val    = int(0.2 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
    )