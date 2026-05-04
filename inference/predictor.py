"""
inference/predictor.py
Módulo de inferencia: genera viajes sintéticos y calcula intervalos de confianza
por tramo para una ruta nueva.
"""

import numpy as np
import torch
from model.nflow_model import ConditionalFlowModel

# ── IMPORTANTE: MINS, MAXS y VAR_NAMES se importan desde synthetic.py ─────────
# Nunca duplicar estos rangos aquí. Si cambian en synthetic.py (p.ej. al añadir
# nuevas variables o ajustar rangos), el predictor los recoge automáticamente.
from data.synthetic import generate_conditioning_vector, T, F, MINS, MAXS, VAR_NAMES

N_TRAMOS = 6


def denormalize(trips: np.ndarray) -> np.ndarray:
    """Convierte trips normalizados [-1,1] a valores reales."""
    return (trips + 1) / 2 * (MAXS - MINS) + MINS


def compute_segments(trips: np.ndarray, n_segments: int = N_TRAMOS) -> list[dict]:
    """
    Divide los viajes en tramos y calcula estadísticos por tramo.

    Args:
        trips: (n_samples, T, F) normalizados
        n_segments: número de tramos

    Returns:
        Lista de dicts con estadísticos P5/P50/P95 por tramo y variable
    """
    trips_real   = denormalize(trips)
    segment_len  = T // n_segments
    segments     = []

    for i in range(n_segments):
        start = i * segment_len
        end   = (i + 1) * segment_len if i < n_segments - 1 else T

        seg_means = trips_real[:, start:end, :].mean(axis=1)  # (n_samples, F)

        seg_info = {"tramo": i + 1, "inicio_min": start, "fin_min": end}
        for f_idx, var_name in enumerate(VAR_NAMES):
            vals = seg_means[:, f_idx]
            seg_info[var_name] = {
                "p5":    float(np.percentile(vals, 5)),
                "p50":   float(np.percentile(vals, 50)),
                "p95":   float(np.percentile(vals, 95)),
                "media": float(vals.mean()),
                "std":   float(vals.std()),
            }
        segments.append(seg_info)

    return segments


class FleetPredictor:
    """
    Interfaz de alto nivel para predecir perfiles de viaje en rutas nuevas.

    Uso:
        predictor = FleetPredictor(model)
        result = predictor.predict_route(avg_slope=2.0, avg_temp=10.0, ...)
        predictor.print_report(result)
    """

    def __init__(self, model: ConditionalFlowModel, device: str = None):
        self.model  = model
        self.device = torch.device(device if device else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def predict_route(
        self,
        avg_slope:    float = 0.0,
        avg_temp:     float = 15.0,
        precipitation: float = 0.0,
        load_pct:     float = 0.7,
        vehicle_type: int   = 0,
        day_of_week:  int   = 0,
        n_samples:    int   = 100,
        n_segments:   int   = N_TRAMOS,
        wind_kmh:     float = 0.0,   # positivo=frontal, negativo=trasero
    ) -> dict:
        """
        Genera n_samples viajes sintéticos para la ruta especificada
        y devuelve los intervalos de confianza por tramo.
        """
        rng          = np.random.default_rng()
        n_styles     = 10
        per_style    = n_samples // n_styles   # 10 con n_samples=100
        driving_styles = rng.uniform(0, 1, n_styles)
        all_trips    = []
        for ds in driving_styles:
            c = generate_conditioning_vector(
                avg_slope=avg_slope,
                avg_temp=avg_temp,
                precipitation=precipitation,
                load_pct=load_pct,
                vehicle_type=vehicle_type,
                day_of_week=day_of_week,
                driving_style=float(ds),
            )
            c_tensor = torch.tensor(c)
            batch    = self.model.sample(c_tensor, n_samples=per_style)
            all_trips.append(batch.cpu().numpy())
        trips_np = np.concatenate(all_trips, axis=0)  # (n_samples, T, F)

        # ── Corrección de viento (sin reentrenar) ────────────────────────────
        # A velocidad de autopista, la aerodinámica supone ~28% del consumo.
        # Corregimos solo esa fracción según la velocidad efectiva con viento.
        if wind_kmh != 0.0:
            AERO_FRACTION = 0.28
            trips_real_w = denormalize(trips_np.copy())
            v  = np.maximum(trips_real_w[:, :, 0], 10.0)   # velocidad km/h
            c  = trips_real_w[:, :, 1]                      # consumo l/100km
            wind_factor = 1.0 + AERO_FRACTION * (
                (v + wind_kmh) ** 2 - v ** 2
            ) / v ** 2
            trips_real_w[:, :, 1] = np.clip(c * wind_factor, 5, 120)
            trips_np = np.clip(
                2 * (trips_real_w - MINS) / (MAXS - MINS) - 1, -1, 1
            ).astype(np.float32)

        segments   = compute_segments(trips_np, n_segments=n_segments)
        trips_real = denormalize(trips_np)

        total_consumption = trips_real[:, :, 1].mean(axis=1)
        total_speed       = trips_real[:, :, 0].mean(axis=1)

        summary = {
            "consumo_medio_l100km": {
                "p5":  float(np.percentile(total_consumption, 5)),
                "p50": float(np.percentile(total_consumption, 50)),
                "p95": float(np.percentile(total_consumption, 95)),
            },
            "velocidad_media_kmh": {
                "p5":  float(np.percentile(total_speed, 5)),
                "p50": float(np.percentile(total_speed, 50)),
                "p95": float(np.percentile(total_speed, 95)),
            },
            "n_muestras":       n_samples,
            "duracion_minutos": T,
        }

        return {
            "trips_raw": trips_np,
            "segments":  segments,
            "summary":   summary,
            "condition": c,
        }

    def print_report(self, result: dict):
        s = result["summary"]
        print("\n" + "═" * 60)
        print("  INFORME DE RUTA — cNSFfleet Predictor")
        print("═" * 60)
        print(f"  Muestras generadas : {s['n_muestras']}")
        print(f"  Duración estimada  : {s['duracion_minutos']} min")
        print()
        print("  CONSUMO MEDIO (l/100km)")
        print(f"    P05: {s['consumo_medio_l100km']['p5']:.1f}")
        print(f"    P50: {s['consumo_medio_l100km']['p50']:.1f}   ← estimación central")
        print(f"    P95: {s['consumo_medio_l100km']['p95']:.1f}")
        print()
        print("  VELOCIDAD MEDIA (km/h)")
        print(f"    P05: {s['velocidad_media_kmh']['p5']:.1f}")
        print(f"    P50: {s['velocidad_media_kmh']['p50']:.1f}")
        print(f"    P95: {s['velocidad_media_kmh']['p95']:.1f}")
        print()
        print("  DETALLE POR TRAMOS")
        print(f"  {'Tramo':<8} {'Consumo P50':>12} {'IC 90%':>22} {'Vel P50':>10}")
        print("  " + "-" * 56)
        for seg in result["segments"]:
            c_p5  = seg["consumo"]["p5"]
            c_p50 = seg["consumo"]["p50"]
            c_p95 = seg["consumo"]["p95"]
            v_p50 = seg["velocidad"]["p50"]
            print(
                f"  {seg['tramo']:<8} "
                f"{c_p50:>10.1f}   "
                f"[{c_p5:.1f} – {c_p95:.1f}]"
                f"{v_p50:>10.1f}"
            )
        print("═" * 60 + "\n")