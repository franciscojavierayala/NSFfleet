"""
anomaly/filter.py
Filtro de anomalías para viajes reales entrantes antes del reentrenamiento.

Dos etapas:
  1. Validación física: valores fuera de rango físico → descarte inmediato
  2. Isolation Forest: outlier estadístico respecto al pool de viajes conocidos

Mejoras para datos reales vs sintéticos:
  - Soporta TripScaler dinámico (desnormaliza correctamente con rangos reales)
  - Contamination configurable según calidad del dataset
  - Diagnóstico detallado: qué variable disparó la anomalía y por qué
  - Guarda y carga el filtro entrenado (persistencia entre sesiones)
  - Estadísticas del batch filtrado
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from data.synthetic import T, F, VAR_NAMES, MINS, MAXS


# ── Rangos físicos válidos (valores reales post-desnormalización) ─────────────
PHYSICAL_LIMITS = {
    "velocidad":     (0,   160),
    "consumo":       (5,   120),
    "RPM":           (400, 4000),
    "temp_motor":    (40,  130),
    "pendiente":     (-25,  25),
    "temp_ext":      (-40,  60),
    "precipitacion": (0,   100),
    "carga":         (0,     1.05),
}

# Umbrales de alerta (más estrictos que los físicos — para datos operacionales)
OPERATIONAL_LIMITS = {
    "velocidad":     (0,   105),   # > 105 km/h en camión → alerta (aunque físicamente posible)
    "consumo":       (8,    90),
    "RPM":           (500, 3200),
    "temp_motor":    (55,  110),
    "pendiente":     (-20,  20),
    "temp_ext":      (-35,  55),
    "precipitacion": (0,    80),
    "carga":         (0,     1.02),
}

# Causas posibles por variable (para mensajes de diagnóstico)
ANOMALY_CAUSES = {
    "velocidad": {
        "low":  "Vehículo parado durante demasiado tiempo o sensor GPS sin señal",
        "high": "Velocidad imposible para un camión — posible error de GPS o sensor",
    },
    "consumo": {
        "low":  "Consumo negativo o nulo — sensor de combustible defectuoso",
        "high": "Consumo extremo — posible avería mecánica, fuga de combustible o dato corrupto",
    },
    "RPM": {
        "low":  "RPM por debajo del ralentí — motor apagado o sensor desconectado",
        "high": "RPM excesivas — posible fallo de transmisión o dato corrupto",
    },
    "temp_motor": {
        "low":  "Temperatura de motor imposible — sensor de temperatura defectuoso",
        "high": "Sobrecalentamiento del motor — riesgo de avería grave",
    },
    "pendiente": {
        "low":  "Pendiente negativa extrema — posible error en altimetría o GPS",
        "high": "Pendiente positiva extrema — posible error en altimetría o GPS",
    },
    "temp_ext": {
        "low":  "Temperatura exterior imposible para condiciones de operación",
        "high": "Temperatura exterior extrema — fuera del rango de operación normal",
    },
    "precipitacion": {
        "low":  "Valor negativo imposible — sensor de lluvia defectuoso",
        "high": "Precipitación extrema — verificar sensor meteorológico",
    },
    "carga": {
        "low":  "Carga negativa imposible — sensor de peso defectuoso",
        "high": "Carga superior a la máxima permitida — posible sobrecarga ilegal",
    },
}


# ── Desnormalización compatible con scaler real o sintético ──────────────────
def _denormalize(trip_norm: np.ndarray, scaler=None) -> np.ndarray:
    """
    Desnormaliza un viaje de [-1, 1] a valores reales.
    Si se proporciona un TripScaler lo usa; si no, usa los rangos sintéticos.
    """
    if scaler is not None:
        return scaler.inverse_transform(trip_norm)
    return (trip_norm + 1) / 2 * (MAXS - MINS) + MINS


# ── Extracción de features para Isolation Forest ──────────────────────────────
def extract_features(trip: np.ndarray, scaler=None) -> np.ndarray:
    """
    Extrae un vector de features estadísticas enriquecido de un viaje (T, F).
    Incluye media, std, percentiles, tendencia y correlaciones clave.
    """
    trip_real = _denormalize(trip, scaler)

    features = []
    for f_idx in range(F):
        col = trip_real[:, f_idx]
        features.extend([
            col.mean(),
            col.std(),
            np.percentile(col, 5),
            np.percentile(col, 25),
            np.percentile(col, 75),
            np.percentile(col, 95),
            col.max() - col.min(),          # rango total
            np.diff(col).std(),             # variabilidad temporal
        ])

    # Features cruzadas clave (correlaciones físicas esperadas)
    vel  = trip_real[:, 0]
    cons = trip_real[:, 1]
    rpm  = trip_real[:, 2]
    pend = trip_real[:, 4]

    def _safe_corr(a, b):
        if a.std() < 1e-6 or b.std() < 1e-6:
            return 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(a, b)[0, 1]
        return float(r) if np.isfinite(r) else 0.0

    features.extend([
        _safe_corr(vel,  cons),
        _safe_corr(vel,  rpm),
        _safe_corr(pend, cons),
    ])

    return np.array(features, dtype=np.float32)


# ── Clase principal ────────────────────────────────────────────────────────────
class AnomalyFilter:
    """
    Filtro de dos etapas para viajes reales entrantes.

    Etapa 1: Validación física (reglas hard-coded por variable)
    Etapa 2: Isolation Forest estadístico sobre el pool de referencia

    Args:
        contamination:  fracción esperada de outliers (0.03 para datos reales,
                        0.05 para datos sintéticos)
        use_operational: si True, usa límites operacionales más estrictos
                         además de los físicos
        scaler:         TripScaler para desnormalizar con rangos reales
                        (None = usar rangos sintéticos por defecto)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        use_operational: bool = False,
        scaler=None,
    ):
        self.contamination     = contamination
        self.use_operational   = use_operational
        self.scaler            = scaler
        self.iso_forest        = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=150,
            max_samples="auto",
        )
        self.is_fitted         = False
        self._feature_baseline = None  # estadísticas del pool para diagnóstico

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    def fit(self, trips: list[np.ndarray]):
        """
        Entrena el Isolation Forest con viajes conocidos como válidos.
        trips: lista de arrays (T, F) normalizados
        """
        features = np.stack([extract_features(t, self.scaler) for t in trips])
        self.iso_forest.fit(features)
        self.is_fitted = True

        # Guardar estadísticas del pool para diagnóstico contextual
        self._feature_baseline = {
            "mean": features.mean(axis=0),
            "std":  features.std(axis=0),
        }
        print(f"AnomalyFilter entrenado con {len(trips)} viajes de referencia.")
        print(f"  contamination={self.contamination} | "
              f"límites={'operacionales' if self.use_operational else 'físicos'}")

    # ── Evaluación de un viaje ────────────────────────────────────────────────
    def check(self, trip: np.ndarray) -> tuple[str, str, dict]:
        """
        Evalúa un viaje entrante.

        Returns:
            ("VALID" | "ANOMALY", motivo, detalles_dict)
        """
        trip_real = _denormalize(trip, self.scaler)
        limits    = OPERATIONAL_LIMITS if self.use_operational else PHYSICAL_LIMITS
        details   = {}

        # ── Etapa 1: Validación física por variable ──────────────────────────
        for f_idx, var_name in enumerate(VAR_NAMES):
            col    = trip_real[:, f_idx]
            lo, hi = limits[var_name]

            pct_low  = float(np.mean(col < lo))
            pct_high = float(np.mean(col > hi))
            pct_out  = pct_low + pct_high

            if pct_out > 0.05:
                direction = "low" if pct_low > pct_high else "high"
                cause     = ANOMALY_CAUSES.get(var_name, {}).get(direction, "Valor fuera de rango")
                reason    = (
                    f"'{var_name}': {pct_out*100:.1f}% de pasos fuera de "
                    f"[{lo}, {hi}] — {cause}"
                )
                details["variable"]   = var_name
                details["pct_out"]    = pct_out
                details["direction"]  = direction
                details["cause"]      = cause
                return "ANOMALY", reason, details

        # ── Etapa 2: Isolation Forest ─────────────────────────────────────────
        if self.is_fitted:
            feats = extract_features(trip, self.scaler).reshape(1, -1)
            pred  = self.iso_forest.predict(feats)[0]
            score = float(self.iso_forest.score_samples(feats)[0])

            if pred == -1:
                # Diagnóstico: qué feature se desvía más del baseline
                if self._feature_baseline is not None:
                    z_scores = np.abs(
                        (feats[0] - self._feature_baseline["mean"])
                        / (self._feature_baseline["std"] + 1e-6)
                    )
                    worst_feat_idx = int(np.argmax(z_scores))
                    # Mapear índice de feature a variable (8 features por variable)
                    var_idx        = worst_feat_idx // 8
                    var_name       = VAR_NAMES[var_idx] if var_idx < F else "correlación"
                    z_val          = float(z_scores[worst_feat_idx])

                    cause = (
                        f"Estadística anómala en '{var_name}' "
                        f"(z-score={z_val:.1f}). "
                        "Posible avería, conducción atípica o ruta inusual."
                    )
                else:
                    cause = "Patrón estadístico fuera del rango normal del pool de referencia."

                details["score"]   = score
                details["cause"]   = cause
                details["z_score"] = z_val if self._feature_baseline is not None else None
                return "ANOMALY", f"Outlier estadístico (score={score:.3f}). {cause}", details

        return "VALID", "", {}

    # ── Filtrado de lote ──────────────────────────────────────────────────────
    def filter_batch(
        self,
        trips: list[np.ndarray],
        verbose: bool = True,
    ) -> tuple[list, list]:
        """
        Filtra un lote de viajes entrantes.

        Returns:
            (viajes_validos, viajes_descartados)
            viajes_descartados: lista de dicts con keys trip, reason, details
        """
        valid, discarded = [], []

        for i, trip in enumerate(trips):
            label, reason, details = self.check(trip)
            if label == "VALID":
                valid.append(trip)
            else:
                discarded.append({"trip": trip, "reason": reason, "details": details})
                if verbose:
                    print(f"  [DESCARTADO] Viaje {i}: {reason}")

        if verbose:
            pct_valid = len(valid) / len(trips) * 100 if trips else 0
            print(f"\nResumen del filtrado:")
            print(f"  ✅ Válidos   : {len(valid):3d} ({pct_valid:.0f}%)")
            print(f"  ❌ Descartados: {len(discarded):3d} ({100-pct_valid:.0f}%)")
            print(f"  Total        : {len(trips):3d}")

            if discarded:
                print("\nCausas de descarte:")
                causes = {}
                for d in discarded:
                    cause = d["details"].get("cause", "desconocido")[:60]
                    causes[cause] = causes.get(cause, 0) + 1
                for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
                    print(f"  • {cause} ({count}x)")

        return valid, discarded

    # ── Persistencia ──────────────────────────────────────────────────────────
    def save(self, path: str = "checkpoints/anomaly_filter.pkl"):
        """Guarda el filtro entrenado en disco."""
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"AnomalyFilter guardado en: {path}")

    @classmethod
    def load(cls, path: str = "checkpoints/anomaly_filter.pkl") -> "AnomalyFilter":
        """Carga un filtro previamente entrenado."""
        with open(path, "rb") as f:
            af = pickle.load(f)
        print(f"AnomalyFilter cargado desde: {path} "
              f"(contamination={af.contamination}, fitted={af.is_fitted})")
        return af
