"""
data/real_dataset.py
Pipeline de preprocesamiento para datos reales de telemetría de camiones.

Formato de entrada esperado (CSV o Parquet):
  Columnas obligatorias:
    - timestamp        : datetime o string ISO8601
    - velocidad        : km/h
    - consumo          : l/100km
    - rpm              : RPM del motor
    - temp_motor       : °C
    - pendiente        : % (positivo = subida)
    - temp_ext         : °C temperatura exterior
    - precipitacion    : mm/h
    - carga            : 0.0–1.0 (porcentaje de carga máxima)

  Columnas opcionales pero recomendadas:
    - vehicle_id       : identificador del vehículo
    - vehicle_type     : 0=tractor, 1=rigido, 2=cisterna
    - trip_id          : identificador del viaje

Uso básico:
    from data.real_dataset import RealTripDataset, get_real_dataloaders, build_scaler

    # 1. Construir y guardar el scaler desde los datos de entrenamiento
    build_scaler("data/trips_train.parquet", "checkpoints/scaler.json")

    # 2. Obtener dataloaders
    train_loader, val_loader = get_real_dataloaders(
        train_path="data/trips_train.parquet",
        val_path="data/trips_val.parquet",   # opcional
        scaler_path="checkpoints/scaler.json",
    )

    # 3. Usar con el Trainer existente sin cambios
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(epochs=50)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from data.synthetic import T, F, C_DIM, generate_conditioning_vector


# ── Nombres de columnas esperadas ─────────────────────────────────────────────
REQUIRED_COLS = [
    "timestamp", "velocidad", "consumo", "rpm",
    "temp_motor", "pendiente", "temp_ext", "precipitacion", "carga",
]

FEATURE_COLS = [
    "velocidad", "consumo", "rpm", "temp_motor",
    "pendiente", "temp_ext", "precipitacion", "carga",
]

# Rangos físicos para validación (valores imposibles = sensor roto)
PHYSICAL_LIMITS = {
    "velocidad":     (0,    160),
    "consumo":       (5,    120),
    "rpm":           (400,  4000),
    "temp_motor":    (40,   130),
    "pendiente":     (-25,  25),
    "temp_ext":      (-40,  60),
    "precipitacion": (0,    100),
    "carga":         (0,    1.05),
}

# Mapeo flexible de nombres de columna (para datasets con nombres distintos)
COLUMN_ALIASES = {
    "speed":          "velocidad",
    "speed_kmh":      "velocidad",
    "fuel":           "consumo",
    "fuel_consumption": "consumo",
    "engine_rpm":     "rpm",
    "engine_temp":    "temp_motor",
    "coolant_temp":   "temp_motor",
    "slope":          "pendiente",
    "grade":          "pendiente",
    "outside_temp":   "temp_ext",
    "ambient_temp":   "temp_ext",
    "rain":           "precipitacion",
    "rainfall":       "precipitacion",
    "load":           "carga",
    "payload_pct":    "carga",
}


# ── Scaler dinámico ────────────────────────────────────────────────────────────
class TripScaler:
    """
    Normaliza viajes reales al rango [-1, 1] usando percentiles robustos.
    Usar percentil 1 y 99 en vez de min/max para ignorar outliers extremos.

    El scaler se guarda en JSON junto al checkpoint del modelo para garantizar
    que entrenamiento e inferencia usen exactamente la misma normalización.
    """

    def __init__(self):
        self.mins: np.ndarray | None = None
        self.maxs: np.ndarray | None = None
        self.feature_cols = FEATURE_COLS

    def fit(self, trips: list[np.ndarray]):
        """
        Calcula mins/maxs a partir de una lista de viajes (T, F).
        Usa percentil 1 y 99 para robustez ante outliers.
        """
        all_data = np.concatenate(trips, axis=0)  # (N*T, F)
        self.mins = np.percentile(all_data, 1, axis=0).astype(np.float32)
        self.maxs = np.percentile(all_data, 99, axis=0).astype(np.float32)

        # Evitar divisiones por cero en variables constantes
        delta = self.maxs - self.mins
        delta[delta < 1e-6] = 1.0
        self.maxs = self.mins + delta

        print("Scaler ajustado:")
        for i, col in enumerate(self.feature_cols):
            print(f"  {col:15s}: [{self.mins[i]:.2f}, {self.maxs[i]:.2f}]")

    def transform(self, trip: np.ndarray) -> np.ndarray:
        """Normaliza un viaje (T, F) a [-1, 1]."""
        if self.mins is None:
            raise RuntimeError("Scaler no ajustado. Llama a fit() primero.")
        return (2 * (trip - self.mins) / (self.maxs - self.mins) - 1).astype(np.float32)

    def inverse_transform(self, trip: np.ndarray) -> np.ndarray:
        """Desnormaliza un viaje de [-1, 1] a valores reales."""
        return ((trip + 1) / 2 * (self.maxs - self.mins) + self.mins).astype(np.float32)

    def save(self, path: str):
        """Guarda el scaler en JSON."""
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "mins": self.mins.tolist(),
                "maxs": self.maxs.tolist(),
                "feature_cols": self.feature_cols,
            }, f, indent=2)
        print(f"Scaler guardado en: {path}")

    @classmethod
    def load(cls, path: str) -> "TripScaler":
        """Carga el scaler desde JSON."""
        with open(path) as f:
            data = json.load(f)
        scaler = cls()
        scaler.mins = np.array(data["mins"], dtype=np.float32)
        scaler.maxs = np.array(data["maxs"], dtype=np.float32)
        scaler.feature_cols = data["feature_cols"]
        return scaler


# ── Carga y limpieza de datos ─────────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    """
    Carga un CSV o Parquet y estandariza los nombres de columna.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        raise ValueError(f"Formato no soportado: {path.suffix}. Usa CSV o Parquet.")

    # Normalizar nombres de columna a minúsculas
    df.columns = df.columns.str.lower().str.strip()

    # Aplicar aliases
    df.rename(columns=COLUMN_ALIASES, inplace=True)

    # Verificar columnas obligatorias
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas obligatorias faltantes: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}\n"
            f"Puedes añadir aliases en COLUMN_ALIASES si tus columnas tienen otro nombre."
        )

    # Parsear timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def clean_trip(df: pd.DataFrame, max_pct_invalid: float = 0.15) -> pd.DataFrame | None:
    """
    Limpia un único viaje:
      1. Elimina filas con NaN en columnas de telemetría
      2. Verifica que no más del max_pct_invalid de filas están fuera de rango físico
      3. Interpola gaps pequeños (hasta 5 min)

    Returns None si el viaje tiene demasiados datos corruptos.
    """
    df = df.copy()

    # Convertir a numérico
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Contar valores fuera de rango físico
    n_invalid = 0
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col in df.columns:
            out = (df[col] < lo) | (df[col] > hi)
            n_invalid += out.sum()
            df.loc[out, col] = np.nan  # marcar como NaN para interpolación

    pct_invalid = n_invalid / (len(df) * F)
    if pct_invalid > max_pct_invalid:
        return None  # demasiado corrupto

    # Interpolación lineal para gaps pequeños
    df[FEATURE_COLS] = df[FEATURE_COLS].interpolate(
        method="linear", limit=5, limit_direction="both"
    )

    # Eliminar filas aún con NaN
    df = df.dropna(subset=FEATURE_COLS)

    return df if len(df) > T // 4 else None  # mínimo 25% del T esperado


def resample_to_1min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples los datos a resolución 1 minuto usando interpolación.
    """
    df = df.set_index("timestamp")
    df_numeric = df[FEATURE_COLS]
    df_resampled = df_numeric.resample("1min").mean()
    df_resampled = df_resampled.interpolate(method="time", limit=10)
    df_resampled = df_resampled.dropna()
    df_resampled = df_resampled.reset_index()
    return df_resampled


def segment_trip(df: pd.DataFrame, window_size: int = T, stride: int = T // 2) -> list[np.ndarray]:
    """
    Divide un viaje largo en ventanas de tamaño window_size.
    Stride = T//2 significa 50% de solapamiento entre ventanas.

    Args:
        df:          DataFrame con columnas FEATURE_COLS, resolución 1min
        window_size: pasos temporales por ventana (default=T=480)
        stride:      desplazamiento entre ventanas

    Returns:
        Lista de arrays (window_size, F)
    """
    values = df[FEATURE_COLS].values.astype(np.float32)
    segments = []

    for start in range(0, len(values) - window_size + 1, stride):
        segment = values[start: start + window_size]
        if len(segment) == window_size:
            segments.append(segment)

    # Si el viaje es más corto que window_size pero suficientemente largo,
    # hacer padding con el último valor conocido
    if len(segments) == 0 and len(values) >= window_size // 2:
        padded = np.pad(
            values,
            ((0, window_size - len(values)), (0, 0)),
            mode="edge"
        )
        segments.append(padded[:window_size])

    return segments


def extract_conditioning(df: pd.DataFrame, vehicle_type: int = 0) -> np.ndarray:
    """
    Construye el vector de condicionamiento c desde un DataFrame de viaje real.
    """
    # Inferir día de la semana desde el timestamp de inicio
    day_of_week = 0
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"].iloc[0])
        day_of_week = ts.dayofweek  # 0=lunes

    avg_slope     = float(df["pendiente"].mean())
    avg_temp      = float(df["temp_ext"].mean())
    precipitation = float(df["precipitacion"].mean())
    load_pct      = float(df["carga"].mean())

    return generate_conditioning_vector(
        avg_slope=avg_slope,
        avg_temp=avg_temp,
        precipitation=precipitation,
        load_pct=load_pct,
        vehicle_type=vehicle_type,
        day_of_week=day_of_week,
    )


# ── Dataset PyTorch con datos reales ──────────────────────────────────────────
class RealTripDataset(Dataset):
    """
    Dataset PyTorch para viajes reales de telemetría.
    Compatible con el Trainer existente (devuelve los mismos tipos que TripDataset).

    Args:
        path:            ruta al CSV/Parquet
        scaler:          TripScaler ya ajustado (fit) o None para ajustar aquí
        window_size:     pasos temporales por ventana
        stride:          solapamiento entre ventanas
        trip_id_col:     columna que identifica cada viaje (None = todo es un viaje)
        vehicle_type_col: columna con el tipo de vehículo (None = asumir 0)
        max_pct_invalid: máximo % de datos corruptos tolerado por viaje
        verbose:         imprimir estadísticas de carga
    """

    def __init__(
        self,
        path: str,
        scaler: TripScaler | None = None,
        window_size: int = T,
        stride: int | None = None,
        trip_id_col: str | None = "trip_id",
        vehicle_type_col: str | None = "vehicle_type",
        max_pct_invalid: float = 0.15,
        verbose: bool = True,
    ):
        super().__init__()
        self.stride = stride or window_size // 2

        df_raw = load_raw(path)

        # Separar viajes
        if trip_id_col and trip_id_col in df_raw.columns:
            trip_groups = [g for _, g in df_raw.groupby(trip_id_col)]
        else:
            warnings.warn(
                f"Columna '{trip_id_col}' no encontrada. "
                "Tratando todo el archivo como un único viaje largo."
            )
            trip_groups = [df_raw]

        # Procesar cada viaje
        all_raw_segments = []
        all_conditions   = []
        n_discarded = 0

        for trip_df in trip_groups:
            # Tipo de vehículo del viaje
            v_type = 0
            if vehicle_type_col and vehicle_type_col in trip_df.columns:
                v_type = int(trip_df[vehicle_type_col].mode()[0])

            # Limpiar
            trip_clean = clean_trip(trip_df, max_pct_invalid)
            if trip_clean is None:
                n_discarded += 1
                continue

            # Resamplear a 1 min
            if "timestamp" in trip_clean.columns:
                trip_1min = resample_to_1min(trip_clean)
            else:
                trip_1min = trip_clean  # ya está a 1 min

            # Segmentar en ventanas
            segments = segment_trip(trip_1min, window_size, self.stride)
            if not segments:
                n_discarded += 1
                continue

            # Vector de condicionamiento del viaje completo
            c = extract_conditioning(trip_1min, v_type)

            for seg in segments:
                all_raw_segments.append(seg)
                all_conditions.append(c)

        if verbose:
            print(f"\nDataset real cargado desde: {path}")
            print(f"  Viajes procesados : {len(trip_groups) - n_discarded}")
            print(f"  Viajes descartados: {n_discarded}")
            print(f"  Ventanas totales  : {len(all_raw_segments)}")

        if len(all_raw_segments) == 0:
            raise ValueError(
                "No se pudieron extraer ventanas válidas del dataset. "
                "Revisa el formato del CSV y los rangos físicos en PHYSICAL_LIMITS."
            )

        # Ajustar scaler si no se proporcionó
        if scaler is None:
            self.scaler = TripScaler()
            self.scaler.fit(all_raw_segments)
        else:
            self.scaler = scaler

        # Normalizar y convertir a tensores
        self.trips      = []
        self.conditions = []
        for seg, c in zip(all_raw_segments, all_conditions):
            seg_norm = self.scaler.transform(seg)
            self.trips.append(torch.tensor(seg_norm))
            self.conditions.append(torch.tensor(c))

        if verbose:
            print(f"  Listo. {len(self.trips)} muestras para entrenamiento.\n")

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        return self.trips[idx], self.conditions[idx]


# ── Funciones de conveniencia ─────────────────────────────────────────────────
def build_scaler(train_path: str, scaler_path: str = "checkpoints/scaler.json") -> TripScaler:
    """
    Construye y guarda el scaler desde los datos de entrenamiento.
    Llamar UNA SOLA VEZ antes de entrenar. Nunca recalcular con datos de validación.
    """
    print("Construyendo scaler desde datos de entrenamiento...")
    df = load_raw(train_path)

    trip_id_col = "trip_id" if "trip_id" in df.columns else None
    if trip_id_col:
        groups = [g for _, g in df.groupby(trip_id_col)]
    else:
        groups = [df]

    raw_segments = []
    for trip_df in groups:
        cleaned = clean_trip(trip_df)
        if cleaned is None:
            continue
        if "timestamp" in cleaned.columns:
            cleaned = resample_to_1min(cleaned)
        segments = segment_trip(cleaned)
        raw_segments.extend(segments)

    scaler = TripScaler()
    scaler.fit(raw_segments)
    scaler.save(scaler_path)
    return scaler


def get_real_dataloaders(
    train_path: str,
    val_path: str | None = None,
    scaler_path: str = "checkpoints/scaler.json",
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Devuelve train_loader y val_loader listos para el Trainer.

    Si val_path es None, hace split automático 80/20 del train_path.
    El scaler se construye desde train_path si no existe ya en scaler_path.

    Uso:
        train_loader, val_loader = get_real_dataloaders(
            train_path="data/mis_viajes.parquet",
            scaler_path="checkpoints/scaler.json",
        )
    """
    # Cargar o construir scaler
    scaler_file = Path(scaler_path)
    if scaler_file.exists():
        print(f"Cargando scaler existente: {scaler_path}")
        scaler = TripScaler.load(scaler_path)
    else:
        print(f"Scaler no encontrado, construyendo desde: {train_path}")
        scaler = build_scaler(train_path, scaler_path)

    # Dataset completo de entrenamiento
    train_dataset = RealTripDataset(
        path=train_path,
        scaler=scaler,
        verbose=True,
    )

    if val_path:
        val_dataset = RealTripDataset(
            path=val_path,
            scaler=scaler,  # IMPORTANTE: mismo scaler que train
            verbose=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    else:
        # Split automático
        n_val   = int(val_split * len(train_dataset))
        n_train = len(train_dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader


# ── Utilidad: convertir dataset ETS2 al formato esperado ──────────────────────
def convert_ets2_telemetry(input_path: str, output_path: str):
    """
    Convierte un CSV de telemetría ETS2 (formato SDK community) al formato
    estándar esperado por RealTripDataset.

    Columnas ETS2 típicas:
        time, truck_speed, fuel_consumption, engine_rpm,
        engine_temperature, world_x, world_y, world_z,
        trailer_mass, weather_rain_intensity, ...
    """
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.lower()

    out = pd.DataFrame()
    out["timestamp"]     = pd.to_datetime(df.get("time", pd.RangeIndex(len(df))), unit="s", errors="coerce")
    out["velocidad"]     = df.get("truck_speed", df.get("speed", np.nan)) * 3.6  # m/s → km/h
    out["consumo"]       = df.get("fuel_consumption", df.get("fuel_avg_consumption", np.nan))
    out["rpm"]           = df.get("engine_rpm", np.nan)
    out["temp_motor"]    = df.get("engine_temperature", df.get("oil_temperature", 90.0))
    out["pendiente"]     = df.get("cabin_angular_acceleration_y", 0.0) * 10  # aproximación
    out["temp_ext"]      = df.get("ambient_temperature", 15.0)
    out["precipitacion"] = df.get("weather_rain_intensity", 0.0) * 20
    out["carga"]         = df.get("trailer_mass", 0.0) / df.get("trailer_mass", 1.0).max()
    out["vehicle_type"]  = 0   # ETS2: tractores principalmente
    out["trip_id"]       = df.get("trip_id", 0)

    # Guardar
    Path(output_path).parent.mkdir(exist_ok=True)
    if output_path.endswith(".parquet"):
        out.to_parquet(output_path, index=False)
    else:
        out.to_csv(output_path, index=False)

    print(f"Dataset ETS2 convertido: {len(out)} filas → {output_path}")
    return out


# ── Script de validación rápida ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Uso: python -m data.real_dataset <ruta_al_csv>\n"
            "Ejemplo: python -m data.real_dataset data/mis_viajes.csv\n\n"
            "El script verificará que el formato es correcto y mostrará\n"
            "estadísticas del dataset sin entrenar nada."
        )
        sys.exit(0)

    path = sys.argv[1]
    print(f"\nValidando dataset: {path}")
    print("=" * 50)

    try:
        df = load_raw(path)
        print(f"✅ Archivo cargado: {len(df)} filas")
        print(f"   Columnas: {list(df.columns)}")
        print(f"   Rango temporal: {df['timestamp'].min()} → {df['timestamp'].max()}")

        print("\nEstadísticas por variable:")
        for col in FEATURE_COLS:
            if col in df.columns:
                lo, hi = PHYSICAL_LIMITS[col]
                pct_out = ((df[col] < lo) | (df[col] > hi)).mean() * 100
                print(
                    f"  {col:15s}: "
                    f"media={df[col].mean():.2f}  "
                    f"std={df[col].std():.2f}  "
                    f"[{df[col].min():.2f}, {df[col].max():.2f}]  "
                    f"fuera_rango={pct_out:.1f}%"
                )

        print("\n✅ Formato válido. Puedes usar este archivo con RealTripDataset.")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
