"""
main.py
Demo ejecutable del sistema NSFFleet.

Ejecutar con:
    python main.py                      # modo sintético (por defecto)
    python main.py --mode real          # modo datos reales
    python main.py --mode real --data data/mis_viajes.parquet

Pasos:
  1. Carga dataset (sintético o real)
  2. Entrena el cVAE
  3. Predice una ruta nueva con intervalos de confianza
  4. Filtra anomalías sobre viajes entrantes
  5. Muestra visualización de los resultados
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from data.synthetic import MINS, MAXS


# ── Configuración ─────────────────────────────────────────────────────────────
CFG = {
    # Modo de datos: "synthetic" | "real"
    "data_mode":    "synthetic",

    # Rutas de datos reales (solo se usan si data_mode = "real")
    "real_train_path": "data/trips_train.parquet",
    "real_val_path":   None,                        # None = split automático 80/20
    "scaler_path":     "checkpoints/scaler.json",

    # Entrenamiento sintético
    "n_trips":       4000,
    "batch_size":    64,
    "epochs":        100,
    "lr":            1e-4,
    "warmup_epochs": 25,
    "latent_dim":    64,
    "n_samples":     100,
    "checkpoint_dir": "checkpoints",
    "seed":          42,
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])


# ── 1. DATOS ──────────────────────────────────────────────────────────────────
def step_data(data_mode: str = "synthetic"):
    print("=" * 60)
    if data_mode == "real":
        print("PASO 1 — Cargando dataset REAL de telemetría...")
        print("=" * 60)

        train_path = Path(CFG["real_train_path"])
        if not train_path.exists():
            print(f"\n⚠️  Archivo no encontrado: {train_path}")
            print("   Opciones:")
            print("   1. Coloca tu CSV/Parquet en data/trips_train.parquet")
            print("   2. Usa --data <ruta_a_tu_archivo>")
            print("   3. Ejecuta sin --mode real para usar datos sintéticos")
            print("\n   Cambiando a modo sintético automáticamente...\n")
            return step_data("synthetic")

        from data.real_dataset import get_real_dataloaders
        train_loader, val_loader = get_real_dataloaders(
            train_path=str(train_path),
            val_path=CFG["real_val_path"],
            scaler_path=CFG["scaler_path"],
            batch_size=CFG["batch_size"],
            seed=CFG["seed"],
        )
        print(f"  Train: {len(train_loader.dataset)} ventanas")
        print(f"  Val:   {len(val_loader.dataset)} ventanas\n")
        return train_loader, val_loader, "real"

    else:
        print("PASO 1 — Generando dataset SINTÉTICO...")
        print("=" * 60)
        from data.synthetic import get_dataloaders
        train_loader, val_loader = get_dataloaders(
            n_trips=CFG["n_trips"],
            batch_size=CFG["batch_size"],
            seed=CFG["seed"],
        )
        print(f"  Train: {len(train_loader.dataset)} viajes")
        print(f"  Val:   {len(val_loader.dataset)} viajes\n")
        return train_loader, val_loader, "synthetic"


# ── 2. ENTRENAMIENTO ──────────────────────────────────────────────────────────
def step_train(train_loader, val_loader, data_mode: str = "synthetic"):
    print("=" * 60)
    print(f"PASO 2 — Entrenando ConditionalFlowModel (NSF)  [modo: {data_mode}]")
    print("=" * 60)
 
    # MIGRACIÓN: ConditionalVAE → ConditionalFlowModel
    # El Trainer ya no recibe warmup_epochs (los flujos no necesitan KL annealing)
    from model.nflow_model import ConditionalFlowModel
    from train.trainer import Trainer
 
    epochs = CFG["epochs"] if data_mode == "synthetic" else max(CFG["epochs"], 50)
 
    model = ConditionalFlowModel()   # hiperparámetros por defecto optimizados para CPU
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=CFG["lr"],
        checkpoint_dir=CFG["checkpoint_dir"],
    )
    history = trainer.train(epochs=epochs)
 
    # FIX: recargar el mejor checkpoint guardado en disco.
    # Tras trainer.train(), el modelo ya se recarga internamente, pero recargamos
    # explícitamente aquí para garantizar consistencia con la lógica de app.py.
    best_path = Path(CFG["checkpoint_dir"]) / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n  ✓ Mejor modelo recargado desde: {best_path}")
        print(f"    best_val_nll  : {trainer.best_val_nll:.4f}")
        print(f"    best_val_loss : {trainer.best_val_loss:.4f}")
    else:
        print("\n  ⚠️  No se encontró best_model.pt — usando pesos de la última época.")
 
    # Guardar metadatos del entrenamiento
    meta_path = Path(CFG["checkpoint_dir"]) / "training_meta.json"
    meta = {
        "data_mode":    data_mode,
        "model_type":   "ConditionalFlowModel",
        "epochs":       epochs,
        "n_train":      len(train_loader.dataset),
        "n_val":        len(val_loader.dataset),
        "best_val_nll": trainer.best_val_nll,
        "best_val_loss": trainer.best_val_loss,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
 
    return model, trainer, history


# ── 3. INFERENCIA — RUTA NUEVA ────────────────────────────────────────────────
def step_inference(model):
    print("\n" + "=" * 60)
    print("PASO 3 — Predicción de ruta nueva")
    print("=" * 60)
    print("  Ruta ejemplo: Madrid → Zaragoza → Barcelona")
    print("  Condiciones: pendiente media 2%, 8°C, sin lluvia, carga 75%\n")

    from inference.predictor import FleetPredictor
    predictor = FleetPredictor(model)
    result = predictor.predict_route(
        avg_slope=2.0,
        avg_temp=8.0,
        precipitation=0.0,
        load_pct=0.75,
        vehicle_type=0,
        day_of_week=1,
        n_samples=CFG["n_samples"],
    )
    predictor.print_report(result)
    return result, predictor


# ── 4. FILTRO DE ANOMALÍAS ────────────────────────────────────────────────────
def step_anomaly_filter(train_loader, data_mode: str = "synthetic"):
    print("=" * 60)
    print("PASO 4 — Filtro de anomalías")
    print("=" * 60)

    from anomaly.filter import AnomalyFilter
    from data.synthetic import generate_trip

    # Pool de referencia: primeros 500 viajes del train
    pool = [trip.numpy() for trip, _ in train_loader.dataset][:500]

    contamination = 0.03 if data_mode == "real" else 0.05
    af = AnomalyFilter(contamination=contamination)
    af.fit(pool)

    print(f"\n  Modo: {data_mode} — contamination={contamination}")
    print("  Evaluando viajes entrantes simulados...")

    incoming = []

    if data_mode == "real":
        for trip, _ in train_loader.dataset:
            incoming.append(trip.numpy())
            if len(incoming) >= 8:
                break
    else:
        for _ in range(8):
            trip = generate_trip(
                avg_slope=np.random.uniform(-3, 3),
                avg_temp=np.random.uniform(5, 20),
                load_pct=np.random.uniform(0.5, 0.9),
            )
            incoming.append(trip)

    # 2 viajes anómalos (siempre simulados para poder controlarlos)
    bad_trip_1 = generate_trip()
    bad_trip_1[:, 1] = -0.9   # consumo negativo → sensor roto
    incoming.append(bad_trip_1)

    bad_trip_2 = generate_trip()
    bad_trip_2[:, 0] = 0.99   # velocidad máxima sostenida → error GPS
    incoming.append(bad_trip_2)

    valid_trips, discarded = af.filter_batch(incoming, verbose=True)
    return valid_trips, discarded, af


# ── 5. VISUALIZACIÓN ──────────────────────────────────────────────────────────
def step_visualize(result, history, data_mode: str = "synthetic"):
    print("\n" + "=" * 60)
    print("PASO 5 — Generando visualización...")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 10))
    title = f"NSFFleet — Demo  [datos: {data_mode}]"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Curvas de entrenamiento
    ax1 = fig.add_subplot(gs[0, 0])
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_recon"], label="Train recon", color="#1F5C99")
    ax1.plot(epochs_range, history["val_recon"],   label="Val recon",   color="#E07B39",
             linestyle="--")
    ax1.set_title("Pérdida de reconstrucción")
    ax1.set_xlabel("Época")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # NLL del flujo 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_range, history["train_nll"], label="Train NLL", color="#2CA02C")
    ax2.plot(epochs_range, history["val_nll"],   label="Val NLL",   color="#9467BD",
             linestyle="--")
    ax2.set_title("Log-verosimilitud (NLL)")
    ax2.set_xlabel("Época")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)


    # Distribución de consumo
    ax3 = fig.add_subplot(gs[0, 2])
    trips_real = (result["trips_raw"] + 1) / 2 * (MAXS[1] - MINS[1]) + MINS[1]
    mean_consumption = trips_real[:, :, 1].mean(axis=1)
    ax3.hist(mean_consumption, bins=20, color="#1F5C99", alpha=0.8, edgecolor="white")
    p5, p50, p95 = np.percentile(mean_consumption, [5, 50, 95])
    ax3.axvline(p50, color="red",    linestyle="--", label=f"P50={p50:.1f}")
    ax3.axvline(p5,  color="orange", linestyle=":",  label=f"P5={p5:.1f}")
    ax3.axvline(p95, color="orange", linestyle=":",  label=f"P95={p95:.1f}")
    ax3.set_title("Distribución consumo (l/100km)")
    ax3.set_xlabel("l/100km")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Perfiles de velocidad
    ax4 = fig.add_subplot(gs[1, :2])
    trips_v = (result["trips_raw"][:, :, 0] + 1) / 2 * (MAXS[0] - MINS[0]) + MINS[0]
    for i in range(5):
        ax4.plot(trips_v[i], alpha=0.4, linewidth=0.8)
    ax4.plot(np.percentile(trips_v, 50, axis=0), color="black",
             linewidth=1.5, label="Mediana (P50)")
    ax4.fill_between(
        range(trips_v.shape[1]),
        np.percentile(trips_v, 5,  axis=0),
        np.percentile(trips_v, 95, axis=0),
        alpha=0.15, color="blue", label="IC 90%",
    )
    ax4.set_title("Perfiles de velocidad — 5 viajes sintéticos de ejemplo")
    ax4.set_xlabel("Tiempo (min)")
    ax4.set_ylabel("Velocidad (km/h)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # Consumo P50 por tramo
    ax5 = fig.add_subplot(gs[1, 2])
    segs = sorted(result["segments"], key=lambda x: x["tramo"])
    tramos = [f"T{s['tramo']}" for s in segs]
    p50s   = [s["consumo"]["p50"] for s in segs]
    p5s    = [s["consumo"]["p5"]  for s in segs]
    p95s   = [s["consumo"]["p95"] for s in segs]
    yerr   = [
        [p50 - p5  for p50, p5  in zip(p50s, p5s)],
        [p95 - p50 for p95, p50 in zip(p95s, p50s)],
    ]
    ax5.bar(tramos, p50s, color="#1F5C99", alpha=0.8, yerr=yerr,
            capsize=5, error_kw={"color": "#E07B39", "linewidth": 1.5})
    ax5.set_title("Consumo P50 por tramo (IC 90%)")
    ax5.set_ylabel("l/100km")
    ax5.grid(alpha=0.3, axis="y")

    out_path = Path("outputs")
    out_path.mkdir(exist_ok=True)
    out_file = out_path / f"demo_results_{data_mode}.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"  Gráfico guardado en: {out_file}")
    plt.close()


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSFFleet — Demo")
    parser.add_argument(
        "--mode", choices=["synthetic", "real"], default="synthetic",
        help="Fuente de datos: 'synthetic' (por defecto) o 'real'",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Ruta al CSV/Parquet con datos reales (solo con --mode real)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Sobrescribir número de épocas de entrenamiento",
    )
    args = parser.parse_args()

    # Aplicar argumentos
    CFG["data_mode"] = args.mode
    if args.data:
        CFG["real_train_path"] = args.data
    if args.epochs:
        CFG["epochs"] = args.epochs

    print(f"\n{'='*60}")
    print(f"  NSFFleet  —  modo: {CFG['data_mode'].upper()}")
    print(f"{'='*60}\n")

    # 1. Datos
    train_loader, val_loader, data_mode = step_data(CFG["data_mode"])

    # 2. Entrenar y recargar el mejor checkpoint
    model, trainer, history = step_train(train_loader, val_loader, data_mode)

    # 3. Inferencia sobre ruta nueva (usa el mejor modelo, no el de la última época)
    result, predictor = step_inference(model)

    # 4. Filtro de anomalías
    valid_trips, discarded, af = step_anomaly_filter(train_loader, data_mode)

    # 5. Visualizar
    step_visualize(result, history, data_mode)

    print(f"\n✓ Demo completada  [modo: {data_mode}]")
    print(f"  Checkpoint : checkpoints/best_model.pt")
    print(f"  Metadatos  : checkpoints/training_meta.json")
    print(f"  Gráfico    : outputs/demo_results_{data_mode}.png\n")
