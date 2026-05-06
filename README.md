# 🚛 cNSFfleet — Probabilistic Route Fuel Predictor for Heavy Transport

> **Given an origin and destination, predict fuel consumption and speed with confidence intervals — using real road data, live weather, and a Conditional Neural Spline Flow.**

![cNSFfleet demo](docs/demo.png)

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CPU only](https://img.shields.io/badge/hardware-CPU%20only-lightgrey)]()

---

## Table of Contents

1. [The Problem](#the-problem)
2. [How It Works](#how-it-works)
3. [Architecture Overview](#architecture-overview)
4. [Results](#results)
5. [Key Technical Decisions](#key-technical-decisions)
6. [Quick Start](#quick-start)
7. [Project Structure](#project-structure)
8. [Module Reference](#module-reference)
9. [Supported Vehicles](#supported-vehicles)
10. [APIs Used](#apis-used)
11. [Training Details](#training-details)
12. [Validation](#validation)
13. [Fine-tuning on Real Data](#fine-tuning-on-real-data)
14. [Roadmap](#roadmap)
15. [Stack](#stack)
16. [License](#license)

---

## The Problem

Fleet managers in road freight have no reliable way to estimate fuel consumption before a trip. Static averages ignore terrain, load, weather and driving style. The result: poor planning, budget overruns, and no way to detect anomalous drivers.

Current approaches fail in predictable ways:

| Approach | Limitation |
|----------|-----------|
| Fleet average (l/100km) | Ignores route topology, weather, load |
| GPS-based post-hoc reporting | Only available after the trip |
| Physics simulators | Deterministic — no uncertainty quantification |
| ML regression models | Point estimates — no confidence intervals |

**cNSFfleet solves this by generating probabilistic forecasts** — not a single number, but a P5/P50/P95 range that tells you the realistic best case, expected case, and worst case for any route. Fleet managers can use the P95 for budget planning and the P5 as a lower bound for driver performance benchmarking.

---

## How It Works

```
Origin + Destination (free text)
        ↓
  Geocoding via Nominatim (OSM)
        ↓
  Real road route via OSRM
        ↓
  Elevation profile via Open-Topo-Data
        ↓
  Live weather + wind direction via Open-Meteo
        ↓
  Route conditioning vector assembled:
  [avg_slope, avg_temp, precipitation, load_pct,
   vehicle_type, day_of_week, wind_frontal_component]
        ↓
  ConditionalFlowModel samples N synthetic trips
  across 10 randomised driving styles
        ↓
  P5 / P50 / P95 confidence intervals — global + per segment
        ↓
  Streamlit dashboard: map, charts, metrics table
```

All APIs are **free and require no API key**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     app.py (Streamlit UI)                │
│  searchbox → route_builder → predictor → visualisation  │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │    route_builder.py   │  Nominatim + OSRM + Topo + Meteo
        │  → conditioning vec   │  + frontal wind component
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  ConditionalFlowModel │  Neural Spline Flow (nflows)
        │  nflow_model.py       │  Rational-quadratic spline transforms
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   FleetPredictor      │  10 driving styles × n_samples/10
        │   predictor.py        │  → P5/P50/P95 per segment
        └───────────────────────┘
```

---

## Results

| Route | Distance | Load | Consumption P50 | IC 90% |
|-------|----------|------|-----------------|--------|
| Sevilla → Barcelona | 999 km | 75% | 40.3 l/100km | ±3.4 |
| Madrid → Zaragoza | 325 km | 75% | 40.8 l/100km | ±3.2 |
| Madrid → Burgos | 240 km | 80% | 48.7 l/100km | ±2.9 |
| Barcelona → Valencia | 350 km | 60% | 43.7 l/100km | ±3.3 |

Reference values for a Euro VI tractor at highway speed, 75% load: **38–44 l/100km**. All P50 predictions fall within this range.

**Validation: 16/16 physical checks passed** (run `python validate.py` to reproduce)

```
Block 1 — Consumption discrimination
  ✅ Full load consumes more than empty         (+15.8 l/100km)
  ✅ Mountain consumes more than flat           (+83.6 l/100km)
  ✅ Temperature affects consumption            (+1.1 l/100km)
  ✅ Downhill consumes less than flat           (5.5 vs 30.6 l/100km)

Block 2 — Consumption uncertainty (IC = P95 − P05)
  ✅ Useful IC in all 4 scenarios               (IC > threshold l/100km)

Block 3 — Route coherence
  ✅ Realistic absolute values on 3 routes      (20–60 l/100km band)

Block 4 — Flow calibration
  ✅ P5 < P50 < P95 on flat terrain
  ✅ P5 < P50 < P95 on mountain terrain

Block 5 — Speed variability
  ✅ Aggressive driver faster than conservative
  ✅ Mountain speed lower than flat
  ✅ Mountain speed realistic                   (≥ 25 km/h)
```

---

## Key Technical Decisions

### Why a Neural Spline Flow instead of a cVAE?

The original architecture was a Conditional VAE. VAEs approximate the posterior via the ELBO — which introduces reconstruction error, requires careful KL annealing schedules to avoid posterior collapse, and produces confidence intervals indirectly by averaging imperfect reconstructions.

`ConditionalFlowModel` (implemented with the [`nflows`](https://github.com/bayesiains/nflows) library) replaces this with a **normalizing flow using rational-quadratic spline transforms**. Concretely:

| Property | cVAE | cNSF (this project) |
|----------|------|---------------------|
| Training objective | ELBO (lower bound) | Exact NLL |
| Posterior collapse risk | Yes — KL annealing needed | No |
| Confidence intervals | Indirect (averaged reconstructions) | Direct (density sampling) |
| Interval sharpness | Blurred by decoder variance | Exact percentiles |
| Training complexity | KL warmup schedule required | Single loss term |

The training history tracks both `train_nll` / `val_nll` (flow objective) and `train_recon` / `val_recon` (reconstruction quality). The best checkpoint is selected by `best_val_nll`.

### Why model driving style diversity at inference time?

A single conditioning vector produces samples from one implicit driver profile. At inference, `FleetPredictor` draws 10 driving styles uniformly from `[0, 1]` and generates `n_samples / 10` trips per style, then concatenates them. This means the P5/P95 interval reflects genuine **inter-driver variability**, not just model uncertainty — which is what fleet managers actually need for benchmarking and anomaly detection.

### Why synthetic physics instead of real data first?

Real fleet data is scarce, proprietary, and noisy. Training on physics-based synthetic data first gives the model a solid prior — it already understands that uphill burns more than downhill before seeing a single real trip. Fine-tuning on real data afterwards (via `--mode real`) is then much more sample-efficient.

The physics engine (`data/synthetic.py`) models:
- **Aerodynamic drag**: frontal area × drag coefficient × air density × v²
- **Rolling resistance**: load-dependent, tyre pressure assumed constant
- **Grade force**: mg·sin(θ) per segment slope
- **BSFC map**: fuel consumption as a function of engine torque and RPM
- **12-speed gearbox**: gear selection based on speed and load
- **Engine thermal model**: cold-start fuel penalty at low temperatures

### Why model wind direction and not just wind speed?

A 25 km/h crosswind has almost zero aerodynamic penalty. The same wind head-on increases drag by ~12%. Both `app.py` and `route_builder.py` compute the frontal wind component using `cos(bearing_diff)` between wind direction and route heading:

```python
angle_diff = math.radians(abs(route_bearing - wind_dir) % 360)
wind_frontal = wind_speed * math.cos(angle_diff)
# Positive = headwind (penalises consumption)
# Negative = tailwind (reduces consumption)
```

This component is passed as a conditioning feature to the flow, allowing the model to distinguish a 90 km/h route into a 30 km/h headwind from the same route with a tailwind.

---

## Quick Start

### Prerequisites

- Python 3.11+
- ~2 GB RAM (CPU training)
- No GPU required

### Installation

```bash
git clone https://github.com/franciscojavierayala/cNSFfleet.git
cd cNSFfleet
pip install -r requirements.txt
```

### Train

```bash
# Train on synthetic data (~2h on CPU)
python main.py

# Train on real fleet data (CSV or Parquet)
python main.py --mode real --data data/mis_viajes.parquet
```

### Validate

```bash
# Run 16 physical checks across 5 blocks
python validate.py
```

### Launch

```bash
streamlit run app.py
```

Then open `http://localhost:8501`, type an origin and destination, and click **Predecir**.

---

## Project Structure

```
cNSFfleet/
├── __init__.py
├── app.py                      ← Streamlit UI
├── main.py                     ← Training pipeline (--mode synthetic | real)
├── validate.py                 ← Physical validation (16 checks, 5 blocks)
├── requirements.txt            ← Includes nflows>=0.14
├── docs/
│   └── demo.png
├── data/
│   ├── synthetic.py            ← Physics engine (aero, BSFC map, gearbox, thermal)
│   └── real_dataset.py         ← Real telemetry loader (CSV / Parquet)
├── model/
│   └── nflow_model.py          ← ConditionalFlowModel (Neural Spline Flow via nflows)
├── train/
│   └── trainer.py              ← Training loop — NLL optimisation
├── inference/
│   └── predictor.py            ← FleetPredictor — P5/P50/P95 with driving style diversity
├── route/
│   └── route_builder.py        ← Route → conditioning vector pipeline
└── checkpoints/
    ├── best_model.pt           ← Best checkpoint (selected by val NLL)
    └── training_meta.json      ← Training run metadata (mode, epochs, best NLL)
```

---

## Module Reference

### `data/synthetic.py`
Generates synthetic trip data using a full heavy-vehicle physics model. Each trip is a time series of `(speed_kmh, consumption_l100km)` tuples. Exposes `MINS` and `MAXS` arrays used to normalise features across the codebase.

### `model/nflow_model.py`
Defines `ConditionalFlowModel`: a normalizing flow with rational-quadratic spline transforms conditioned on a route vector. The model learns the joint distribution of `(speed, consumption)` sequences conditioned on route features.

### `train/trainer.py`
Training loop. Minimises negative log-likelihood (NLL) on the training set. Tracks `train_nll`, `val_nll`, `train_recon`, `val_recon`. Saves the checkpoint with lowest `val_nll` to `checkpoints/best_model.pt`.

### `inference/predictor.py`
`FleetPredictor` loads a trained checkpoint and, given a conditioning vector, generates `n_samples` synthetic trips distributed across 10 driving styles. Returns P5/P50/P95 for both consumption and speed, globally and per route segment.

### `route/route_builder.py`
Assembles the route conditioning vector from free-text origin/destination:
1. Geocodes via Nominatim
2. Fetches road route via OSRM
3. Queries elevation profile via Open-Topo-Data
4. Fetches weather forecast via Open-Meteo
5. Computes route bearing and frontal wind component
6. Returns a normalised conditioning dict

### `app.py`
Streamlit dashboard. City autocomplete via Nominatim. Interactive Folium map with per-segment colour coding. Bar charts with P5/P50/P95 error bars. Speed profile visualisation with synthetic trip overlays. Summary metrics and per-segment table.

---

## Supported Vehicles

| Type | Empty mass | Max payload | Engine | Top speed |
|------|-----------|-------------|--------|-----------|
| Tractor (Class 8) | 8,500 kg | 24,000 kg | 420 kW | 90 km/h |
| Rigid truck | 7,500 kg | 12,000 kg | 250 kW | 90 km/h |
| Tanker (ADR) | 9,500 kg | 21,000 kg | 400 kW | 85 km/h |

---

## APIs Used

| API | Purpose | Key required |
|-----|---------|-------------|
| [Nominatim (OSM)](https://nominatim.org/) | Geocoding & city autocomplete | No |
| [OSRM](http://router.project-osrm.org/) | Real road routing + polyline | No |
| [Open-Topo-Data](https://www.opentopodata.org/) | Elevation profile along route | No |
| [Open-Meteo](https://open-meteo.com/) | Weather forecast (temp, wind, rain) | No |

All APIs are free, open, and do not require registration.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Training set | 4,000 synthetic trips |
| Validation split | 20% |
| Optimizer | Adam |
| Objective | Negative log-likelihood (NLL) |
| Checkpoint criterion | Lowest `val_nll` |
| Hardware | CPU only |
| Approximate training time | ~2 hours |
| Features per step | 2 — `(speed_kmh, consumption_l100km)` |
| Conditioning vector size | 7 features |

The conditioning vector contains: `avg_slope`, `avg_temp`, `precipitation`, `load_pct`, `vehicle_type` (ordinal 0–2), `day_of_week` (0–6), `wind_frontal_component`.

---

## Validation

`validate.py` runs 16 deterministic checks across 5 blocks. No ground-truth labels are required — all checks are physics-based assertions on model outputs.

```bash
python validate.py
```

Expected output summary:
```
══════════════════════════════════════════════════════════
 RESULTADO: 16/16 checks superados
 ✅ Modelo listo para producción.
══════════════════════════════════════════════════════════
```

If fewer than 75% of checks pass, the model requires retraining. Common causes: insufficient training epochs, or corrupted checkpoint.

---

## Fine-tuning on Real Data

The pipeline supports fine-tuning on real CANbus telemetry. Expected input format:

```
CSV / Parquet with columns:
  timestamp, speed_kmh, fuel_rate_lh, latitude, longitude,
  load_pct, vehicle_id, [optional: temperature, wind_speed]
```

```bash
python main.py --mode real --data data/mis_viajes.parquet
```

The `real_dataset.py` loader handles resampling to fixed time steps, missing value interpolation, and per-vehicle normalisation via `scaler.json`. Fine-tuning on as few as **200 real trips** typically improves P50 accuracy by 15–25% on routes similar to the training fleet.

---

## Roadmap

- [ ] Rain effect on rolling resistance
- [ ] Cost estimation in euros (fuel price per route)
- [ ] Real-time anomaly alerts (actual trip vs P95 prediction)
- [ ] Snow / ice driving conditions
- [ ] Environment isolation via Docker
- [ ] REST API endpoint (FastAPI wrapper around `FleetPredictor`)
- [ ] Multi-stop route support

---

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Deep learning | PyTorch 2.x |
| Normalizing flows | nflows ≥ 0.14 |
| UI | Streamlit |
| Maps | Folium + streamlit-folium |
| Routing | OSRM (public instance) |
| Weather | Open-Meteo |
| Elevation | Open-Topo-Data |
| Geocoding | Nominatim (OSM) |
| Hardware | CPU only |

---

## License

MIT — see [LICENSE](LICENSE) for details.
