# 🚛 cVAE Fleet — Route Fuel Predictor for Heavy Transport

> **Given an origin and destination, predict fuel consumption and speed with confidence intervals — using real road data, live weather, and a Conditional Variational Autoencoder.**

![cVAE Fleet demo](docs/demo.png)

---

## The Problem

Fleet managers in road freight have no reliable way to estimate fuel consumption before a trip. Static averages ignore terrain, load, weather and driving style. The result: poor planning, budget overruns, and no way to detect anomalous drivers.

**cVAE Fleet solves this by generating probabilistic forecasts** — not a single number, but a P5/P50/P95 range that tells you the realistic best case, expected case, and worst case for any route.

---

## How It Works

```
Origin + Destination
        ↓
  Real route (OSRM)
        ↓
  Elevation profile (Open-Topo-Data)  +  Live weather + wind direction (Open-Meteo)
        ↓
  Conditional VAE generates N synthetic trips
        ↓
  P5 / P50 / P95 confidence intervals — per segment
```

The model was trained on **4,000 synthetic trips** generated with real heavy vehicle physics (aerodynamics, rolling resistance, grade force, BSFC map, 12-speed gearbox, engine thermal model). All APIs are free with no API key required.

---

## Results

| Route | Distance | Load | Consumption P50 | IC 90% |
|-------|----------|------|-----------------|--------|
| Sevilla → Barcelona | 999 km | 75% | 40.3 l/100km | ±3.4 |
| Madrid → Zaragoza | 325 km | 75% | 40.8 l/100km | ±3.2 |
| Madrid → Burgos | 240 km | 80% | 48.7 l/100km | ±2.9 |
| Barcelona → Valencia | 350 km | 60% | 43.7 l/100km | ±3.3 |

**Validation: 11/11 physical checks passed**

```
✅ Full load consumes more than empty     (+15.8 l/100km)
✅ Mountain consumes more than flat       (+83.6 l/100km)
✅ Temperature affects consumption        (+1.1 l/100km)
✅ Downhill consumes less than flat       (5.5 vs 30.6)
✅ Confidence intervals are useful        (IC > 1.5 l/100km in all scenarios)
✅ Realistic absolute values              (30.6 l/100km on flat, 0% slope)
```

---

## Key Technical Decisions

**Why a cVAE instead of a regression model or LSTM?**

A regression model gives one number. A cVAE generates a *distribution* of plausible trips, which is exactly what a fleet manager needs — not "you'll spend 42 l/100km" but "there's a 90% chance you'll spend between 36 and 45".

**Why synthetic physics instead of real data first?**

Real fleet data is scarce, proprietary, and noisy. Training on physics-based synthetic data first gives the model a solid prior — it already understands that uphill burns more than downhill before seeing a single real trip. Fine-tuning on real data afterwards is then much more sample-efficient.

**Why model wind direction and not just wind speed?**

A 25 km/h crosswind has almost zero aerodynamic penalty. The same wind head-on increases drag by ~12%. The model computes the frontal component using the bearing difference between wind direction and route heading.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (CPU, ~2h)
python main.py

# Validate physics
python validate.py

# Launch the app
streamlit run app.py
```

---

## Project Structure

```
cVAEfleet/
├── app.py                  ← Streamlit UI
├── main.py                 ← Training pipeline
├── validate.py             ← 11-point physical validation suite
├── data/
│   └── synthetic.py        ← Physics engine (aero, BSFC map, gearbox, thermal)
├── model/
│   └── cvae.py             ← Conditional VAE architecture
├── train/
│   └── trainer.py          ← Training loop with KL annealing
├── inference/
│   └── predictor.py        ← Confidence interval generation
├── route/
│   └── route_builder.py    ← Route → conditioning vector pipeline
└── anomaly/
    └── filter.py           ← Isolation Forest anomaly detection
```

---

## Supported Vehicles

| Type | Empty mass | Max payload | Engine | Top speed |
|------|-----------|-------------|--------|-----------|
| Tractor (Class 8) | 8,500 kg | 24,000 kg | 420 kW | 90 km/h |
| Rigid truck | 7,500 kg | 12,000 kg | 250 kW | 90 km/h |
| Tanker (ADR) | 9,500 kg | 21,000 kg | 400 kW | 85 km/h |

---

## Roadmap

- [ ] Fine-tuning on real fleet CANbus data
- [ ] Rain effect on rolling resistance
- [ ] Cost estimation in euros
- [ ] Real-time anomaly alerts (actual trip vs P95 prediction)
- [ ] Snow / ice driving conditions

---

## Stack

Python 3.11 · PyTorch · Streamlit · OSRM · Open-Meteo · Open-Topo-Data · Folium · CPU only

---

## License

MIT
