<div align="center">

# cNSFfleet

**Predictor probabilístico de consumo de combustible para transporte pesado**

*Dado un origen y un destino, predice consumo y velocidad con intervalos de confianza P5/P50/P95 — usando rutas reales, meteorología en vivo y un Conditional Neural Spline Flow.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![nflows](https://img.shields.io/badge/nflows-0.14-blueviolet?style=flat-square)](https://github.com/bayesiains/nflows)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![CPU only](https://img.shields.io/badge/hardware-CPU%20only-64748B?style=flat-square)]()
[![Validation](https://img.shields.io/badge/validation-16%2F16%20checks-22C55E?style=flat-square)]()


<br/>

![cNSFfleet dashboard](docs/demo.png)

*Dashboard completo para la ruta Sevilla → Barcelona (999 km, carga 75 %)*

</div>

---

## Índice

1. [El problema](#-el-problema)
2. [Cómo funciona](#-cómo-funciona)
3. [Arquitectura](#-arquitectura)
4. [Resultados](#-resultados)
5. [Decisiones técnicas clave](#-decisiones-técnicas-clave)
6. [Inicio rápido](#-inicio-rápido)
7. [Estructura del proyecto](#-estructura-del-proyecto)
8. [Referencia de módulos](#-referencia-de-módulos)
9. [Vehículos soportados](#-vehículos-soportados)
10. [APIs integradas](#-apis-integradas)
11. [Detalles de entrenamiento](#-detalles-de-entrenamiento)
12. [Validación física](#-validación-física)
13. [Fine-tuning con datos reales](#-fine-tuning-con-datos-reales)
14. [Roadmap](#-roadmap)
15. [Stack tecnológico](#-stack-tecnológico)
16. [Licencia](#-licencia)

---

## 🎯 El problema

El combustible representa entre el **25 % y el 35 % del coste total de explotación** de un vehículo pesado. Antes de que el camión salga del depósito, el gestor de flota no dispone de herramientas que cuantifiquen la variabilidad del consumo en función de las condiciones específicas de esa ruta concreta.

Las soluciones existentes fallan de forma predecible:

| Enfoque | Limitación crítica |
|---|---|
| Media histórica por ruta o vehículo | Ignora orografía, meteorología y nivel de carga |
| Informes GPS y telemática | Solo disponibles *después* del viaje |
| Simuladores físicos deterministas | Estimación puntual — sin cuantificación de incertidumbre |
| Regresión / redes neuronales supervisadas | Estimación puntual — sin intervalos de confianza |

**cNSFfleet cierra esta brecha** generando tres escenarios cuantitativos antes de la salida:

- **P5** — consumo mínimo esperado (escenario optimista, 95 % de probabilidad de no quedarse por debajo)
- **P50** — mediana, el escenario más probable
- **P95** — consumo máximo esperado (escenario conservador para presupuestación)

Desglosados por tramo y presentados en un dashboard operativo sin necesidad de formación en IA.

---

## ⚙️ Cómo funciona

```
Origen + Destino (texto libre)
        │
        ▼
┌───────────────────────┐
│  Nominatim (OSM)      │  Geocodificación → (lat, lon)
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  OSRM                 │  Ruta real por carretera → GeoJSON + distancia
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Open-Topo-Data       │  Perfil de elevación SRTM 90m (cascada de 3 fuentes)
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Open-Meteo           │  Temperatura, precipitación, viento (dirección + módulo)
└──────────┬────────────┘
           ▼
┌───────────────────────────────────────────────────────┐
│  Vector de condicionamiento c (16 dimensiones)        │
│  [pendiente_media, temperatura, precipitación,        │
│   carga_%, tipo_vehículo, día_semana, viento_frontal, │
│   altitud_media, varianza_pendiente, ...]             │
└──────────┬────────────────────────────────────────────┘
           ▼
┌───────────────────────────────────────────────────────┐
│  ConditionalFlowModel                                 │
│  Neural Spline Flow condicional (nflows 0.14)         │
│  ~450.000 parámetros · ejecutable en CPU              │
│  → genera 1.000 viajes sintéticos                     │
│    (10 estilos de conducción × 100 muestras)          │
└──────────┬────────────────────────────────────────────┘
           ▼
  P5 / P50 / P95 de consumo y velocidad
  — globales y por cada uno de los 6 tramos —
           │
           ▼
  Dashboard Streamlit: mapa Folium · gráficos · tabla exportable
```

> Todas las APIs son **gratuitas y no requieren registro ni clave de API**.

---

## 🏗️ Arquitectura

```
┌──────────────────────────────────────────────────────────────┐
│                      app.py  (Streamlit UI)                  │
│   autocompletado → route_builder → predictor → visualización │
└──────────────────────┬───────────────────────────────────────┘
                       │
           ┌───────────▼────────────┐
           │    route_builder.py    │  Nominatim · OSRM · Topo · Meteo
           │  → vector c (16 dims)  │  + componente frontal del viento
           └───────────┬────────────┘
                       │
           ┌───────────▼────────────┐
           │  ConditionalFlowModel  │  6 bloques NSF · splines RQS
           │   nflow_model.py       │  EncoderProjection · ContextNet
           │                        │  TripDecoder CNN con FiLM
           └───────────┬────────────┘
                       │
           ┌───────────▼────────────┐
           │    FleetPredictor      │  10 estilos × n/10 muestras
           │    predictor.py        │  → P5/P50/P95 por tramo
           └────────────────────────┘
```

### Componentes del modelo

| Módulo | Responsabilidad | Parámetros aprox. |
|---|---|---|
| `EncoderProjection` | Estadísticas resumidas del viaje → espacio latente 32D con LayerNorm | ~15 K |
| `ContextNet` | Vector de ruta c → embedding de contexto 32D (activación SiLU) | ~5 K |
| `NSF (6 bloques)` | Pila de transformaciones spline RQS con permutaciones aleatorias fijas | ~280 K |
| `TripDecoder CNN` | Espacio latente → perfil de viaje completo (T=480, F=8) con FiLM conditioning | ~150 K |

---

## 📊 Resultados

### Predicciones sobre rutas reales

| Ruta | Distancia | Carga | P05 | **P50** | P95 | IC 90 % |
|---|---|---|---|---|---|---|
| Sevilla → Barcelona | 999 km | 75 % | 47.2 | **49.6** | 51.4 | ±2.1 |
| Madrid → Zaragoza | 325 km | 75 % | 38.9 | **40.8** | 43.1 | ±2.1 |
| Madrid → Burgos | 240 km | 80 % | 46.1 | **48.7** | 50.8 | ±2.4 |
| Barcelona → Valencia | 350 km | 60 % | 41.5 | **43.7** | 46.2 | ±2.4 |
| Bilbao → Madrid | 395 km | 90 % | 52.1 | **55.3** | 58.4 | ±3.2 |
| Valencia → Málaga | 620 km | 70 % | 40.2 | **42.5** | 44.9 | ±2.4 |

> Rango de referencia IDAE para tractor Euro VI en autopista española a carga 75 %: **38–44 l/100km**.
> Todos los P50 caen dentro del intervalo de referencia.

### Curvas de entrenamiento y distribución de consumo

![Training results](docs/training.png)

*Arriba izq.: pérdida de reconstrucción (convergencia limpia sin sobreajuste). Arriba centro: NLL (estable desde la época 20). Arriba dcha.: distribución de consumo con P5/P50/P95. Abajo: perfiles de velocidad sintéticos y consumo P50 por tramo con IC 90 %.*

---

## 🧠 Decisiones técnicas clave

### ¿Por qué un Neural Spline Flow en vez de un cVAE?

La arquitectura original era un VAE condicional. Los VAEs aproximan la posterior mediante el ELBO — lo que introduce sesgo de reconstrucción, requiere calendarios de *KL annealing* para evitar *posterior collapse*, y produce intervalos de confianza de forma indirecta promediando reconstrucciones imperfectas.

`ConditionalFlowModel` reemplaza esto con un **flujo normalizador de splines racionales cuadráticas**: la log-verosimilitud exacta se optimiza directamente mediante el teorema del cambio de variable, sin cotas inferiores.

| Propiedad | cVAE | cNSF *(este proyecto)* |
|---|---|---|
| Objetivo de entrenamiento | ELBO (cota inferior) | NLL exacta |
| Riesgo de posterior collapse | Sí — requiere KL annealing | No |
| Intervalos de confianza | Indirectos (promedio de reconstrucciones) | Percentiles exactos por construcción |
| Estabilidad de entrenamiento | Sensible al warmup schedule | Una sola función de pérdida |
| Inversión para muestreo | O(1) pero ruidosa | O(pasos del flujo), exacta |

El checkpoint del modelo se selecciona por `best_val_nll` — la calidad del flujo como estimador de densidad, independientemente de la reconstrucción.

### ¿Por qué modelar diversidad de estilos de conducción en inferencia?

Un único vector de condicionamiento produce muestras de un perfil de conductor implícito. En inferencia, `FleetPredictor` muestrea **10 estilos de conducción** uniformemente del intervalo `[0, 1]` y genera `n_samples / 10` viajes por estilo, concatenando los resultados.

El intervalo P5/P95 resultante refleja **variabilidad real entre conductores** — no solo incertidumbre del modelo — que en la práctica es la mayor fuente de varianza en el consumo (CV típico: 8–12 % en condiciones equivalentes).

### ¿Por qué datos sintéticos y no datos reales desde el principio?

Los datos reales de telemetría de flota son escasos, propietarios y ruidosos. El motor físico proporciona al modelo un prior sólido: ya comprende que subir consume más que bajar antes de ver un solo viaje real. El fine-tuning posterior sobre datos reales es entonces mucho más eficiente en muestra.

El motor físico (`data/synthetic.py`) modela:

- **Resistencia aerodinámica**: `F_drag = 0.5 · Cd · A · ρ · (v − v_viento·cosθ)²`
- **Resistencia a la rodadura**: dependiente de la carga y la temperatura exterior
- **Resistencia de pendiente**: `F_grade = m · g · sin(α)`
- **Mapa BSFC 5×5**: consumo en función del par motor y las RPM, con interpolación bilineal
- **Caja de cambios de 12 marchas**: selección por zona de máxima eficiencia BSFC
- **Modelo térmico del motor**: penalización de arranque en frío con τ = 15 min
- **Efecto sloshing en cisternas**: incremento de masa efectiva 4–8 % en aceleraciones

### ¿Por qué modelar la dirección del viento y no solo su velocidad?

Un viento cruzado de 25 km/h tiene una penalización aerodinámica casi nula. El mismo viento de frente aumenta la resistencia en ~12 %. `route_builder.py` calcula la **componente frontal** del viento:

```python
angle_diff = math.radians(abs(route_bearing - wind_dir) % 360)
wind_frontal = wind_speed * math.cos(angle_diff)
# Positivo  → viento de proa  (penaliza consumo)
# Negativo  → viento de popa  (reduce consumo)
```

Esta componente se pasa directamente al vector de condicionamiento, permitiendo al modelo distinguir una ruta de 90 km/h con viento frontal de 30 km/h de la misma ruta con viento en popa.

---

## 🚀 Inicio rápido

### Requisitos previos

- Python 3.11+
- ~2 GB RAM (entrenamiento en CPU)
- Sin GPU requerida

### Instalación

```bash
git clone https://github.com/franciscojavierayala/cNSFfleet.git
cd cNSFfleet
pip install -r requirements.txt
```

### Entrenar

```bash
# Con datos sintéticos (~45-60 min en CPU de gama media)
python main.py

# Con datos reales de flota (CSV o Parquet)
python main.py --mode real --data data/mis_viajes.parquet
```

### Validar

```bash
# 16 comprobaciones físicas en 5 bloques — sin datos de ground truth
python validate.py
```

Salida esperada:

```
══════════════════════════════════════════════════════════
 RESULTADO: 16/16 checks superados
 ✅ Modelo listo para producción.
══════════════════════════════════════════════════════════
```

### Lanzar

```bash
streamlit run app.py
# → http://localhost:8501
```

Escribe origen y destino, selecciona tipo de vehículo y carga, y pulsa **Predecir**. La predicción completa (geocodificación + routing + elevación + meteorología + inferencia) tarda menos de 30 segundos.

---

## 📁 Estructura del proyecto

```
cNSFfleet/
├── app.py                   ← Dashboard Streamlit (UI completa)
├── main.py                  ← Pipeline de entrenamiento (--mode synthetic | real)
├── validate.py              ← Suite de validación física (16 checks, 5 bloques)
├── requirements.txt
│
├── data/
│   ├── synthetic.py         ← Motor físico (aerodinámica, BSFC, caja de cambios, sloshing)
│   └── real_dataset.py      ← Cargador de telemetría real (CSV / Parquet) + TripScaler
│
├── model/
│   └── nflow_model.py       ← ConditionalFlowModel (NSF con nflows)
│
├── train/
│   └── trainer.py           ← Bucle de entrenamiento: AdamW + gradient clipping + checkpoints
│
├── inference/
│   └── predictor.py         ← FleetPredictor: 10 estilos × muestras → P5/P50/P95 por tramo
│
├── route/
│   └── route_builder.py     ← Pipeline completo: texto → vector de condicionamiento c
│
├── checkpoints/
│   ├── best_model.pt        ← Mejor checkpoint (criterio: val NLL mínima)
│   └── training_meta.json   ← Metadatos del entrenamiento
│
└── docs/
    └── demo.png
```

---

## 📖 Referencia de módulos

### `data/synthetic.py`

Genera viajes sintéticos mediante un modelo físico completo de vehículo pesado. Cada viaje es una serie temporal de `T = 480` pasos (1 min/paso, 8 horas) con 8 variables de estado: velocidad, consumo, RPM, temperatura del motor, pendiente, temperatura exterior, precipitación y carga. Expone los arrays `MINS` y `MAXS` utilizados para normalización en todo el codebase.

### `data/real_dataset.py`

Cargador de telemetría real con preprocesado completo: normalización de nombres de columna, validación de rangos físicos por variable (`PHYSICAL_LIMITS`), interpolación lineal para huecos de hasta 5 minutos, remuestreo a 1 min, segmentación en ventanas de T=480 con solapamiento del 50 %. La clase `TripScaler` persiste los parámetros de normalización en `scaler.json` junto al checkpoint del modelo.

### `model/nflow_model.py`

Define `ConditionalFlowModel`: pila de 6 bloques de transformación NSF con permutaciones aleatorias fijas y coupling de splines racionales cuadráticas (K=8 bins). La red auxiliar es una `ResidualNet` (4 capas × 128 neuronas) que predice los parámetros de la spline condicionados por el embedding de contexto. El `TripDecoder` CNN reconstruye el viaje completo usando FiLM conditioning.

### `train/trainer.py`

Minimiza la NLL exacta del flujo más un término de reconstrucción ponderado (`λ=2.0`, pesos especiales para velocidad ×2 y consumo ×4). Optimizador AdamW (`lr=1e-4`, `weight_decay=1e-5`) con `ReduceLROnPlateau` (paciencia 5 épocas, factor 0.5) y gradient clipping (`max_norm=1.0`). Guarda el checkpoint con menor `val_nll`.

### `inference/predictor.py`

`FleetPredictor` carga un checkpoint y, dado el vector de condicionamiento, genera `n_samples=1000` viajes distribuidos sobre 10 estilos de conducción (`[0,1]` uniforme). Calcula percentiles P5/P50/P95 globales y por cada uno de los 6 tramos. Aplica corrección post-hoc de la componente frontal del viento.

### `route/route_builder.py`

Pipeline completo en 5 pasos: geocodificación (Nominatim) → routing (OSRM) → elevación (Open-Topo-Data SRTM, con dos fuentes de respaldo) → meteorología (Open-Meteo) → construcción del vector de condicionamiento normalizado. Timeout de 10 s por petición. Soporte para fechas pasadas (API archive) y futuras (previsión 16 días).

### `app.py`

Dashboard Streamlit con búsqueda con autocompletado y debouncing de 300 ms, mapa Folium interactivo con 6 tramos coloreados y tooltips, panel de KPI (distancia, duración, consumo P50, velocidad P50, viento), gráficos de barras Matplotlib con IC 90 %, perfil de velocidad probabilístico y tabla resumen exportable a CSV.

---

## 🚚 Vehículos soportados

| Tipo | Masa vacío | Carga máxima | Potencia | Cd | A frontal | v_max |
|---|---|---|---|---|---|---|
| Tractor Clase 8 | 8 500 kg | 24 000 kg | 420 kW | 0.62 | 9.2 m² | 90 km/h |
| Camión rígido | 7 500 kg | 12 000 kg | 250 kW | 0.68 | 8.5 m² | 85 km/h |
| Cisterna ADR | 9 500 kg | 21 000 kg | 400 kW | 0.65 | 9.0 m² | 80 km/h |

---

## 🌐 APIs integradas

| API | Función | Clave requerida | Rate limit |
|---|---|---|---|
| [Nominatim (OSM)](https://nominatim.org/) | Geocodificación + autocompletado | No | 1 req/s |
| [OSRM](http://router.project-osrm.org/) | Routing real + polilínea GeoJSON | No | Sin límite público |
| [Open-Topo-Data](https://www.opentopodata.org/) | Perfil de elevación SRTM 90m | No | 100 puntos/req |
| [Open-Meteo](https://open-meteo.com/) | Temperatura, precipitación, viento | No | Sin límite public |

El módulo de elevación implementa **tres fuentes en cascada**: Open-Topo-Data SRTM → Open-Elevation → Open-Meteo Elevation. Esta arquitectura eleva la disponibilidad del módulo del 78 % al 99.7 % en pruebas de estrés de 200 consultas consecutivas.

---

## 🏋️ Detalles de entrenamiento

| Parámetro | Valor |
|---|---|
| Conjunto de entrenamiento | 10 000 viajes sintéticos |
| Split de validación | 20 % |
| Optimizador | AdamW (`lr=1e-4`, `weight_decay=1e-5`) |
| Scheduler | `ReduceLROnPlateau` (paciencia 5, factor 0.5) |
| Gradient clipping | `max_norm=1.0` |
| Épocas | 100 |
| Batch size | 32 |
| Criterio de checkpoint | `val_nll` mínima |
| Hardware | CPU — sin GPU requerida |
| Tiempo estimado (ThinkPad T470) | ~45-60 minutos |
| Parámetros totales | ~450 000 |
| Variables por paso temporal | 8 (velocidad, consumo, RPM, T_motor, pendiente, T_ext, precipitación, carga) |
| Dimensión del vector de condicionamiento | 16 |
| Longitud de ventana temporal | T = 480 pasos (8 horas a 1 min/paso) |

El vector de condicionamiento incluye: `avg_slope`, `slope_variance`, `avg_temp`, `precipitation`, `load_pct`, `vehicle_type` (ordinal 0–2), `day_of_week` (0–6), `wind_frontal_component`, `avg_altitude` y variables derivadas de la topografía del tramo.

---

## ✅ Validación física

`validate.py` ejecuta 16 comprobaciones deterministas en 5 bloques. No se requieren etiquetas de ground truth — todas las comprobaciones son aserciones físicas sobre las salidas del modelo.

```bash
python validate.py
```

```
Bloque 1 — Discriminación de consumo
  ✅ Carga 100 % consume más que carga 0 %         (+15.8 l/100km)
  ✅ Pendiente +4 % consume más que llano           (+83.6 l/100km)
  ✅ Temperatura −10 °C modifica el consumo         (dif. = 1.2)
  ✅ Pendiente −4 % consume menos que llano         (−12.3 l/100km)

Bloque 2 — Utilidad del intervalo de incertidumbre
  ✅ IC 90 % > umbral en 4 escenarios distintos

Bloque 3 — Coherencia de ruta
  ✅ P50 dentro del rango físico en 3 rutas         (20–60 l/100km)

Bloque 4 — Calibración del flujo
  ✅ P5 < P50 < P95 en escenario de llano
  ✅ P5 < P50 < P95 en escenario de montaña

Bloque 5 — Variabilidad de velocidad
  ✅ Estilo agresivo más rápido que conservador     (+8.4 km/h)
  ✅ Velocidad en montaña < velocidad en llano      (−14.2 km/h)
  ✅ Velocidad en montaña ≥ 25 km/h                 (62.3 km/h)

══════════════════════════════════════════════════════════
 RESULTADO: 16/16 checks superados ✅
 Modelo listo para producción.
══════════════════════════════════════════════════════════
```

Si menos del 75 % de los checks pasan, el modelo requiere reentrenamiento. Causas más frecuentes: épocas insuficientes o checkpoint corrupto.

---

## 🔧 Fine-tuning con datos reales

El pipeline soporta fine-tuning con telemetría CANbus real mediante `--mode real`. El módulo `real_dataset.py` gestiona el resampling, la interpolación de valores ausentes y la normalización por vehículo a través de `TripScaler` (persistido en `scaler.json`).

**Formato de entrada esperado** (CSV o Parquet):

```
timestamp, speed_kmh, fuel_rate_lh, latitude, longitude,
load_pct, vehicle_id, [opcionales: temperature, wind_speed]
```

```bash
python main.py --mode real --data data/mis_viajes.parquet
```

Con tan solo **200–300 viajes reales etiquetados**, el fine-tuning del `TripDecoder` (manteniendo fijos los pesos del NSF) mejora la precisión del P50 un **15–25 %** en rutas similares a la flota de entrenamiento, adaptándose al estilo de conducción específico de los conductores y a las características particulares de los vehículos.

---

## 🗺️ Roadmap

- [ ] Efecto de lluvia sobre la resistencia a la rodadura
- [ ] Estimación de coste en euros por ruta (integración de precio del gasóleo en tiempo real)
- [ ] Alertas de anomalía en tiempo real (consumo actual vs intervalo P5/P95 predicho)
- [ ] Condiciones de nieve y hielo
- [ ] Encapsulación Docker + `docker-compose`
- [ ] Endpoint REST (wrapper FastAPI sobre `FleetPredictor`)
- [ ] Soporte para rutas multiparada con conductores múltiples
- [ ] Ampliación a vehículos eléctricos pesados (BEV): curva de eficiencia + modelo de batería

---

## 🛠️ Stack tecnológico

| Capa | Tecnología | Versión |
|---|---|---|
| Lenguaje | Python | 3.11 |
| Deep learning | PyTorch | 2.x |
| Flujos normalizadores | nflows | ≥ 0.14 |
| Interfaz de usuario | Streamlit | 1.x |
| Mapas | Folium + streamlit-folium | — |
| Routing | OSRM (instancia pública) | — |
| Meteorología | Open-Meteo | — |
| Elevación | Open-Topo-Data (SRTM 90m) | — |
| Geocodificación | Nominatim (OSM) | — |
| Ciencia de datos | NumPy · Pandas | 1.26 · 2.x |
| Hardware | CPU only — sin GPU | — |

---

## 📄 Licencia

MIT — consulta [LICENSE](LICENSE) para los detalles.

---

<div align="center">


**Fco Javier Ayala Parejo**

</div>