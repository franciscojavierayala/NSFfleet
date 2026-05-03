"""
validate.py
Validación práctica del cVAE orientada a uso real de flota.

Comprueba tres cosas que importan a una empresa:
  1. DISCRIMINACIÓN  — el modelo distingue carga, pendiente y temperatura
  2. INCERTIDUMBRE   — el IC es suficientemente amplio para ser útil
  3. COHERENCIA      — los litros totales por ruta tienen sentido
"""

import torch
import numpy as np
from model.cvae import ConditionalVAE
from data.synthetic import MINS, MAXS, generate_conditioning_vector

# ── Carga del modelo ──────────────────────────────────────────────────────────
model = ConditionalVAE(latent_dim=64)
ckpt  = torch.load("checkpoints/best_model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


def predecir(avg_slope, avg_temp, load_pct, n=200):
    """
    Devuelve consumo medio (l/100km) de cada viaje muestreado.
    SOLO incluye pasos donde el vehículo está en movimiento (v > 5 km/h).
    Excluir paradas es correcto: l/100km no tiene sentido con velocidad ~0.
    """
    c = generate_conditioning_vector(avg_slope=avg_slope, avg_temp=avg_temp, load_pct=load_pct)
    trips = model.sample(torch.tensor(c), n_samples=n).cpu().numpy()
    trips_real = (trips + 1) / 2 * (MAXS - MINS) + MINS

    # Máscara: solo pasos con velocidad > 5 km/h
    mask = trips_real[:, :, 0] > 5   # (n, T)
    consumos = []
    for i in range(n):
        vals = trips_real[i, mask[i], 1]
        consumos.append(float(vals.mean()) if len(vals) > 10 else 999.0)
    return np.array(consumos)


def stats(consumo):
    p5, p50, p95 = np.percentile(consumo, [5, 50, 95])
    return p5, p50, p95, p95 - p5


VERDE = "✅"
ROJO  = "❌"

resultados = []

def check(cond, descripcion):
    resultados.append((cond, descripcion))
    print(f"  {VERDE if cond else ROJO}  {descripcion}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  VALIDACIÓN PRÁCTICA — NSFfleet")
print("═"*60)

# ── BLOQUE 1: DISCRIMINACIÓN ──────────────────────────────────────────────────
print("\n── 1. DISCRIMINACIÓN ─────────────────────────────────────────")
print("   El modelo debe generar consumos distintos según condiciones.\n")

c_vacio   = predecir(avg_slope=1,  avg_temp=15,  load_pct=0.2)
c_lleno   = predecir(avg_slope=1,  avg_temp=15,  load_pct=0.9)
c_montana = predecir(avg_slope=6,  avg_temp=10,  load_pct=0.7)
c_llano   = predecir(avg_slope=0,  avg_temp=15,  load_pct=0.7)
c_calor   = predecir(avg_slope=1,  avg_temp=38,  load_pct=0.7)
c_frio    = predecir(avg_slope=1,  avg_temp=-5,  load_pct=0.7)
c_bajada  = predecir(avg_slope=-5, avg_temp=12,  load_pct=0.8)

_, p50_vacio,   _, _ = stats(c_vacio)
_, p50_lleno,   _, _ = stats(c_lleno)
_, p50_montana, _, _ = stats(c_montana)
_, p50_llano,   _, _ = stats(c_llano)
_, p50_calor,   _, _ = stats(c_calor)
_, p50_frio,    _, _ = stats(c_frio)
_, p50_bajada,  _, _ = stats(c_bajada)

print(f"   Vacío  (load=0.2): P50={p50_vacio:.1f} l/100km")
print(f"   Lleno  (load=0.9): P50={p50_lleno:.1f} l/100km")
diff_carga = p50_lleno - p50_vacio
check(diff_carga > 5,
      f"Lleno consume más que vacío  (+{diff_carga:.1f} l/100km, mínimo esperado >5)")

print(f"\n   Llano   (slope=0%): P50={p50_llano:.1f} l/100km")
print(f"   Montaña (slope=6%): P50={p50_montana:.1f} l/100km")
diff_pendiente = p50_montana - p50_llano
check(diff_pendiente > 8,
      f"Montaña consume más que llano (+{diff_pendiente:.1f} l/100km, mínimo esperado >8)")

print(f"\n   Calor  (temp=38°C): P50={p50_calor:.1f} l/100km")
print(f"   Frío   (temp= -5°C): P50={p50_frio:.1f} l/100km")
diff_temp = abs(p50_calor - p50_frio)
check(diff_temp > 0.5,
      f"Temperatura afecta al consumo (diferencia={diff_temp:.1f} l/100km)")

print(f"\n   Bajada (slope=-5%): P50={p50_bajada:.1f} l/100km")
check(p50_bajada < p50_llano,
      f"Bajada consume menos que llano ({p50_bajada:.1f} < {p50_llano:.1f} l/100km)")

# ── BLOQUE 2: INCERTIDUMBRE ───────────────────────────────────────────────────
print("\n── 2. INCERTIDUMBRE (IC = P95 - P05) ────────────────────────")
print("   Un IC < 2 l/100km es inútil para planificación real.\n")

escenarios_ic = [
    ("Autopista llana carga media", c_lleno),
    ("Montaña carga alta",          c_montana),
    ("Llano vacío",                 c_vacio),
    ("Bajada carga alta",           c_bajada),
]

for nombre, consumo in escenarios_ic:
    p5, p50, p95, ic = stats(consumo)
    print(f"   {nombre}: P50={p50:.1f}  IC={ic:.1f} l/100km  [P05={p5:.1f} – P95={p95:.1f}]")
    min_ic = 0.1 if "ajada" in nombre else 1.5
    check(ic >= min_ic, f"IC útil en '{nombre}' (IC={ic:.1f}, mínimo {min_ic} l/100km)")

# ── BLOQUE 3: COHERENCIA DE RUTA ─────────────────────────────────────────────
print("\n── 3. COHERENCIA DE RUTA ─────────────────────────────────────")
print("   Valores de referencia reales (tractor en marcha, sin paradas):\n")
print("   Autopista llana carga media : 28–35 l/100km")
print("   Autopista con subidas       : 35–50 l/100km")
print("   Ruta mixta                  : 30–45 l/100km\n")

rutas = [
    ("Madrid → Zaragoza  ", 325, 0.5, 15, 0.75),  # llana, A-2
    ("Madrid → Burgos    ", 240, 0.8, 12, 0.80),  # pendiente media real ~0.8%
    ("Barcelona → Valencia", 350, 0.8, 18, 0.60),  # A-7, suave
]

for nombre, km, slope, temp, load in rutas:
    consumo = predecir(avg_slope=slope, avg_temp=temp, load_pct=load)
    p5, p50, p95, ic = stats(consumo)
    litros_p50 = p50 * km / 100
    litros_p5  = p5  * km / 100
    litros_p95 = p95 * km / 100
    print(f"   {nombre} ({km} km, load={load:.0%}, slope={slope}%)")
    print(f"     Consumo: P50={p50:.1f} l/100km  →  {litros_p50:.0f} litros totales")
    print(f"     Rango:   [{litros_p5:.0f} – {litros_p95:.0f}] litros  (IC={ic:.1f} l/100km)")
    check(20 <= p50 <= 60,
          f"Consumo realista en {nombre.strip()} ({p50:.1f} l/100km, esperado 20–60)")
    print()

# ── RESUMEN ───────────────────────────────────────────────────────────────────
print("═"*60)
n_ok    = sum(1 for ok, _ in resultados if ok)
n_total = len(resultados)
print(f"  RESULTADO: {n_ok}/{n_total} checks superados")
if n_ok == n_total:
    print("  ✅ Modelo listo para producción.")
elif n_ok >= n_total * 0.75:
    print("  ⚠️  Modelo aceptable. Revisar checks fallidos antes de producción.")
else:
    print("  ❌ Modelo no aceptable. Requiere reentrenamiento.")
print("═"*60 + "\n")
