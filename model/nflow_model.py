"""
model/nflow_model.py
Modelo generativo condicional basado en Neural Spline Flows (NSF).

Reemplaza al ConditionalVAE manteniendo compatibilidad total de interfaz
con FleetPredictor (inference/predictor.py) y app.py.

Arquitectura general (flujo de datos):
  ENTRENAMIENTO
    trip (B, T, F) → extract_stats → stats (B, D_STATS=48)
                  → encoder_proj  → z    (B, D_FLOW=32)
    c    (B, C_DIM) → context_net  → ctx  (B, CTX_DIM=32)
    NLL  = -flow.log_prob(z, ctx)
    recon = decoder(z, c)          → trip_hat (B, T, F)
    L_total = NLL + λ * MSE(recon, trip)

  MUESTREO
    c (C_DIM,) → context_net → ctx (1, CTX_DIM)
    z ~ flow.sample(n_samples, context=ctx) → (n_samples, D_FLOW)
    trip = decoder(z, c_expanded)           → (n_samples, T, F)

Restricciones de hardware cumplidas (ThinkPad T470, CPU):
  - D_FLOW = 32 ≤ 32  ✓
  - flow_steps = 4    ≤ 5  ✓
  - hidden_features = 64   ≤ 128  ✓
  - num_bins = 8           ✓
  - Parámetros totales ≈ 400K  (muy ligero para CPU)
"""

import torch
import torch.nn as nn
import nflows.utils
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    PiecewiseRationalQuadraticCouplingTransform,
    RandomPermutation,
)
from nflows.nn.nets import ResidualNet

from data.synthetic import T, F, C_DIM

# ── Hiperparámetros de arquitectura ───────────────────────────────────────────
N_SEGS: int = 6          # tramos de telemetría para extraer estadísticos
D_STATS: int = N_SEGS * F  # 48 — dimensión del vector de estadísticos por viaje
D_FLOW: int = 32         # ≤ 32 según restricción de hardware
CTX_DIM: int = 32        # dimensión del embedding de contexto
FLOW_STEPS: int = 4      # ≤ 5 según restricción
HIDDEN_FEAT: int = 64    # ≤ 128 según restricción
NUM_BINS: int = 8        # equilibrio expresividad/coste
TAIL_BOUND: float = 5.0  # límite de la spline en las colas lineales
RECON_WEIGHT: float = 0.5   # subido ligeramente para que FiLM reciba señal suficiente  # λ: peso del término de reconstrucción en el loss


# ── Utilidades de preprocesado ─────────────────────────────────────────────────
def extract_segment_stats(trip: torch.Tensor, n_segs: int = N_SEGS) -> torch.Tensor:
    """
    Extrae medias por tramo de un batch de viajes.

    Args:
        trip:  (B, T, F) — viajes normalizados en [-1, 1]
        n_segs: número de tramos

    Returns:
        stats: (B, n_segs * F) — medias por tramo, aplanadas
    """
    B, Tlen, Fdim = trip.shape
    seg_len = Tlen // n_segs
    segments = []
    for i in range(n_segs):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segs - 1 else Tlen
        # Media sobre la dimensión temporal del tramo: (B, F)
        segments.append(trip[:, start:end, :].mean(dim=1))
    return torch.cat(segments, dim=1)  # (B, n_segs * F)


# ── Red de contexto ────────────────────────────────────────────────────────────
class ContextNet(nn.Module):
    """
    MLP que embebe el vector de condicionamiento c en un espacio de dimensión CTX_DIM.
    El vector c ya viene normalizado desde generate_conditioning_vector():
      - Pendiente: dividida por 10 → [-1, 1]
      - Temperatura: dividida por 40 → [-1, 1]
      - Precipitación: dividida por 20 → [0, 1]
      - Carga: directamente en [0, 1]
      - Tipo vehículo: one-hot(3)
      - Día semana: one-hot(7)
      - 2 dims de padding → cero
    No se aplica normalización adicional aquí.
    """

    def __init__(self, c_dim: int = C_DIM, ctx_dim: int = CTX_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_dim, ctx_dim),
            nn.SiLU(),                       # SiLU (Swish) más suave que ReLU para flujos
            nn.Linear(ctx_dim, ctx_dim),
            nn.SiLU(),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B, C_DIM) → ctx: (B, CTX_DIM)"""
        return self.net(c)


# ── Proyección encoder/decoder del espacio de estadísticos ────────────────────
class EncoderProjection(nn.Module):
    """
    Proyección lineal aprendida: D_STATS (48) → D_FLOW (32).
    Actúa como un PCA aprendido sobre el espacio de estadísticos de viaje.
    LayerNorm antes de pasar al flujo mejora la estabilidad numérica.
    """

    def __init__(self, d_stats: int = D_STATS, d_flow: int = D_FLOW):
        super().__init__()
        self.proj = nn.Linear(d_stats, d_flow, bias=True)
        self.norm = nn.LayerNorm(d_flow)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """stats: (B, D_STATS) → z: (B, D_FLOW)"""
        return self.norm(self.proj(stats))


# ── FiLM conditioning ─────────────────────────────────────────────────────────
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: modula un mapa de activaciones (B, Ch, T)
    con escala y sesgo predichos a partir del vector de condicionamiento c.

    Para cada canal, aprende:
        y = scale(c) * x + shift(c)

    Esto permite que el condicionamiento controle la magnitud y el offset de
    cada canal de forma independiente, lo que es crítico para que variables
    de diferente naturaleza física (velocidad, consumo, RPM...) respondan
    de forma distinta al mismo vector de contexto.
    """

    def __init__(self, channels: int, c_dim: int):
        super().__init__()
        self.scale = nn.Linear(c_dim, channels)
        self.shift = nn.Linear(c_dim, channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Ch, T) — mapa de activaciones convolucional
            c: (B, C_DIM) — vector de condicionamiento

        Returns:
            (B, Ch, T) — activaciones moduladas
        """
        s = self.scale(c).unsqueeze(2)   # (B, Ch, 1) → broadcast sobre T
        b = self.shift(c).unsqueeze(2)   # (B, Ch, 1)
        return x * s + b


# ── Decoder convolucional con FiLM ────────────────────────────────────────────
class TripDecoder(nn.Module):
    """
    Decoder con FiLM conditioning en cada capa de deconvolución.

    El condicionamiento c actúa en tres niveles:
      1. Concatenado con z en la capa FC inicial (igual que antes).
      2. FiLM tras la primera ConvTranspose: modula canales de alta resolución baja.
      3. FiLM tras la segunda ConvTranspose: modula canales de resolución media.

    Esto obliga al decoder a diferenciar el consumo de la velocidad según el
    tipo de vehículo, la carga y las condiciones meteorológicas, que son
    exactamente las variables que controlan la física del consumo.

    Flujo:
      cat(z, c) → FC → reshape (B, 64, T//8)
        → ConvT1 → FiLM(c) → ReLU
        → ConvT2 → FiLM(c) → ReLU
        → ConvT3 → Tanh
        → permute → (B, T, F)
    """

    def __init__(
        self,
        d_flow: int = D_FLOW,
        c_dim: int = C_DIM,
        t: int = T,
        f: int = F,
    ):
        super().__init__()
        self.t = t
        self.f = f
        self._t_compressed = t // 8   # 60 con T=480

        # FC inicial: proyecta (z, c) al espacio de activaciones convolucional
        self.fc = nn.Sequential(
            nn.Linear(d_flow + c_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64 * self._t_compressed),
            nn.SiLU(),
        )

        # Deconvoluciones: 60 → 120 → 240 → 480
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.film1   = FiLM(32, c_dim)   # modula 32 canales tras primer upsample

        self.deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.film2   = FiLM(16, c_dim)   # modula 16 canales tras segundo upsample

        self.deconv3 = nn.ConvTranspose1d(16, f, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.out_act = nn.Tanh()         # salida en [-1, 1] igual que el cVAE original

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D_FLOW)
            c: (B, C_DIM)

        Returns:
            trip: (B, T, F)
        """
        B = z.size(0)
        x = torch.cat([z, c], dim=1)              # (B, D_FLOW + C_DIM)
        x = self.fc(x)                             # (B, 64 * T//8)
        x = x.view(B, 64, self._t_compressed)     # (B, 64, T//8=60)

        x = torch.relu(self.deconv1(x))            # (B, 32, 120)
        x = self.film1(x, c)                       # FiLM: escala y sesgo por canal

        x = torch.relu(self.deconv2(x))            # (B, 16, 240)
        x = self.film2(x, c)                       # FiLM: escala y sesgo por canal

        x = self.out_act(self.deconv3(x))          # (B, F, 480)
        return x.permute(0, 2, 1)                  # (B, T, F)


# ── Construcción del flujo NSF ─────────────────────────────────────────────────
def _build_nsf(
    features: int = D_FLOW,
    context_features: int = CTX_DIM,
    num_flow_steps: int = FLOW_STEPS,
    hidden_features: int = HIDDEN_FEAT,
    num_bins: int = NUM_BINS,
    tail_bound: float = TAIL_BOUND,
) -> Flow:
    """
    Construye un Neural Spline Flow condicional con PiecewiseRationalQuadratic couplings.

    Cada paso del flujo contiene:
      1. RandomPermutation — mezcla las dimensiones para que el enmascaramiento
         no cree dependencias parciales estables.
      2. PiecewiseRationalQuadraticCoupling — transforma la mitad de las dims
         con una spline condicionada en las otras dims (y en el contexto c).

    La distribución base es N(0, I) en D_FLOW dimensiones.

    Args:
        features:         dimensión del espacio del flujo (D_FLOW = 32)
        context_features: dimensión del embedding de contexto (CTX_DIM = 32)
        num_flow_steps:   número de pasos de transformación (4)
        hidden_features:  capas ocultas de la ResNet interna (64)
        num_bins:         número de bins de la spline racional cuadrática (8)
        tail_bound:       límite a partir del cual las colas son lineales (5.0)

    Returns:
        flow: objeto Flow de nflows con métodos log_prob y sample
    """
    transforms = []
    for step in range(num_flow_steps):
        # Permutación aleatoria: evita dependencias estructurales fijas
        transforms.append(RandomPermutation(features=features))

        # Máscara alternante: en pasos pares transforma los índices pares
        mask = nflows.utils.create_alternating_binary_mask(
            features=features, even=(step % 2 == 0)
        )

        # Coupling transform con ResNet condicional interna
        def _make_resnet(in_f: int, out_f: int, ctx: int = context_features, h: int = hidden_features) -> ResidualNet:
            return ResidualNet(
                in_features=in_f,
                out_features=out_f,
                hidden_features=h,
                context_features=ctx,
                num_blocks=2,          # 2 bloques residuales por acoplamiento
                activation=torch.nn.functional.silu,
                dropout_probability=0.0,  # sin dropout para flujos (rompe biyectividad en inferencia)
                use_batch_norm=False,     # BN incompatible con evaluación de densidad exacta
            )

        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=_make_resnet,
                num_bins=num_bins,
                tails="linear",   # colas lineales fuera del soporte de la spline
                tail_bound=tail_bound,
                apply_unconditional_transform=True,
            )
        )

    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal(shape=[features]),
    )


# ── Modelo principal ───────────────────────────────────────────────────────────
class ConditionalFlowModel(nn.Module):
    """
    Modelo generativo condicional basado en Neural Spline Flows.

    Reemplaza ConditionalVAE con interfaz drop-in compatible con FleetPredictor.

    El flujo opera en un espacio latente comprimido (D_FLOW=32) derivado de
    estadísticos de viaje por tramos (medias por segmento, D_STATS=48).
    Un decoder convolucional reconstruye la secuencia completa (T, F) desde
    el vector latente y el condicionamiento.

    Interfaz pública (idéntica al ConditionalVAE):
      - forward(x, c)         → log_prob (B,)
      - sample(c, n_samples)  → (n_samples, T, F)
      - log_prob(x, c)        → log_prob (B,)
    """

    def __init__(
        self,
        d_flow: int = D_FLOW,
        ctx_dim: int = CTX_DIM,
        flow_steps: int = FLOW_STEPS,
        hidden_features: int = HIDDEN_FEAT,
        num_bins: int = NUM_BINS,
        recon_weight: float = RECON_WEIGHT,
    ):
        super().__init__()
        self.d_flow = d_flow
        self.recon_weight = recon_weight

        # Red de embedding de contexto: C_DIM → CTX_DIM
        self.context_net = ContextNet(c_dim=C_DIM, ctx_dim=ctx_dim)

        # Proyección estadísticos → espacio del flujo
        self.encoder_proj = EncoderProjection(d_stats=D_STATS, d_flow=d_flow)

        # Flujo normalizado (NSF con PRQC)
        self.flow = _build_nsf(
            features=d_flow,
            context_features=ctx_dim,
            num_flow_steps=flow_steps,
            hidden_features=hidden_features,
            num_bins=num_bins,
        )

        # Decoder: reconstruye secuencia completa desde z y c
        self.decoder = TripDecoder(d_flow=d_flow, c_dim=C_DIM, t=T, f=F)

    # ── API pública (contrato drop-in) ────────────────────────────────────────

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Calcula la log-verosimilitud de x dado el contexto c.

        Equivalente funcional al forward del cVAE pero sin reconstrucción.
        En entrenamiento, úsese training_loss() que incluye el término de
        reconstrucción para entrenar el decoder.

        Args:
            x: (B, T, F) — viaje normalizado
            c: (B, C_DIM) — vector de condicionamiento

        Returns:
            log_prob: (B,) — log p(x|c) por muestra
        """
        stats = extract_segment_stats(x)          # (B, D_STATS)
        z = self.encoder_proj(stats)              # (B, D_FLOW)
        ctx = self.context_net(c)                 # (B, CTX_DIM)
        return self.flow.log_prob(z, context=ctx)  # (B,)

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        API adicional que expone la densidad exacta del flujo.
        Idéntico a forward(); se provee como alias semánticamente explícito
        para scoring de viajes en inferencia (no disponible en el cVAE).

        Args:
            x: (B, T, F) — viaje normalizado
            c: (B, C_DIM) — vector de condicionamiento

        Returns:
            log_prob: (B,) — log p(x|c)
        """
        return self.forward(x, c)

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Genera n_samples viajes sintéticos para un condicionamiento c dado.

        CONTRATO CRÍTICO: firma idéntica al ConditionalVAE.sample() para
        compatibilidad con FleetPredictor:
          predictor.py llama → model.sample(c_tensor, n_samples=n_samples)

        Args:
            c:        (C_DIM,) o (1, C_DIM) — vector de condicionamiento
            n_samples: número de viajes sintéticos a generar

        Returns:
            trips: (n_samples, T, F) — viajes sintéticos normalizados en [-1, 1]
        """
        self.eval()
        device = next(self.parameters()).device

        if c.dim() == 1:
            c = c.unsqueeze(0)         # (1, C_DIM)
        c = c.to(device)

        # Embedding de contexto para el flujo: (1, CTX_DIM)
        ctx = self.context_net(c)

        # Muestreo del flujo: flow.sample(n, ctx=(1, CTX_DIM)) → (1, n, D_FLOW)
        z = self.flow.sample(n_samples, context=ctx).squeeze(0)  # (n_samples, D_FLOW)

        # Decodificar a secuencia de viaje completa
        c_expanded = c.expand(n_samples, -1)      # (n_samples, C_DIM)
        trips = self.decoder(z, c_expanded)       # (n_samples, T, F)
        return trips

    # ── Entrenamiento ────────────────────────────────────────────────────────────

    def training_loss(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcula la pérdida combinada NLL + reconstrucción.

        L_total = NLL + λ * L_recon
          NLL     = -mean(log p_θ(z|c))  — entrena context_net, encoder_proj, flow
          L_recon = MSE(decoder(z, c), x) — entrena encoder_proj, decoder
                    (z no se detacha: permite gradiente compartido)

        El término de reconstrucción es necesario para entrenar el decoder;
        sin él, el decoder no recibe gradientes y sample() produciría ruido.
        El peso λ=0.3 está calibrado para que NLL domine (flujo bien ajustado)
        y la reconstrucción sea un regularizador suave del decoder.

        Args:
            x: (B, T, F) — batch de viajes
            c: (B, C_DIM) — vectores de condicionamiento

        Returns:
            total_loss:  escalar
            nll:         NLL media por muestra (para monitorización)
            recon_loss:  MSE media por muestra (para monitorización)
        """
        # Extraer estadísticos y proyectar al espacio del flujo
        stats = extract_segment_stats(x)           # (B, D_STATS)
        z = self.encoder_proj(stats)               # (B, D_FLOW)
        ctx = self.context_net(c)                  # (B, CTX_DIM)

        # ── Término NLL del flujo (pérdida principal) ──────────────────────────
        # log p_θ(z|c) = log p_Z(f_θ(z, c)) + log|det J_{f_θ}(z, c)|
        # nflows calcula ambos términos internamente mediante la biyectividad del flujo
        log_prob = self.flow.log_prob(z, context=ctx)  # (B,)
        nll = -log_prob.mean()

        # ── Término de reconstrucción (entrena el decoder) ────────────────────
        # z se usa sin .detach() para permitir gradiente compartido encoder_proj ← recon
        trip_hat = self.decoder(z, c)              # (B, T, F)

        # Pesos por feature: consumo (idx 1) ponderado 3×, velocidad (idx 0) 2×
        weights = torch.ones(F, device=x.device)
        weights[0] = 2.0
        weights[1] = 4.0    # consumo: peso alto para que FiLM aprenda a diferenciarlo
        recon_loss = (weights * (trip_hat - x).pow(2)).mean()

        total_loss = nll + self.recon_weight * recon_loss
        return total_loss, nll.detach(), recon_loss.detach()

    def count_parameters(self) -> dict[str, int]:
        """Desglose de parámetros por componente (útil para validación de hardware)."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "context_net":    count(self.context_net),
            "encoder_proj":   count(self.encoder_proj),
            "flow":           count(self.flow),
            "decoder":        count(self.decoder),
            "total":          count(self),
        }
