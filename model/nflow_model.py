"""
model/nflow_model.py
Modelo generativo condicional basado en Neural Spline Flows (NSF).

Reemplaza al ConditionalVAE manteniendo compatibilidad total de interfaz
con FleetPredictor (inference/predictor.py) y app.py.

Arquitectura general (flujo de datos):
  ENTRENAMIENTO
    trip (B, T, F) → extract_stats → stats (B, D_STATS=96)   ← media+std por tramo
                  → encoder_proj  → z    (B, D_FLOW=32)
    c    (B, C_DIM) → context_net  → ctx  (B, CTX_DIM=32)
    NLL  = -flow.log_prob(z, ctx)
    recon = decoder(z, c)          → (mu, log_sigma) (B, T, F) cada uno
    L_recon = NLL gaussiana ponderada
    L_total = NLL + λ * L_recon

  MUESTREO
    c (C_DIM,) → context_net → ctx (1, CTX_DIM)
    z ~ flow.sample(n_samples, context=ctx) → (n_samples, D_FLOW)
    trip ~ decoder.sample(z, c_expanded)    → (n_samples, T, F)

Cambios respecto a la versión anterior:
  1. TripDecoder ahora predice (mu, log_sigma) en lugar de un único trip.
     deconv3 tiene F*2 canales de salida. El muestreo usa
     trip ~ N(mu, exp(log_sigma)²), lo que introduce variabilidad real.
  2. L_recon cambia de MSE a NLL gaussiana ponderada, obligando al modelo
     a aprender la incertidumbre residual de forma calibrada.
  3. ConditionalFlowModel.sample() usa decoder.sample() en lugar de
     decoder.forward(), propagando la estocasticidad al inferir.
  4. El comentario del docstring del flujo de datos actualizado.

Restricciones de hardware (ThinkPad T470, CPU):
  - D_FLOW = 32  ✓
  - flow_steps = 6
  - hidden_features = 64  ✓
  - Parámetros totales ≈ 450K
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
N_SEGS: int = 6              # tramos de telemetría
D_STATS: int = N_SEGS * F * 2  # 96 — media + std por tramo y feature
D_FLOW: int = 32             # dimensión del espacio del flujo
CTX_DIM: int = 32            # dimensión del embedding de contexto
FLOW_STEPS: int = 6          # pasos de transformación
HIDDEN_FEAT: int = 64        # neuronas ocultas de la ResNet interna
NUM_BINS: int = 8            # bins de la spline racional cuadrática
TAIL_BOUND: float = 5.0      # límite a partir del cual las colas son lineales
RECON_WEIGHT: float = 2.0    # λ: peso del término de reconstrucción


# ── Utilidades de preprocesado ─────────────────────────────────────────────────
def extract_segment_stats(trip: torch.Tensor, n_segs: int = N_SEGS) -> torch.Tensor:
    """
    Extrae media Y desviación estándar por tramo de un batch de viajes.

    Args:
        trip:   (B, T, F) — viajes normalizados en [-1, 1]
        n_segs: número de tramos

    Returns:
        stats: (B, n_segs * F * 2) — [medias | stds] por tramo, aplanadas
    """
    B, Tlen, Fdim = trip.shape
    seg_len = Tlen // n_segs
    segments = []
    for i in range(n_segs):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segs - 1 else Tlen
        seg = trip[:, start:end, :]
        seg_mean = seg.mean(dim=1)
        seg_std  = seg.std(dim=1, unbiased=True).clamp(min=1e-6)
        segments.append(torch.cat([seg_mean, seg_std], dim=1))  # (B, 2F)
    return torch.cat(segments, dim=1)  # (B, N_SEGS * 2F = 96)


# ── Red de contexto ────────────────────────────────────────────────────────────
class ContextNet(nn.Module):
    def __init__(self, c_dim: int = C_DIM, ctx_dim: int = CTX_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_dim, ctx_dim),
            nn.SiLU(),
            nn.Linear(ctx_dim, ctx_dim),
            nn.SiLU(),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B, C_DIM) → ctx: (B, CTX_DIM)"""
        return self.net(c)


# ── Proyección encoder del espacio de estadísticos ────────────────────────────
class EncoderProjection(nn.Module):
    def __init__(self, d_stats: int = D_STATS, d_flow: int = D_FLOW):
        super().__init__()
        self.proj = nn.Linear(d_stats, d_flow, bias=True)
        self.norm = nn.LayerNorm(d_flow)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """stats: (B, D_STATS) → z: (B, D_FLOW)"""
        return self.norm(self.proj(stats))


# ── FiLM conditioning ─────────────────────────────────────────────────────────
class FiLM(nn.Module):
    """Feature-wise Linear Modulation: y = scale(c) * x + shift(c)"""

    def __init__(self, channels: int, c_dim: int):
        super().__init__()
        self.scale = nn.Linear(c_dim, channels)
        self.shift = nn.Linear(c_dim, channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, Ch, T)  c: (B, C_DIM) → (B, Ch, T)"""
        s = self.scale(c).unsqueeze(2)
        b = self.shift(c).unsqueeze(2)
        return x * s + b


# ── Decoder convolucional con FiLM — salida estocástica ───────────────────────
class TripDecoder(nn.Module):
    """
    Decoder con FiLM conditioning.

    Flujo:
      cat(z, c) → FC → reshape (B, 64, T//8)
        → ConvT1 → FiLM(c) → ReLU
        → ConvT2 → FiLM(c) → ReLU
        → ConvT3 → split → mu (Tanh), log_sigma (clamp)

    forward() devuelve (mu, log_sigma).
    sample()  devuelve mu + eps * exp(log_sigma), clipped a [-1, 1].
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
        self._t_compressed = t // 8  # 60 con T=480

        self.fc = nn.Sequential(
            nn.Linear(d_flow + c_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64 * self._t_compressed),
            nn.SiLU(),
        )

        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.film1   = FiLM(32, c_dim)

        self.deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.film2   = FiLM(16, c_dim)

        # F*2 canales: primera mitad → mu, segunda mitad → log_sigma
        self.deconv3 = nn.ConvTranspose1d(16, f * 2, kernel_size=7, stride=2, padding=3, output_padding=1)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D_FLOW)
            c: (B, C_DIM)
        Returns:
            mu:        (B, T, F) en [-1, 1]
            log_sigma: (B, T, F) en [-4, 0]  → sigma en [0.018, 1.0]
        """
        B = z.size(0)
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = x.view(B, 64, self._t_compressed)

        x = torch.relu(self.deconv1(x))
        x = self.film1(x, c)

        x = torch.relu(self.deconv2(x))
        x = self.film2(x, c)

        x = self.deconv3(x).permute(0, 2, 1)   # (B, T, F*2)
        mu, log_sigma = x.chunk(2, dim=-1)
        mu        = torch.tanh(mu)
        log_sigma = log_sigma.clamp(-4, 0)
        return mu, log_sigma

    def sample(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Muestrea un viaje: trip ~ N(mu, exp(log_sigma)²).

        Returns:
            trip: (B, T, F) en [-1, 1]
        """
        mu, log_sigma = self.forward(z, c)
        eps  = torch.randn_like(mu)
        return (mu + eps * log_sigma.exp()).clamp(-1, 1)


# ── Construcción del flujo NSF ─────────────────────────────────────────────────
def _build_nsf(
    features: int = D_FLOW,
    context_features: int = CTX_DIM,
    num_flow_steps: int = FLOW_STEPS,
    hidden_features: int = HIDDEN_FEAT,
    num_bins: int = NUM_BINS,
    tail_bound: float = TAIL_BOUND,
) -> Flow:
    transforms = []
    for step in range(num_flow_steps):
        transforms.append(RandomPermutation(features=features))

        mask = nflows.utils.create_alternating_binary_mask(
            features=features, even=(step % 2 == 0)
        )

        def _make_resnet(
            in_f: int,
            out_f: int,
            ctx: int = context_features,
            h: int = hidden_features,
        ) -> ResidualNet:
            return ResidualNet(
                in_features=in_f,
                out_features=out_f,
                hidden_features=h,
                context_features=ctx,
                num_blocks=2,
                activation=torch.nn.functional.silu,
                dropout_probability=0.0,
                use_batch_norm=False,
            )

        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=_make_resnet,
                num_bins=num_bins,
                tails="linear",
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
    Modelo generativo condicional basado en Neural Spline Flows (cNSF).

    Interfaz pública idéntica al ConditionalVAE (drop-in replacement):
      - forward(x, c)         → log_prob (B,)
      - sample(c, n_samples)  → (n_samples, T, F)
      - log_prob(x, c)        → log_prob (B,)
      - training_loss(x, c)   → (total, nll, recon)
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

        self.context_net  = ContextNet(c_dim=C_DIM, ctx_dim=ctx_dim)
        self.encoder_proj = EncoderProjection(d_stats=D_STATS, d_flow=d_flow)
        self.flow = _build_nsf(
            features=d_flow,
            context_features=ctx_dim,
            num_flow_steps=flow_steps,
            hidden_features=hidden_features,
            num_bins=num_bins,
        )
        self.decoder = TripDecoder(d_flow=d_flow, c_dim=C_DIM, t=T, f=F)

    # ── API pública ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Log-verosimilitud exacta de x dado c. → (B,)"""
        stats = extract_segment_stats(x)
        z     = self.encoder_proj(stats)
        ctx   = self.context_net(c)
        return self.flow.log_prob(z, context=ctx)

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Alias semántico de forward()."""
        return self.forward(x, c)

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Genera n_samples viajes sintéticos condicionados a c.

        Args:
            c:         (C_DIM,) o (1, C_DIM)
            n_samples: número de viajes a generar
        Returns:
            trips: (n_samples, T, F) en [-1, 1]
        """
        self.eval()
        device = next(self.parameters()).device

        if c.dim() == 1:
            c = c.unsqueeze(0)
        c = c.to(device)

        ctx = self.context_net(c)
        z   = self.flow.sample(n_samples, context=ctx).squeeze(0)  # (n_samples, D_FLOW)

        c_expanded = c.expand(n_samples, -1)
        return self.decoder.sample(z, c_expanded)  # ← estocástico

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def training_loss(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pérdida combinada: L = NLL_flow + λ * L_recon

        NLL_flow = -mean(log p_θ(z|c))
        L_recon  = NLL gaussiana ponderada:
                   Σ w_f * [ 0.5 * ((x-mu)/sigma)² + log_sigma ]

        Pesos por feature:
          velocidad (idx 0): ×2
          consumo   (idx 1): ×4  →  peso efectivo = λ × 4 = 8.0

        Returns:
            total_loss, nll (detached), recon_loss (detached)
        """
        stats = extract_segment_stats(x)
        z     = self.encoder_proj(stats)
        ctx   = self.context_net(c)

        log_prob = self.flow.log_prob(z, context=ctx)
        nll      = -log_prob.mean()

        mu, log_sigma = self.decoder(z, c)
        sigma = log_sigma.exp()

        weights    = torch.ones(F, device=x.device)
        weights[0] = 2.0
        weights[1] = 4.0
        recon_loss = (weights * (
            0.5 * ((x - mu) / sigma).pow(2) + log_sigma
        )).mean()

        total_loss = nll + self.recon_weight * recon_loss
        return total_loss, nll.detach(), recon_loss.detach()

    def count_parameters(self) -> dict[str, int]:
        """Desglose de parámetros por componente."""
        def count(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "context_net":  count(self.context_net),
            "encoder_proj": count(self.encoder_proj),
            "flow":         count(self.flow),
            "decoder":      count(self.decoder),
            "total":        count(self),
        }