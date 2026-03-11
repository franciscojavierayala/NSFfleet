"""
model/cvae.py
VAE Condicional sobre series temporales multivariantes.

Arquitectura:
  Encoder: Conv1D → Flatten → Dense → [mu, log_var]
  Decoder: Dense → ConvTranspose1D → secuencia reconstruida
  Condicionamiento: vector c concatenado en encoder y decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
from data.synthetic import T, F, C_DIM


LATENT_DIM = 64


# ── Encoder ───────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Comprime (trip, c) → (mu, log_var) en el espacio latente.
    trip: (B, T, F)
    c:    (B, C_DIM)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Convoluciones 1D sobre la dimensión temporal
        # Input: (B, F + C_DIM, T) — c se concatena como canales extra
        in_channels = F + C_DIM
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Tamaño después de las convoluciones: T // 8
        conv_out_size = 256 * (T // 8)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc_mu      = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)

    def forward(self, trip: torch.Tensor, c: torch.Tensor):
        B = trip.size(0)

        # Expandir c a lo largo del tiempo y concatenar: (B, F+C_DIM, T)
        c_expanded = c.unsqueeze(2).expand(-1, -1, T)          # (B, C_DIM, T)
        x = torch.cat([trip.permute(0, 2, 1), c_expanded], dim=1)  # (B, F+C_DIM, T)

        x = self.conv(x)                    # (B, 256, T//8)
        x = x.view(B, -1)                   # flatten
        x = self.fc(x)

        mu      = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


# ── FiLM conditioning ─────────────────────────────────────────────────────────
class FiLM(nn.Module):
    def __init__(self, channels, c_dim):
        super().__init__()
        self.scale = nn.Linear(c_dim, channels)
        self.shift = nn.Linear(c_dim, channels)

    def forward(self, x, c):
        s = self.scale(c).unsqueeze(2)
        b = self.shift(c).unsqueeze(2)
        return x * s + b


# ── Decoder ───────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        conv_in_size = 256 * (T // 8)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + C_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, conv_in_size),
            nn.ReLU(),
        )
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.film1   = FiLM(128, C_DIM)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.film2   = FiLM(64, C_DIM)
        self.deconv3 = nn.ConvTranspose1d(64, F, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.out_act = nn.Tanh()

    def forward(self, z: torch.Tensor, c: torch.Tensor):
        B = z.size(0)
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = x.view(B, 256, T // 8)
        x = func.relu(self.deconv1(x))
        x = self.film1(x, c)
        x = func.relu(self.deconv2(x))
        x = self.film2(x, c)
        x = self.out_act(self.deconv3(x))
        return x.permute(0, 2, 1)


# ── cVAE completo ─────────────────────────────────────────────────────────────
class ConditionalVAE(nn.Module):
    """
    VAE Condicional completo.

    Modos de uso:
      - Entrenamiento/fine-tuning: forward(trip, c) → reconstrucción + mu + log_var
      - Inferencia (generación):   sample(c, n_samples) → N viajes sintéticos
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick: z = mu + eps * std"""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu   # en eval, usar la media directamente para reconstrucción

    def forward(self, trip: torch.Tensor, c: torch.Tensor):
        mu, log_var = self.encoder(trip, c)
        z = self.reparametrize(mu, log_var)
        reconstruction = self.decoder(z, c)
        return reconstruction, mu, log_var

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Genera n_samples viajes sintéticos para un condicionamiento c dado.
        c: (C_DIM,) o (1, C_DIM)
        Devuelve: (n_samples, T, F)
        """
        self.eval()
        device = next(self.parameters()).device

        if c.dim() == 1:
            c = c.unsqueeze(0)          # (1, C_DIM)
        c = c.to(device).expand(n_samples, -1)   # (n_samples, C_DIM)

        z = torch.randn(n_samples, self.latent_dim, device=device)
        trips = self.decoder(z, c)      # (n_samples, T, F)
        return trips


# ── Función de pérdida ─────────────────────────────────────────────────────────
def cvae_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
    free_bits: float = 0.1,        
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    weights = torch.ones(F, device=original.device)
    weights[0] = 2.0
    weights[1] = 3.0   # consumo peso mayor

    recon_loss = (weights * (reconstruction - original).pow(2)).mean()

    # Free bits: KL por dimensión, con mínimo garantizado
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, latent_dim)
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)             # no puede bajar de 0.5
    kl_loss = kl_per_dim.mean()

    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss
