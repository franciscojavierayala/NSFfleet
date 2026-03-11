"""
train/trainer.py
Bucle de entrenamiento del cVAE con KL annealing y early stopping.
"""

import torch
import torch.optim as optim
from pathlib import Path
from model.cvae import ConditionalVAE, cvae_loss


def kl_weight_schedule(epoch: int, warmup_epochs: int = 20, max_weight: float = 0.5) -> float:
    if epoch >= warmup_epochs:
        return max_weight
    return max_weight * (epoch / warmup_epochs)


class Trainer:
    def __init__(
        self,
        model: ConditionalVAE,
        train_loader,
        val_loader,
        lr: float = 1e-4,
        warmup_epochs: int = 20,
        patience: int = 5,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.warmup_epochs = warmup_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=patience, factor=0.5
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_recon": [], "val_recon": [],
            "train_kl": [], "val_kl": [],
            "kl_weight": [],
        }
        self.best_val_loss  = float("inf")
        self.best_val_recon = float("inf")   # ← métrica principal para guardar checkpoint

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        kl_w = kl_weight_schedule(epoch, self.warmup_epochs)
        total_loss = recon_loss = kl_loss = 0.0
        n = 0
        for trips, conditions in self.train_loader:
            trips      = trips.to(self.device)
            conditions = conditions.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, mu, log_var = self.model(trips, conditions)
            loss, recon, kl = cvae_loss(reconstruction, trips, mu, log_var, kl_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            recon_loss += recon.item()
            kl_loss    += kl.item()
            n += 1
        return {"loss": total_loss/n, "recon": recon_loss/n, "kl": kl_loss/n}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        kl_w = kl_weight_schedule(epoch, self.warmup_epochs)
        total_loss = recon_loss = kl_loss = 0.0
        n = 0
        for trips, conditions in self.val_loader:
            trips      = trips.to(self.device)
            conditions = conditions.to(self.device)
            reconstruction, mu, log_var = self.model(trips, conditions)
            loss, recon, kl = cvae_loss(reconstruction, trips, mu, log_var, kl_w)
            total_loss += loss.item()
            recon_loss += recon.item()
            kl_loss    += kl.item()
            n += 1
        return {"loss": total_loss/n, "recon": recon_loss/n, "kl": kl_loss/n}

    def train(self, epochs: int = 50, verbose: bool = True) -> dict:
        print(f"Entrenando en: {self.device}")
        print(f"Parámetros del modelo: {sum(p.numel() for p in self.model.parameters()):,}\n")

        for epoch in range(epochs):
            kl_w          = kl_weight_schedule(epoch, self.warmup_epochs)
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)

            self.scheduler.step(val_metrics["recon"])  # scheduler sobre recon, no loss total

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_recon"].append(train_metrics["recon"])
            self.history["val_recon"].append(val_metrics["recon"])
            self.history["train_kl"].append(train_metrics["kl"])
            self.history["val_kl"].append(val_metrics["kl"])
            self.history["kl_weight"].append(kl_w)

            # ── Guardar mejor modelo basándose en val_recon (no val_loss) ──────
            # val_loss incluye KL que sube durante el warmup → métrica inestable
            # val_recon mide la calidad real de reconstrucción
            if val_metrics["recon"] < self.best_val_recon:
                self.best_val_recon = val_metrics["recon"]
                self.best_val_loss  = val_metrics["loss"]
                self.save_checkpoint("best_model.pt")

            if verbose:
                marker = " ✓" if val_metrics["recon"] < self.best_val_recon * 1.001 else ""
                print(
                    f"Época {epoch+1:3d}/{epochs} | "
                    f"KL_w={kl_w:.3f} | "
                    f"Train: loss={train_metrics['loss']:.4f} "
                    f"(recon={train_metrics['recon']:.4f}, kl={train_metrics['kl']:.4f}) | "
                    f"Val: recon={val_metrics['recon']:.4f} loss={val_metrics['loss']:.4f}{marker}"
                )

        # Recargar el mejor modelo al final
        ckpt_path = self.checkpoint_dir / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"\n  ✓ Mejor modelo recargado desde: {ckpt_path}")
            print(f"    best_val_recon : {self.best_val_recon:.4f}")
            print(f"    best_val_loss  : {self.best_val_loss:.4f}")

        return self.history

    def save_checkpoint(self, filename: str):
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss":        self.best_val_loss,
            "best_val_recon":       self.best_val_recon,   # ← guardado correctamente
            "history":              self.history,
        }, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_val_loss  = ckpt.get("best_val_loss",  float("inf"))
        self.best_val_recon = ckpt.get("best_val_recon", float("inf"))
        self.history        = ckpt.get("history", self.history)
        print(f"Checkpoint cargado: {path} (best_val_recon={self.best_val_recon:.4f})")
