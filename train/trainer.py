"""
train/trainer.py
Bucle de entrenamiento del ConditionalFlowModel (NSF) con early stopping.
    - Pérdida NLL + reconstrucción a través de ConditionalFlowModel.training_loss()
    - Métricas train_nll / val_nll / train_recon / val_recon
    - Checkpoint guardado cuando val_nll mejora (análogo al val_recon anterior)
    - Formato del checkpoint (clave "model_state_dict") — app.py no requiere cambios
    - Optimizer + ReduceLROnPlateau scheduler
    - Gradient clipping en max_norm=1.0
    - Interfaz pública: Trainer(model, train_loader, val_loader, ...)
"""

import torch
import torch.optim as optim
from pathlib import Path
from model.nflow_model import ConditionalFlowModel


class Trainer:
    """
    Entrena un ConditionalFlowModel con pérdida NLL + reconstrucción.

    Args:
        model:          instancia de ConditionalFlowModel
        train_loader:   DataLoader de entrenamiento (viajes, condicionamientos)
        val_loader:     DataLoader de validación
        lr:             tasa de aprendizaje inicial (default 1e-4)
        patience:       épocas sin mejora antes de reducir lr (default 5)
        checkpoint_dir: directorio donde guardar best_model.pt
    """

    def __init__(
        self,
        model: ConditionalFlowModel,
        train_loader,
        val_loader,
        lr: float = 1e-4,
        patience: int = 5,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # AdamW: regularización implícita de pesos, más robusta que Adam para flujos
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=patience, factor=0.5, min_lr=1e-6
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Historial de métricas (sin kl — los flujos no tienen término KL)
        self.history: dict = {
            "train_loss":  [],
            "val_loss":    [],
            "train_nll":   [],
            "val_nll":     [],
            "train_recon": [],
            "val_recon":   [],
        }
        self.best_val_nll: float = float("inf")
        self.best_val_loss: float = float("inf")
        # Fine-tuning: cargar checkpoint previo si existe
        if (self.checkpoint_dir / "best_model.pt").exists():
            self.load_checkpoint("best_model.pt")
    # ── Época de entrenamiento ─────────────────────────────────────────────────
    def _train_epoch(self) -> dict:
        self.model.train()
        total_loss = nll_acc = recon_acc = 0.0
        n = 0

        for trips, conditions in self.train_loader:
            trips = trips.to(self.device)
            conditions = conditions.to(self.device)

            self.optimizer.zero_grad()
            loss, nll, recon = self.model.training_loss(trips, conditions)
            loss.backward()

            # Gradient clipping: esencial en flujos para estabilizar el Jacobiano
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            nll_acc += nll.item()
            recon_acc += recon.item()
            n += 1

        return {"loss": total_loss / n, "nll": nll_acc / n, "recon": recon_acc / n}

    # ── Época de validación ────────────────────────────────────────────────────
    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.model.eval()
        total_loss = nll_acc = recon_acc = 0.0
        n = 0

        for trips, conditions in self.val_loader:
            trips = trips.to(self.device)
            conditions = conditions.to(self.device)

            loss, nll, recon = self.model.training_loss(trips, conditions)

            total_loss += loss.item()
            nll_acc += nll.item()
            recon_acc += recon.item()
            n += 1

        return {"loss": total_loss / n, "nll": nll_acc / n, "recon": recon_acc / n}

    # ── Bucle principal ────────────────────────────────────────────────────────
    def train(self, epochs: int = 100, verbose: bool = True) -> dict:
        """
        Ejecuta el bucle de entrenamiento completo.

        Args:
            epochs:  número máximo de épocas
            verbose: imprime métricas por época

        Returns:
            history: dict con listas de métricas por época
        """
        param_info = self.model.count_parameters()
        print(f"Entrenando en: {self.device}")
        print(f"Parámetros del modelo: {param_info['total']:,}")
        print(f"  context_net:  {param_info['context_net']:,}")
        print(f"  encoder_proj: {param_info['encoder_proj']:,}")
        print(f"  flow (NSF):   {param_info['flow']:,}")
        print(f"  decoder:      {param_info['decoder']:,}\n")

        for epoch in range(epochs):
            train_m = self._train_epoch()
            val_m = self._val_epoch()

            # Scheduler sobre val_nll: equivalente al val_recon del trainer original
            self.scheduler.step(val_m["nll"])

            # Actualizar historial
            self.history["train_loss"].append(train_m["loss"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["train_nll"].append(train_m["nll"])
            self.history["val_nll"].append(val_m["nll"])
            self.history["train_recon"].append(train_m["recon"])
            self.history["val_recon"].append(val_m["recon"])

            # Guardar checkpoint si la NLL de validación mejora
            if val_m["nll"] < self.best_val_nll:
                self.best_val_nll = val_m["nll"]
                self.best_val_loss = val_m["loss"]
                self.save_checkpoint("best_model.pt")

            if verbose:
                marker = " ✓" if val_m["nll"] <= self.best_val_nll * 1.001 else ""
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Época {epoch + 1:3d}/{epochs} | "
                    f"lr={lr_now:.2e} | "
                    f"Train: loss={train_m['loss']:.4f} "
                    f"(nll={train_m['nll']:.4f}, recon={train_m['recon']:.4f}) | "
                    f"Val: nll={val_m['nll']:.4f} loss={val_m['loss']:.4f}{marker}"
                )

        # Recargar el mejor checkpoint al final del entrenamiento
        ckpt_path = self.checkpoint_dir / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"\n  ✓ Mejor modelo recargado desde: {ckpt_path}")
            print(f"    best_val_nll  : {self.best_val_nll:.4f}")
            print(f"    best_val_loss : {self.best_val_loss:.4f}")

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """
        Guarda checkpoint con el mismo formato que el trainer original.
        La clave "model_state_dict" es OBLIGATORIA para compatibilidad con app.py.
        """
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss":        self.best_val_loss,
                "best_val_nll":         self.best_val_nll,
                "history":              self.history,
            },
            self.checkpoint_dir / filename,
        )

    def load_checkpoint(self, filename: str) -> None:
        """Carga un checkpoint guardado previamente."""
        path = self.checkpoint_dir / filename
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_val_nll  = ckpt.get("best_val_nll",  float("inf"))
        self.history       = ckpt.get("history", self.history)
        print(f"Checkpoint cargado: {path} (best_val_nll={self.best_val_nll:.4f})")
