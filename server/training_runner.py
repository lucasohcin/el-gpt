"""
Background Training Runner and Coordinator for El GPT 1.0.
Coordinates asynchronous training sessions and exposes real-time telemetry.
"""

import threading
import time
from typing import Any, Dict, List, Optional

from engine.train import train_model


class TrainingCoordinator:
    def __init__(self):
        self._lock = threading.Lock()
        self.is_training: bool = False
        self._stop_requested: bool = False
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.current_step: int = 0
        self.total_steps: int = 0
        self.current_loss: float = 0.0
        self.speed_tok_s: float = 0.0
        self.speed_step_s: float = 0.0
        self.lr: float = 0.0
        self.loss_history: List[Dict[str, Any]] = []
        self.status_message: str = "Idle (Model ready for training or inference)"
        self.last_checkpoint: Optional[str] = "checkpoints/el_gpt_1.0_latest.pt"
        self._thread: Optional[threading.Thread] = None

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_training": self.is_training,
                "epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "step": self.current_step,
                "total_steps": self.total_steps,
                "loss": self.current_loss,
                "speed_tok_s": self.speed_tok_s,
                "speed_step_s": self.speed_step_s,
                "lr": self.lr,
                "loss_history": list(self.loss_history),
                "status_message": self.status_message,
                "last_checkpoint": self.last_checkpoint,
            }

    def start_training(
        self,
        train_file: str = "data/train_massive.jsonl",
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 5e-4,
        scale: str = "10m",
    ) -> bool:
        with self._lock:
            if self.is_training:
                return False
            self.is_training = True
            self._stop_requested = False
            self.current_epoch = 0
            self.total_epochs = epochs
            self.current_step = 0
            self.total_steps = 0
            self.current_loss = 0.0
            self.loss_history = []
            self.status_message = f"Initializing training run ({scale.upper()} architecture)..."

        def _worker():
            try:
                def _callback(metrics: Dict[str, Any]):
                    with self._lock:
                        self.current_epoch = metrics["epoch"]
                        self.total_epochs = metrics["total_epochs"]
                        self.current_step = metrics["step"]
                        self.total_steps = metrics["total_steps"]
                        self.current_loss = metrics["loss"]
                        self.speed_tok_s = metrics["speed_tok_s"]
                        self.speed_step_s = metrics["speed_step_s"]
                        self.lr = metrics["lr"]
                        self.loss_history.append({
                            "step": metrics["step"],
                            "loss": metrics["loss"],
                        })
                        if len(self.loss_history) > 120:
                            self.loss_history.pop(0)
                        self.status_message = f"Training {scale.upper()}: Epoch {self.current_epoch}/{self.total_epochs} (Step {self.current_step}/{self.total_steps})"

                def _should_stop() -> bool:
                    return self._stop_requested

                ckpt = train_model(
                    train_file=train_file,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    scale=scale,
                    callback=_callback,
                    should_stop_fn=_should_stop,
                )

                with self._lock:
                    self.last_checkpoint = ckpt
                    self.status_message = "Training complete! Checkpoint saved & loaded."
            except Exception as e:
                with self._lock:
                    self.status_message = f"Training encountered error: {str(e)}"
            finally:
                with self._lock:
                    self.is_training = False

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        return True

    def stop_training(self):
        with self._lock:
            if self.is_training:
                self._stop_requested = True
                self.status_message = "Stopping training... saving checkpoint."


coordinator = TrainingCoordinator()
