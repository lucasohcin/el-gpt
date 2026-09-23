"""
Training Loop & Engine for El GPT 1.0.
Features:
- Apple Silicon Metal (MPS) acceleration & CPU fallback
- AdamW optimizer with cosine learning rate schedule & warmup
- Dynamic loss tracking, tokens/sec calculation
- Checkpoint saving & real-time metric callbacks for UI
"""

import argparse
import math
import os
import time
from typing import Callable, Dict, Optional
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from .model import ElGPTConfig, ElGPTModel
    from .tokenizer import ElGPTTokenizer
    from .dataset import create_dataloader
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.model import ElGPTConfig, ElGPTModel
    from engine.tokenizer import ElGPTTokenizer
    from engine.dataset import create_dataloader


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    train_file: str,
    val_file: Optional[str] = None,
    output_dir: str = "checkpoints",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    max_seq_len: int = 256,
    weight_decay: float = 0.01,
    device: Optional[torch.device] = None,
    scale: str = "10m",
    callback: Optional[Callable[[Dict], None]] = None,
    should_stop_fn: Optional[Callable[[], bool]] = None,
) -> str:
    """
    Trains El GPT with chosen scale (10m, 100m, 500m) and saves the resulting model checkpoint.
    """
    if device is None:
        device = get_default_device()

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = ElGPTTokenizer()

    if scale == "1b":
        config = ElGPTConfig.preset_1b(vocab_size=tokenizer.vocab_size)
    elif scale == "500m":
        config = ElGPTConfig.preset_500m(vocab_size=tokenizer.vocab_size)
    elif scale == "100m":
        config = ElGPTConfig.preset_100m(vocab_size=tokenizer.vocab_size)
    else:
        config = ElGPTConfig.preset_10m(vocab_size=tokenizer.vocab_size)
    config.max_seq_len = max_seq_len

    model = ElGPTModel(config).to(device)
    latest_ckpt = os.path.join(output_dir, f"el_gpt_scratch_{scale}.pt")
    if not os.path.exists(latest_ckpt):
        latest_ckpt = os.path.join(output_dir, "el_gpt_1.0_latest.pt")

    if os.path.exists(latest_ckpt):
        try:
            saved = torch.load(latest_ckpt, map_location=device, weights_only=False)
            if saved.get("vocab_size") == tokenizer.vocab_size and model.get_num_params() == sum(p.numel() for p in saved["model_state_dict"].values() if p.ndim > 0):
                model.load_state_dict(saved["model_state_dict"])
                print(f"[El GPT Training] Resuming training from checkpoint: {latest_ckpt}")
        except Exception as e:
            print(f"[El GPT Training] Starting fresh model ({scale}): {e}")

    train_loader = create_dataloader(
        train_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        shuffle=True,
    )

    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=learning_rate * 0.1)

    print(f"[El GPT 1.0 Training] Device: {device} | Total Parameters: {model.get_num_params():,}")
    print(f"[El GPT 1.0 Training] Steps: {total_steps} across {epochs} epochs (batch size: {batch_size})")

    global_step = 0
    start_time = time.time()
    loss_history = []

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if should_stop_fn and should_stop_fn():
                print("[El GPT 1.0 Training] Stop requested by user.")
                break

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = model(input_ids, targets=targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1

            loss_history.append(round(loss_val, 4))
            elapsed = time.time() - start_time
            steps_per_sec = global_step / max(elapsed, 0.001)
            tokens_processed = global_step * batch_size * max_seq_len
            tokens_per_sec = tokens_processed / max(elapsed, 0.001)

            status_payload = {
                "epoch": epoch,
                "total_epochs": epochs,
                "step": global_step,
                "total_steps": total_steps,
                "loss": round(loss_val, 4),
                "avg_epoch_loss": round(epoch_loss / (batch_idx + 1), 4),
                "lr": round(optimizer.param_groups[0]["lr"], 7),
                "speed_tok_s": round(tokens_per_sec, 1),
                "speed_step_s": round(steps_per_sec, 2),
                "elapsed_s": round(elapsed, 1),
                "loss_history": loss_history[-60:],  # last 60 points for UI graph
            }

            if callback:
                callback(status_payload)

            if global_step % 5 == 0 or global_step == 1:
                print(
                    f"Epoch {epoch}/{epochs} | Step {global_step}/{total_steps} | "
                    f"Loss: {loss_val:.4f} | {tokens_per_sec:.0f} tok/s"
                )

        if should_stop_fn and should_stop_fn():
            break

    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, "el_gpt_1.0_latest.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "vocab_size": tokenizer.vocab_size,
            "epochs_completed": epoch,
            "total_steps": global_step,
            "final_loss": loss_history[-1] if loss_history else 0.0,
        },
        checkpoint_path,
    )
    print(f"[El GPT 1.0 Training] Checkpoint successfully saved to {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train El GPT 1.0")
    parser.add_argument("--train-file", type=str, default="data/train.jsonl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    train_model(
        train_file=args.train_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
