#!/usr/bin/env python3
import os, json, math, argparse, time, warnings, random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import Dir2Dataset  # assumes PYTHONPATH includes your package
from transformations import augment_from_config, default_augmentation

# --------------------------
# Helpers
# --------------------------

def resolve_data_dir(cli_arg: str | None, env_key: str, default: str | None = None) -> str:
    """Prefer CLI arg; else SM env; else default."""
    if cli_arg:
        return cli_arg
    v = os.environ.get(env_key)
    if v:
        return v
    if default:
        return default
    raise ValueError(f"Must provide data dir via --{env_key.lower().replace('sm_channel_', '')} or {env_key}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False, warn_only=True)

def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: dict, model_dir: str, name: str = "checkpoint.pt"):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    tmp = Path(model_dir) / (name + ".tmp")
    final = Path(model_dir) / name
    torch.save(state, tmp)
    os.replace(tmp, final)  # atomic


def load_checkpoint_if_exists(model: nn.Module, optimizer: torch.optim.Optimizer, model_dir: str):
    ckpt = Path(model_dir) / "checkpoint.pt"
    if ckpt.exists():
        data = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        start_epoch = data.get("epoch", 0) + 1
        best = data.get("best_val", None)
        print(f"[resume] loaded {ckpt} (epoch={start_epoch}, best_val={best})")
        return start_epoch, best
    return 0, None


def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    num = 2.0 * (pred * target).sum([1,2])
    den = (pred + target).sum([1,2]) + eps
    return 1.0 - (num / den).mean()


# --------------------------
# Dataloaders
# --------------------------


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

# --------------------------
# Train / Eval
# --------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    running = {"dice": 0.0, "bce": 0.0}
    loss_bce = nn.BCEWithLogitsLoss()
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True).float()
        logits = model(xb)[:, 1, ...]
        bce = loss_bce(logits, yb)
        d = dice_loss(logits, yb)
        running["dice"] += (1.0 - d).item() * xb.size(0)  # report Dice score
        running["bce"]  += bce.item() * xb.size(0)
        n += xb.size(0)
    for k in running:
        running[k] /= max(1, n)
    return running

def train_one_epoch(model, loader, device, optimizer, scaler=None):
    model.train()
    loss_bce = nn.BCEWithLogitsLoss()
    running = {"dice": 0.0, "bce": 0.0}
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)[:, 1, ...]
        bce = loss_bce(logits, yb)
        d = dice_loss(logits, yb)
        loss = bce + d
        loss.backward()
        optimizer.step()

        running["dice"] += (1.0 - d).item() * xb.size(0)
        running["bce"]  += bce.item() * xb.size(0)
        n += xb.size(0)
    for k in running:
        running[k] /= max(1, n)
    return running

# --------------------------
# Main
# --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Segmentation training")
    # Data / I/O
    p.add_argument("--train-dir", type=str, default=None, help="train root")
    p.add_argument("--val-dir",   type=str, default=None, help="val root")
    p.add_argument("--images-path", type=str, default="images")
    p.add_argument("--labels-path", type=str, default="labels")
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./outputs"))
    p.add_argument("--model-filename", type=str, default=None)
    # HParams
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    # Misc
    p.add_argument("--logdir", type=str, default="./runs")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    # Allow SageMaker env fallbacks (works unchanged inside Training jobs)
    train_root = resolve_data_dir(args.train_dir, "SM_CHANNEL_TRAIN")
    val_root   = resolve_data_dir(args.val_dir,   "SM_CHANNEL_VAL", default=train_root)
    model_dir  = args.model_dir

    seed_everything(args.seed)
    device = default_device()
    print(f"[device] {device}")

    # Datasets/Loaders
    train_ds = Dir2Dataset(
        root=train_root,
        images_path=args.images_path,
        labels_path=args.labels_path,
        augment=default_augmentation(),                 # TODO: add Albumentations pipeline
        return_paths=False,
    )
    val_ds = Dir2Dataset(
        root=val_root,
        images_path=args.images_path,
        labels_path=args.labels_path,
        augment=default_augmentation(),
        return_paths=False,
    )
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = make_loader(val_ds,   args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model/Opt
    model = torch.jit.load(
        os.path.join(model_dir, args.model_filename),
        map_location=device
    ).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Resume if requested
    start_epoch, best_val = (0, None)
    if args.resume:
        start_epoch, best_val = load_checkpoint_if_exists(model, optimizer, model_dir)

    # Logging
    writer = SummaryWriter(log_dir=args.logdir)

    # Train
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, device, optimizer)
        val_metrics   = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"epoch {epoch:03d} | "
              f"train dice={train_metrics['dice']:.4f} bce={train_metrics['bce']:.4f} | "
              f"val dice={val_metrics['dice']:.4f} bce={val_metrics['bce']:.4f} | "
              f"{dt:.1f}s")

        # TB scalars
        writer.add_scalar("train/dice", train_metrics["dice"], epoch)
        writer.add_scalar("train/bce",  train_metrics["bce"], epoch)
        writer.add_scalar("val/dice",   val_metrics["dice"], epoch)
        writer.add_scalar("val/bce",    val_metrics["bce"], epoch)

        # Save (keep best by val dice)
        cur = val_metrics["dice"]
        if best_val is None or cur > best_val:
            best_val = cur
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "best_val": best_val, "hparams": vars(args)},
                model_dir, name="checkpoint.pt"
            )

    # Final model export (SageMaker will tarball everything in model_dir)
    torch.save(model.state_dict(), Path(model_dir) / "model_final.pt")
    with open(Path(model_dir) / "metrics.json", "w") as f:
        json.dump({"best_val_dice": best_val}, f)

    writer.close()

if __name__ == "__main__":
    main()
