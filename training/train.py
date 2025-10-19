#!/usr/bin/env python3
import os, json, math, argparse, time, warnings, random
from pathlib import Path
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import Dir2Dataset  # assumes PYTHONPATH includes your package
from transformations import augment_from_config, default_augmentation
from unet import UNet

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


def save_job_comment(comment: str, hparams: dict, best_val_dice: float):
    """Save job comment and metadata to S3 experiment log."""
    try:
        import boto3
        from io import StringIO

        s3 = boto3.client('s3')
        bucket = 'vessel-segmentation-data'
        key = 'experiments/training_log.txt'

        # Get job name from environment or use timestamp
        job_name = os.environ.get('TRAINING_JOB_NAME', f"local-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

        # Create log entry
        log_entry = f"\n{'='*80}\n"
        log_entry += f"Job: {job_name}\n"
        log_entry += f"Timestamp: {timestamp}\n"
        log_entry += f"Comment: {comment}\n"
        log_entry += f"Best Val Dice: {best_val_dice:.4f}\n"
        log_entry += f"Hyperparameters:\n"
        for key_name, value in sorted(hparams.items()):
            if key_name not in ['job_comment']:  # Don't duplicate comment
                log_entry += f"  {key_name}: {value}\n"
        log_entry += f"{'='*80}\n"

        # Try to append to existing log, or create new
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            existing = obj['Body'].read().decode('utf-8')
            new_content = existing + log_entry
        except s3.exceptions.NoSuchKey:
            new_content = "Training Job Log\n" + "="*80 + log_entry

        # Upload updated log
        s3.put_object(Bucket=bucket, Key=key, Body=new_content, ContentType='text/plain')
        print(f"[Job comment saved to s3://{bucket}/{key}]")

    except Exception as e:
        print(f"[Warning] Could not save job comment to S3: {e}")


def dice_loss_from_logits(
    logits: torch.Tensor,        # (B, 2, H, W)
    target: torch.Tensor,        # (B, H, W) with {0,1}
    eps: float = 1e-6,
    fg_class: int = 1,
    ignore_index: int | None = None,
    reduce: str = "mean",
):
    # probs for foreground
    probs = F.softmax(logits, dim=1)[:, fg_class, ...]  # (B,H,W)

    # build mask (optionally ignore pixels)
    if ignore_index is not None:
        valid = (target != ignore_index)
        tgt_fg = (target == fg_class) & valid
        probs = probs * valid.float()
    else:
        tgt_fg = (target == fg_class)

    tgt_fg = tgt_fg.float()

    dims = tuple(range(1, probs.dim()))
    intersection = (probs * tgt_fg).sum(dim=dims)
    denominator  = (probs + tgt_fg).sum(dim=dims).clamp_min(eps)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    loss = 1.0 - dice
    return loss.mean() if reduce == "mean" else loss


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
    running = {"dice": 0.0, "ce": 0.0}
    loss_ce = nn.CrossEntropyLoss(torch.Tensor([1.0, 5.0]).to(device))
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).long()
        logits = model(xb)
        ce = loss_ce(logits, yb)
        d = dice_loss_from_logits(logits, yb)
        running["dice"] += (1.0 - d).item() * xb.size(0)  # report Dice score
        running["ce"]  += ce.item() * xb.size(0)
        n += xb.size(0)
    for k in running:
        running[k] /= max(1, n)
    return running

def train_one_epoch(model, loader, device, optimizer, scaler=None):
    model.train()
    loss_ce = nn.CrossEntropyLoss(torch.Tensor([1.0, 5.0]).to(device))
    running = {"dice": 0.0, "ce": 0.0}
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        ce = loss_ce(logits, yb)
        d = dice_loss_from_logits(logits, yb)
        loss = ce + d
        loss.backward()
        optimizer.step()

        running["dice"] += (1.0 - d).item() * xb.size(0)
        running["ce"]  += ce.item() * xb.size(0)
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
    p.add_argument("--config-file", type=str, default=None)
    # HParams
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    # Misc
    p.add_argument("--logdir", type=str, default="./runs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--job-comment", type=str, default=None, help="Optional comment about this training run")
    # Parse known args only (ignore SageMaker-injected args)
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[Warning] Ignoring unknown arguments: {unknown}")
    return args

def main():
    args = parse_args()

    # Allow SageMaker env fallbacks (works unchanged inside Training jobs)
    train_root = resolve_data_dir(args.train_dir, "SM_CHANNEL_TRAIN")
    val_root   = resolve_data_dir(args.val_dir,   "SM_CHANNEL_VAL", default=train_root)
    model_dir  = args.model_dir

    # Handle config file from SageMaker channel (if it's a directory, look for config inside)
    config_file = args.config_file
    if not config_file:
        # Try to get from SageMaker environment
        config_file = os.environ.get("SM_CHANNEL_CONFIG")

    if config_file and os.path.isdir(config_file):
        # SageMaker downloads single-file channels to a directory
        config_file = os.path.join(config_file, 'default_config.json')

    if not config_file:
        raise ValueError("Must provide --config-file or set SM_CHANNEL_CONFIG environment variable")

    seed_everything(args.seed)
    device = default_device()

    print(f"[device] {device}")
    print(f"[train_root] {train_root}")
    print(f"[val_root] {val_root}")
    print(f"[model_dir] {model_dir}")
    print(f"[config_file] {config_file}")

    # Datasets/Loaders
    train_ds = Dir2Dataset(
        root=train_root,
        images_path=args.images_path,
        labels_path=args.labels_path,
        augment=augment_from_config(config_file),
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
    model = UNet(
        in_ch=1, num_classes=2,
        enc=(32, 32, 64, 128, 128, 128),
        dec=(128, 128, 64, 32, 16),
        p_drop=0.5
    )
    model = model.to(device)
    print(f"[model] moved to {device}")
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
              f"train dice={train_metrics['dice']:.4f} ce={train_metrics['ce']:.4f} | "
              f"val dice={val_metrics['dice']:.4f} ce={val_metrics['ce']:.4f} | "
              f"{dt:.1f}s")

        # TB scalars
        writer.add_scalar("train/dice", train_metrics["dice"], epoch)
        writer.add_scalar("train/ce",  train_metrics["ce"], epoch)
        writer.add_scalar("val/dice",   val_metrics["dice"], epoch)
        writer.add_scalar("val/ce",    val_metrics["ce"], epoch)

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

    # Save job comment if provided
    if args.job_comment:
        save_job_comment(args.job_comment, vars(args), best_val)

    writer.close()

if __name__ == "__main__":
    main()
