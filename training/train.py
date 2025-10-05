# src/train.py
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data
import DataLoader, Dataset


def parse_args():
    p = argparse.ArgumentParser()
    # SageMaker will pass your hyperparameters here
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    # Channels mounted by SageMaker (use if you’re reading real data)
    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--val",   type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    # Output/model dirs
    p.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Replace Dummy() with your real dataset using args.train / args.val
    train_ds, val_ds = Dummy(5000), Dummy(1000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)

    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(train_loader)
        print(f"epoch={epoch} train_loss={avg:.4f}")  # CloudWatch-ready logs

        # quick val
        model.eval(); correct = n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item(); n += y.numel()
        print(f"epoch={epoch} val_acc={(correct/n):.4f}")

        # optional: save periodic checkpoints to model_dir (resilient w/ Spot)
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"checkpoint-epoch{epoch}.pt"))

    # Final model—SageMaker will upload everything in model_dir to S3
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))

    # (Optional) write metrics for downstream steps
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"val_acc": correct/n}, f)

if __name__ == "__main__":
    main()
