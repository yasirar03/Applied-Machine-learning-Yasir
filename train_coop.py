# Train CoOp: Context Optimization for CLIP


# Follows the original CoOp training recipe: SGD + momentum, cosine annealing with a short linear warmup (rather than Adam, which is what the weighted-ensemble script uses) - closer to Zhou et al.'s setup.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common.datasets import get_dataset, get_dataloader, TEST_TRANSFORM
from common.clip_utils import load_clip_model, DEVICE
from common.metrics import save_json, plot_training_curves

from coop_model import CoOpCLIP


RESULTS_DIR = "results"


@torch.no_grad()
def encode_all_images(clip_model, samples, transform, device, batch_size=64):
    """Encode every image ONCE with the frozen CLIP encoder and cache
    features + labels. Batch size kept modest and MPS cache cleared
    periodically - see weighted-ensemble script for why."""
    loader = get_dataloader(samples, transform, batch_size=batch_size, shuffle=False)
    clip_model.eval()

    all_features, all_labels = [], []
    for i, (images, labels) in enumerate(tqdm(loader, desc="  Encoding images", leave=False)):
        images = images.to(device)
        features = clip_model.encode_image(images)
        features = F.normalize(features.float(), dim=-1)
        all_features.append(features.cpu())
        all_labels.append(labels)
        if device == "mps" and i % 10 == 0:
            torch.mps.empty_cache()

    return torch.cat(all_features).to(device), torch.cat(all_labels).to(device)


def evaluate_cached(model, features, labels, batch_size=512):
    model.eval()
    with torch.no_grad():
        text_classifier = model.get_text_classifier()
        correct, total = 0, 0
        for i in range(0, features.size(0), batch_size):
            feats = features[i:i + batch_size]
            lbls = labels[i:i + batch_size]
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * feats @ text_classifier
            preds = logits.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return 100.0 * correct / total


def train_coop(args):
    os.makedirs(f"{RESULTS_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/plots", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/tables", exist_ok=True)

    tag = f"nctx{args.n_ctx}" + ("_csc" if args.csc else "") + (f"_init" if args.ctx_init else "")
    print(f"\nCoOp -- {args.dataset.upper()} -- {tag}")
    print("=" * 60)

    clip_model, _ = load_clip_model("ViT-B/16", DEVICE)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    train_samples, val_samples, test_samples, classnames = get_dataset(
        args.dataset, shots=args.shots
    )
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    print(f"Classes: {len(classnames)} | n_ctx: {args.n_ctx} | CSC: {args.csc} | "
          f"ctx_init: {args.ctx_init or 'random'}")

    model = CoOpCLIP(
        classnames=classnames,
        clip_model=clip_model,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        class_specific_context=args.csc,
        device=DEVICE,
    ).to(DEVICE)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    optimizer = SGD([model.prompt_learner.ctx], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    t_start = time.time()
    print("\nEncoding training images (once)...")
    train_feats, train_labels = encode_all_images(clip_model, train_samples, TEST_TRANSFORM, DEVICE)
    print("Encoding validation images (once)...")
    val_feats, val_labels = encode_all_images(clip_model, val_samples, TEST_TRANSFORM, DEVICE)
    print(f"Encoding done in {time.time() - t_start:.1f}s\n")

    train_losses, val_accs = [], []
    best_val = 0.0
    best_ckpt = f"{RESULTS_DIR}/checkpoints/coop_{args.dataset}_{tag}_best.pt"

    n = train_feats.size(0)
    warmup_epochs = 1
    warmup_lr = 1e-5

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()

        # Short linear warmup, as in the original CoOp recipe, to avoid destabilising the randomly-initialised context vectors with a full-size SGD step on epoch 1.
        if epoch <= warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = warmup_lr

        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            feats = train_feats[idx]
            labels = train_labels[idx]

            text_classifier = model.get_text_classifier()
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * feats @ text_classifier
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch > warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / n_batches
        val_acc = evaluate_cached(model, val_feats, val_labels)
        train_losses.append(avg_loss)
        val_accs.append(val_acc)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", val_acc=f"{val_acc:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "ctx": model.prompt_learner.ctx.data,
                "n_ctx": args.n_ctx,
                "csc": args.csc,
                "ctx_init": args.ctx_init,
                "dataset": args.dataset,
                "val_acc": best_val,
                "args": vars(args),
            }, best_ckpt)

        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print(f"\nTotal training time: {time.time() - t_start:.1f}s")

    print("\nEncoding test images (once)...")
    test_feats, test_labels = encode_all_images(clip_model, test_samples, TEST_TRANSFORM, DEVICE)
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.prompt_learner.ctx.data = ckpt["ctx"]
    test_acc = evaluate_cached(model, test_feats, test_labels)

    print("\nFinal Results")
    print("-" * 40)
    print(f"Best Val Acc: {best_val:.2f}%")
    print(f"Test Acc:     {test_acc:.2f}%")

    result = {
        "method": "CoOp",
        "dataset": args.dataset,
        "shots": args.shots,
        "epochs": args.epochs,
        "lr": args.lr,
        "n_ctx": args.n_ctx,
        "csc": args.csc,
        "ctx_init": args.ctx_init,
        "best_val_acc": round(best_val, 4),
        "test_acc": round(test_acc, 4),
    }
    save_json(result, f"{RESULTS_DIR}/tables/coop_{args.dataset}_{tag}.json")

    plot_training_curves(
        train_losses, val_accs,
        method_name=f"CoOp ({tag})",
        dataset_name=args.dataset,
        save_path=f"{RESULTS_DIR}/plots/coop_{args.dataset}_{tag}_training.png"
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256,
                         help="Image batch size for gradient steps. Kept LARGE by default "
                              "(unlike the weighted-ensemble script's 32) because the "
                              "expensive part of each step is the text-transformer forward "
                              "over all classes, which runs once per step regardless of "
                              "image batch size - fewer steps per epoch = proportionally "
                              "fewer of those forwards, with little downside on small "
                              "few-shot datasets. Set lower only if you hit memory issues.")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--ctx_init", type=str, default=None,
                         help="e.g. 'a photo of a' to init context from a handcrafted "
                              "phrase's embeddings instead of random init")
    parser.add_argument("--csc", action="store_true",
                         help="class-specific context (separate ctx per class)")
    parser.add_argument("--log_every", type=int, default=5)

    args = parser.parse_args()
    train_coop(args)
