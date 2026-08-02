
#Weighted Prompt Ensemble Training

# Train learnable prompt weights on training data.
# Only prompt weights are updated. CLIP is frozen.



import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common.datasets import get_dataset, get_dataloader, TRAIN_TRANSFORM, TEST_TRANSFORM
from common.clip_utils import load_clip_model, IMAGENET_TEMPLATES, DEVICE
from common.metrics import save_json, plot_training_curves

from weighted_prompt_ensemble_model import WeightedPromptEnsemble


RESULTS_DIR = "results"


@torch.no_grad()
def encode_all_images(clip_model, samples, transform, device, batch_size=64):
    """Encode every image ONCE with the frozen CLIP encoder and cache the
    features + labels as tensors. This is the expensive step - do it a
    single time, not once per epoch.

    batch_size kept modest (64, not 128+) and the MPS cache is cleared
    periodically to reduce sustained peak load - matters on fanless
    hardware (e.g. MacBook Air) where large sustained bursts contribute to
    thermal throttling."""
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
    """Evaluate on cached features - no image encoding, just matmuls."""
    model.eval()
    text_classifier = model.get_weighted_text_features().detach()

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, features.size(0), batch_size):
            feats = features[i:i + batch_size]
            lbls = labels[i:i + batch_size]
            logits = 100.0 * feats @ text_classifier
            preds = logits.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    return 100.0 * correct / total


def evaluate_live(model, dataloader):
    """Original evaluation path: encodes images on the fly. Used for the
    --no_cache fallback."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def train_weighted_prompt_ensemble(args):
    os.makedirs(f"{RESULTS_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/plots", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/tables", exist_ok=True)

    print(f"\nWeighted Prompt Ensemble -- {args.dataset.upper()}")
    print(f"Mode: {'cached features (fast)' if not args.no_cache else 'live augmented (slow)'}")
    print("=" * 60)

    clip_model, _ = load_clip_model("ViT-B/16", DEVICE)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    train_samples, val_samples, test_samples, classnames = get_dataset(
        args.dataset, shots=args.shots
    )
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    print(f"Classes: {len(classnames)}")

    templates = IMAGENET_TEMPLATES[:args.num_prompts]

    model = WeightedPromptEnsemble(
        clip_model=clip_model,
        classnames=classnames,
        templates=templates,
        device=DEVICE
    ).to(DEVICE)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Number of prompt templates: {len(templates)}")

    optimizer = Adam([model.prompt_logits], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []
    best_val = 0.0
    best_ckpt = f"{RESULTS_DIR}/checkpoints/weighted_prompt_{args.dataset}_best.pt"

    t_start = time.time()

    if not args.no_cache:
        # Fast path: cache features once, train on tensors 
        print("\nEncoding training images (once)...")
        train_feats, train_labels = encode_all_images(clip_model, train_samples, TEST_TRANSFORM, DEVICE)
        print("Encoding validation images (once)...")
        val_feats, val_labels = encode_all_images(clip_model, val_samples, TEST_TRANSFORM, DEVICE)
        print(f"Encoding done in {time.time() - t_start:.1f}s\n")

        n = train_feats.size(0)
        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, args.batch_size):
                idx = perm[i:i + args.batch_size]
                feats = train_feats[idx]
                labels = train_labels[idx]

                text_classifier = model.get_weighted_text_features()
                logits = 100.0 * feats @ text_classifier
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / n_batches
            val_acc = evaluate_cached(model, val_feats, val_labels)

            train_losses.append(avg_loss)
            val_accs.append(val_acc)

            if val_acc > best_val:
                best_val = val_acc
                torch.save({
                    "prompt_logits": model.prompt_logits.data,
                    "templates": templates,
                    "dataset": args.dataset,
                    "val_acc": best_val,
                    "args": vars(args)
                }, best_ckpt)

            if epoch % args.log_every == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

        print(f"\nTotal training time: {time.time() - t_start:.1f}s")

        print("\nEncoding test images (once)...")
        test_feats, test_labels = encode_all_images(clip_model, test_samples, TEST_TRANSFORM, DEVICE)
        ckpt = torch.load(best_ckpt, map_location=DEVICE)
        model.prompt_logits.data = ckpt["prompt_logits"]
        test_acc = evaluate_cached(model, test_feats, test_labels)

    else:
        # Original live-augmented path (kept for comparison) 
        train_loader = get_dataloader(train_samples, TRAIN_TRANSFORM, batch_size=args.batch_size, shuffle=True)
        val_loader = get_dataloader(val_samples, TEST_TRANSFORM, batch_size=128, shuffle=False)
        test_loader = get_dataloader(test_samples, TEST_TRANSFORM, batch_size=128, shuffle=False)

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            val_acc = evaluate_live(model, val_loader)

            train_losses.append(avg_loss)
            val_accs.append(val_acc)

            if val_acc > best_val:
                best_val = val_acc
                torch.save({
                    "prompt_logits": model.prompt_logits.data,
                    "templates": templates,
                    "dataset": args.dataset,
                    "val_acc": best_val,
                    "args": vars(args)
                }, best_ckpt)

            if epoch % args.log_every == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

        print(f"\nTotal training time: {time.time() - t_start:.1f}s")

        ckpt = torch.load(best_ckpt, map_location=DEVICE)
        model.prompt_logits.data = ckpt["prompt_logits"]
        test_acc = evaluate_live(model, test_loader)

    print("\nFinal Results")
    print("-" * 40)
    print(f"Best Val Acc: {best_val:.2f}%")
    print(f"Test Acc:     {test_acc:.2f}%")

    learned_weights = torch.softmax(model.prompt_logits.detach().cpu(), dim=0)
    top_weights = sorted(zip(templates, learned_weights.tolist()), key=lambda x: x[1], reverse=True)[:10]

    print("\nTop learned prompt templates:")
    for template, weight in top_weights:
        print(f"{weight:.4f} | {template}")

    result = {
        "method": "Weighted Prompt Ensemble",
        "dataset": args.dataset,
        "shots": args.shots,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_prompts": args.num_prompts,
        "cached_features": not args.no_cache,
        "best_val_acc": round(best_val, 4),
        "test_acc": round(test_acc, 4),
        "top_prompt_weights": [{"template": t, "weight": round(w, 6)} for t, w in top_weights]
    }

    save_json(result, f"{RESULTS_DIR}/tables/weighted_prompt_{args.dataset}.json")

    plot_training_curves(
        train_losses, val_accs,
        method_name="Weighted Prompt Ensemble",
        dataset_name=args.dataset,
        save_path=f"{RESULTS_DIR}/plots/weighted_prompt_{args.dataset}_training.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_prompts", type=int, default=80)
    parser.add_argument("--log_every", type=int, default=1,
                         help="Print train/val metrics every N epochs (default: every epoch)")
    parser.add_argument("--no_cache", action="store_true",
                         help="Fall back to the original slow path: re-encode images with "
                              "TRAIN_TRANSFORM augmentation on every epoch. Useful as an "
                              "ablation comparison (cached/no-aug vs live/augmented) but "
                              "much slower.")

    args = parser.parse_args()
    train_weighted_prompt_ensemble(args)
