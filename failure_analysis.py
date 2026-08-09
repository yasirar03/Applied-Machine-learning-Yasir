# Failure-Mode / Per-Class Analysis
#
# The resit brief explicitly wants more than accuracy tables: "why and how
# a model behaves in a certain way, analysis of failure modes, and any
# biases the model may exhibit." This script computes, for all three
# methods (zero-shot, weighted ensemble, CoOp) on the same test images:
#
#   - per-class accuracy (which specific classes each method is weak on)
#   - top confusion (for each class, what it's most often mistaken for)
#   - delta analysis: which classes IMPROVE and which REGRESS when moving
#     from zero-shot -> CoOp - a method being better on average doesn't
#     mean it's better on every class, and the regressions are a genuine
#     failure mode worth discussing, not just the overall accuracy gain.
#
# All three methods are evaluated on the SAME cached image features (image
# encoder is frozen and identical across all methods), so differences are
# entirely attributable to the text/classification side.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common.datasets import get_dataset, get_dataloader, TEST_TRANSFORM
from common.clip_utils import load_clip_model, encode_text_prompts, IMAGENET_TEMPLATES, DEVICE
from common.metrics import save_json, plot_per_class_accuracy

from weighted_prompt_ensemble_model import WeightedPromptEnsemble
from coop_model import CoOpCLIP


RESULTS_DIR = "results"


@torch.no_grad()
def encode_all_images(clip_model, samples, transform, device, batch_size=128):
    loader = get_dataloader(samples, transform, batch_size=batch_size, shuffle=False)
    clip_model.eval()
    feats, labels = [], []
    for images, lbls in loader:
        images = images.to(device)
        f = clip_model.encode_image(images)
        f = F.normalize(f.float(), dim=-1)
        feats.append(f.cpu())
        labels.append(lbls)
    return torch.cat(feats).to(device), torch.cat(labels).to(device)


def per_class_accuracy_and_confusion(preds, labels, n_classes):
    """Returns per-class accuracy dict, and for each class the single most
    common WRONG prediction (its "usual confusion partner")."""
    per_class_acc = {}
    confusion_top = {}
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    for c in range(n_classes):
        mask = labels_np == c
        n = mask.sum()
        if n == 0:
            per_class_acc[c] = None
            continue
        correct = (preds_np[mask] == c).sum()
        per_class_acc[c] = 100.0 * correct / n

        wrong_preds = preds_np[mask][preds_np[mask] != c]
        if len(wrong_preds) > 0:
            vals, counts = np.unique(wrong_preds, return_counts=True)
            confusion_top[c] = int(vals[np.argmax(counts)])

    return per_class_acc, confusion_top


def get_zero_shot_classifier(clip_model, classnames, device):
    return encode_text_prompts(clip_model, classnames, IMAGENET_TEMPLATES, device).float().to(device)


def get_weighted_ensemble_classifier(clip_model, classnames, dataset, device):
    ckpt_path = f"{RESULTS_DIR}/checkpoints/weighted_prompt_{dataset}_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] No weighted-ensemble checkpoint for {dataset}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    model = WeightedPromptEnsemble(clip_model, classnames, ckpt["templates"], device).to(device)
    model.prompt_logits.data = ckpt["prompt_logits"]
    return model.get_weighted_text_features().detach()


def get_coop_classifier(clip_model, classnames, dataset, tag, device):
    ckpt_path = f"{RESULTS_DIR}/checkpoints/coop_{dataset}_{tag}_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] No CoOp checkpoint for {dataset} at tag '{tag}'")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CoOpCLIP(
        classnames, clip_model,
        n_ctx=ckpt["n_ctx"],
        ctx_init=ckpt.get("ctx_init"),
        class_specific_context=ckpt.get("csc", False),
        device=device,
    ).to(device)
    model.prompt_learner.ctx.data = ckpt["ctx"]
    return model.get_text_classifier().detach()


def evaluate_classifier(image_features, labels, classifier, n_classes):
    logits = 100.0 * image_features @ classifier
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item() * 100
    per_class_acc, confusion = per_class_accuracy_and_confusion(preds, labels, n_classes)
    return acc, per_class_acc, confusion


def plot_delta_chart(delta_dict, classnames, title, save_path, top_n=15):
    """Horizontal bar chart of the classes with the biggest accuracy
    gain/loss between two methods - the losses are the interesting part."""
    items = sorted(delta_dict.items(), key=lambda x: x[1])
    combined = items[:top_n] + items[-top_n:]
    # de-dupe in case top_n overlaps (small class counts)
    seen = set()
    unique_combined = []
    for c, d in combined:
        if c not in seen:
            unique_combined.append((c, d))
            seen.add(c)
    unique_combined.sort(key=lambda x: x[1])

    names = [classnames[i] for i, _ in unique_combined]
    deltas = [d for _, d in unique_combined]
    colors = ["crimson" if d < 0 else "seagreen" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, max(4, len(unique_combined) * 0.3)))
    ax.barh(range(len(unique_combined)), deltas, color=colors)
    ax.set_yticks(range(len(unique_combined)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy change (percentage points)")
    ax.set_title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def analyze_dataset(dataset, coop_tag, clip_model, device):
    print(f"\n{'=' * 70}")
    print(f"Failure-mode analysis -- {dataset.upper()}")
    print("=" * 70)

    _, _, test_samples, classnames = get_dataset(dataset)
    n_classes = len(classnames)

    print("Encoding test images (once, shared across all methods)...")
    image_features, labels = encode_all_images(clip_model, test_samples, TEST_TRANSFORM, device)

    method_raw = {}

    zs_classifier = get_zero_shot_classifier(clip_model, classnames, device)
    method_raw["zero_shot"] = evaluate_classifier(image_features, labels, zs_classifier, n_classes)

    we_classifier = get_weighted_ensemble_classifier(clip_model, classnames, dataset, device)
    if we_classifier is not None:
        method_raw["weighted_ensemble"] = evaluate_classifier(image_features, labels, we_classifier, n_classes)

    coop_classifier = get_coop_classifier(clip_model, classnames, dataset, coop_tag, device)
    if coop_classifier is not None:
        method_raw["coop"] = evaluate_classifier(image_features, labels, coop_classifier, n_classes)

    results = {"dataset": dataset, "n_classes": n_classes, "methods": {}}

    for name, (acc, per_class_acc, confusion) in method_raw.items():
        valid = [(i, v) for i, v in per_class_acc.items() if v is not None]
        valid.sort(key=lambda x: x[1])
        worst10 = [(classnames[i], v) for i, v in valid[:10]]
        best10 = [(classnames[i], v) for i, v in valid[-10:]]
        confusion_named = {classnames[c]: classnames[w] for c, w in confusion.items()}

        results["methods"][name] = {
            "overall_acc": round(acc, 4),
            "per_class_acc": {classnames[i]: v for i, v in per_class_acc.items()},
            "worst_10_classes": worst10,
            "best_10_classes": best10,
            "top_confusions": confusion_named,
        }

        plot_per_class_accuracy(
            per_class_acc, classnames, name.replace("_", " ").title(), dataset,
            save_path=f"{RESULTS_DIR}/plots/perclass_{name}_{dataset}.png"
        )

        print(f"\n[{name}] overall: {acc:.2f}%")
        print(f"  Worst classes: {', '.join(f'{c} ({v:.0f}%)' for c, v in worst10[:5])}")

    # Delta: zero-shot -> CoOp, per class - the genuinely interesting part
    if "zero_shot" in method_raw and "coop" in method_raw:
        zs_pc = method_raw["zero_shot"][1]
        coop_pc = method_raw["coop"][1]
        delta = {
            c: (coop_pc[c] - zs_pc[c])
            for c in zs_pc
            if zs_pc[c] is not None and coop_pc.get(c) is not None
        }
        results["zero_shot_to_coop_delta"] = {classnames[i]: round(d, 2) for i, d in delta.items()}

        plot_delta_chart(
            delta, classnames,
            f"{dataset.upper()}: Per-Class Accuracy Change, Zero-Shot -> CoOp",
            save_path=f"{RESULTS_DIR}/plots/delta_zeroshot_to_coop_{dataset}.png"
        )

        gains = sorted(delta.items(), key=lambda x: -x[1])[:5]
        losses = sorted(delta.items(), key=lambda x: x[1])[:5]
        print("\nBiggest gains (Zero-Shot -> CoOp):")
        for i, d in gains:
            print(f"  {classnames[i]:<30} +{d:.1f}pp")
        print("Biggest losses (Zero-Shot -> CoOp) - genuine failure mode, not just noise if large:")
        for i, d in losses:
            print(f"  {classnames[i]:<30} {d:.1f}pp")

    save_json(results, f"{RESULTS_DIR}/tables/failure_analysis_{dataset}.json")

    print(f"\nOverall accuracy summary -- {dataset}:")
    for name, (acc, _, _) in method_raw.items():
        print(f"  {name:<20} {acc:.2f}%")

    return results


def run(datasets, coop_tag):
    clip_model, _ = load_clip_model("ViT-B/16", DEVICE)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    all_results = {}
    for ds in datasets:
        try:
            all_results[ds] = analyze_dataset(ds, coop_tag, clip_model, DEVICE)
        except Exception as e:
            print(f"\n[ERROR] Skipping '{ds}': {e}")
            print("        (continuing with remaining datasets)\n")

    save_json(all_results, f"{RESULTS_DIR}/tables/failure_analysis_all.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                         default=["caltech101", "oxfordpets", "dtd", "eurosat"])
    parser.add_argument("--coop_tag", type=str, default="nctx16",
                         help="Must match the tag used when training CoOp "
                              "(see 04_train_coop.py's checkpoint naming)")
    args = parser.parse_args()
    run(args.datasets, args.coop_tag)
