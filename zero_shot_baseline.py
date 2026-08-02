# Zero-Shot CLIP Baseline

# For each dataset:
#   - evaluates zero-shot accuracy using each of the (up to) 80 handcrafted CLIP prompt templates individually
#   - reports mean / std across templates (-> "plot the variance and mean of the test accuracy", as required by the spec)
#   - also reports the standard equal-weight ensemble accuracy (all template embeddings averaged per class),

# CLIP is used purely in inference mode here - nothing is trained.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip

from common.datasets import get_dataset, get_dataloader, TEST_TRANSFORM, ALL_DATASETS
from common.clip_utils import load_clip_model, IMAGENET_TEMPLATES, DEVICE
from common.metrics import save_json


RESULTS_DIR = "results"


@torch.no_grad()
def build_text_classifier(clip_model, classnames, template, device):
    """Single-template zero-shot classifier: (feature_dim, num_classes)."""
    classnames = [c.replace("_", " ") for c in classnames]
    prompts = [template.format(c) for c in classnames]
    tokens = clip.tokenize(prompts, truncate=True).to(device)

    text_features = clip_model.encode_text(tokens)
    text_features = F.normalize(text_features.float(), dim=-1)

    return text_features.T


@torch.no_grad()
def build_ensemble_text_classifier(clip_model, classnames, templates, device):
    """Equal-weight ensemble: average normalised text embeddings across all
    templates per class, then re-normalise. Standard CLIP zero-shot recipe."""
    classnames = [c.replace("_", " ") for c in classnames]
    all_class_features = []

    for c in classnames:
        prompts = [t.format(c) for t in templates]
        tokens = clip.tokenize(prompts, truncate=True).to(device)

        class_embeddings = clip_model.encode_text(tokens)
        class_embeddings = F.normalize(class_embeddings.float(), dim=-1)

        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = F.normalize(class_embedding, dim=0)

        all_class_features.append(class_embedding)

    return torch.stack(all_class_features, dim=1)  # (feature_dim, num_classes)


@torch.no_grad()
def encode_all_images(clip_model, dataloader, device):
    """Encode every image in the dataloader ONCE and cache the features.
    This is the expensive step - do it a single time per dataset, not once
    per template."""
    clip_model.eval()

    all_features = []
    all_labels = []

    n_batches = len(dataloader)
    for i, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)

        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)

        all_features.append(image_features.cpu())
        all_labels.append(labels)

        if i % 5 == 0 or i == n_batches:
            print(f"  Encoding images: batch {i}/{n_batches}")

    return torch.cat(all_features, dim=0).to(device), torch.cat(all_labels, dim=0).to(device)


def evaluate_cached(image_features, labels, text_classifier):
    """Reuse cached image features against a given text classifier - no
    image encoding happens here, so this is essentially free."""
    logits = 100.0 * image_features @ text_classifier
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return 100.0 * correct / labels.size(0)


def run_zero_shot_baseline(dataset_name, num_prompts, clip_model, device):
    print(f"\nZero-Shot Baseline -- {dataset_name.upper()}")
    print("=" * 60)

    _, _, test_samples, classnames = get_dataset(dataset_name)
    test_loader = get_dataloader(test_samples, TEST_TRANSFORM, batch_size=128, shuffle=False)

    templates = IMAGENET_TEMPLATES[:num_prompts]
    print(f"Classes: {len(classnames)} | Templates: {len(templates)} | Test samples: {len(test_samples)}")

    print("Encoding test images once (cached for all templates)...")
    image_features, labels = encode_all_images(clip_model, test_loader, device)

    per_template_acc = []

    for i, template in enumerate(templates):
        text_classifier = build_text_classifier(clip_model, classnames, template, device)
        acc = evaluate_cached(image_features, labels, text_classifier)
        per_template_acc.append(acc)
        print(f"[{i + 1:02d}/{len(templates)}] {acc:5.2f}%  | {template}")

    per_template_acc = np.array(per_template_acc)
    mean_acc = float(per_template_acc.mean())
    std_acc = float(per_template_acc.std())
    best_idx = int(per_template_acc.argmax())
    worst_idx = int(per_template_acc.argmin())

    print("\nEnsembling all templates (equal weight)...")
    ensemble_classifier = build_ensemble_text_classifier(clip_model, classnames, templates, device)
    ensemble_acc = evaluate_cached(image_features, labels, ensemble_classifier)

    print("\nResults")
    print("-" * 40)
    print(f"Mean template acc: {mean_acc:.2f}% (+/- {std_acc:.2f})")
    print(f"Best template:     {per_template_acc[best_idx]:.2f}% | {templates[best_idx]}")
    print(f"Worst template:    {per_template_acc[worst_idx]:.2f}% | {templates[worst_idx]}")
    print(f"Equal-weight ensemble acc: {ensemble_acc:.2f}%")

    result = {
        "method": "Zero-Shot CLIP Baseline",
        "dataset": dataset_name,
        "num_templates": len(templates),
        "per_template_acc": [round(a, 4) for a in per_template_acc.tolist()],
        "templates": templates,
        "mean_acc": round(mean_acc, 4),
        "std_acc": round(std_acc, 4),
        "best_template": {"template": templates[best_idx], "acc": round(float(per_template_acc[best_idx]), 4)},
        "worst_template": {"template": templates[worst_idx], "acc": round(float(per_template_acc[worst_idx]), 4)},
        "ensemble_acc": round(ensemble_acc, 4),
    }

    os.makedirs(f"{RESULTS_DIR}/tables", exist_ok=True)
    save_json(result, f"{RESULTS_DIR}/tables/zero_shot_{dataset_name}.json")

    return result


def plot_variance_across_datasets(all_results, save_path):
    """Bar chart with error bars: mean +/- std template accuracy per dataset,
    with the equal-weight ensemble accuracy marked as a point on top."""
    datasets = [r["dataset"] for r in all_results]
    means = [r["mean_acc"] for r in all_results]
    stds = [r["std_acc"] for r in all_results]
    ensembles = [r["ensemble_acc"] for r in all_results]

    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, yerr=stds, capsize=5, color="#8ecae6", label="Mean template acc (+/- std)")
    ax.scatter(x, ensembles, color="#e63946", zorder=5, label="Equal-weight ensemble acc")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Zero-Shot CLIP: Mean/Variance Across Handcrafted Prompts")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved plot to {save_path}")


def plot_per_template_distribution(all_results, save_path):
    """Boxplot of per-template accuracy distribution, one box per dataset."""
    datasets = [r["dataset"] for r in all_results]
    data = [r["per_template_acc"] for r in all_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=datasets, showmeans=True)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Distribution of Zero-Shot Accuracy Across Prompt Templates")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def run(args):
    clip_model, _ = load_clip_model("ViT-B/16", DEVICE)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    if args.dataset == "all":
        datasets = args.datasets_list if args.datasets_list else ALL_DATASETS
    else:
        datasets = [args.dataset]

    all_results = []
    for ds in datasets:
        try:
            result = run_zero_shot_baseline(ds, args.num_prompts, clip_model, DEVICE)
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Skipping '{ds}' due to failure: {e}")
            print(f"        (continuing with remaining datasets)\n")

    if not all_results:
        print("\n[FAIL] No datasets completed successfully - nothing to save.")
        return

    save_json(
        {"method": "Zero-Shot CLIP Baseline", "results": all_results},
        f"{RESULTS_DIR}/tables/zero_shot_all_results.json"
    )

    plot_variance_across_datasets(all_results, f"{RESULTS_DIR}/plots/zero_shot_mean_variance.png")
    plot_per_template_distribution(all_results, f"{RESULTS_DIR}/plots/zero_shot_template_boxplot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                         help="Single dataset name, or 'all' to use --datasets_list / ALL_DATASETS")
    parser.add_argument("--datasets_list", type=str, nargs="+", default=None,
                         help="Explicit subset to run when --dataset all, e.g. "
                              "--datasets_list caltech101 oxford_pets dtd eurosat")
    parser.add_argument("--num_prompts", type=int, default=80)

    args = parser.parse_args()
    run(args)
