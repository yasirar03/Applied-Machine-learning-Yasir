

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- safe for subprocess runs
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_mean_variance(per_prompt_accs, dataset_name, save_path=None):
    """
    Plot mean and variance of accuracy across 80 handcrafted prompts.
    per_prompt_accs: list of 80 accuracy values (floats).
    """
    accs = np.array(per_prompt_accs)
    mean_acc = accs.mean()
    std_acc  = accs.std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of all 80 prompts
    axes[0].bar(range(len(accs)), accs, color="steelblue", alpha=0.7)
    axes[0].axhline(mean_acc, color="red", linestyle="--", linewidth=2,
                    label=f"Mean = {mean_acc:.2f}%")
    axes[0].fill_between(range(len(accs)),
                         mean_acc - std_acc, mean_acc + std_acc,
                         alpha=0.2, color="red", label=f"+/-1 Std = {std_acc:.2f}%")
    axes[0].set_xlabel("Prompt Index")
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].set_title(f"{dataset_name} -- Accuracy per Handcrafted Prompt")
    axes[0].legend()

    # Right: histogram
    axes[1].hist(accs, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1].axvline(mean_acc, color="red", linestyle="--", linewidth=2,
                    label=f"Mean = {mean_acc:.2f}%")
    axes[1].set_xlabel("Top-1 Accuracy (%)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{dataset_name} -- Accuracy Distribution")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
    return mean_acc, std_acc


def plot_comparison_chart(results_dict, save_path=None, title=None):
    """
    Bar chart comparing all methods across all datasets.
    results_dict: {method_name: {dataset_name: accuracy}}
    """
    methods   = list(results_dict.keys())
    datasets  = list(next(iter(results_dict.values())).keys())

    df = pd.DataFrame(results_dict).T
    df = df[datasets]

    fig, ax = plt.subplots(figsize=(max(14, len(datasets) * 1.5), 6))
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for i, (method, color) in enumerate(zip(methods, colors)):
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df.loc[method].values, width,
                      label=method, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(title or "Prompt Tuning Methods -- Accuracy Comparison Across Datasets")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def build_results_table(results_dict, save_path=None, title="Results"):
    """
    Build and save a LaTeX-ready results table.
    results_dict: {method_name: {dataset_name: accuracy}}
    """
    df = pd.DataFrame(results_dict).T
    df["Average"] = df.mean(axis=1)
    df = df.round(2)

    print("\n" + "=" * 80)
    print(f"RESULTS TABLE -- {title}")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80 + "\n")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path.replace(".tex", ".csv"))
        latex = df.to_latex(float_format="%.2f", bold_rows=True,
                             caption=f"Accuracy (\\%) -- {title}.",
                             label="tab:results")
        with open(save_path, "w") as f:
            f.write(latex)
        print(f"Saved table: {save_path}")
    return df


def build_top1_top5_table(results_dict, save_path=None):
    """
    Build a combined Top-1 / Top-5 accuracy table.
    results_dict: {method: {dataset: {"top1": x, "top5": y}}}
    Saves both CSV and LaTeX.
    """
    rows = []
    for method, ds_dict in results_dict.items():
        for ds, accs in ds_dict.items():
            rows.append({
                "Method":  method,
                "Dataset": ds,
                "Top-1 (%)": round(accs.get("top1", 0), 2),
                "Top-5 (%)": round(accs.get("top5", 0), 2),
            })
    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("TOP-1 / TOP-5 ACCURACY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path.replace(".tex", ".csv"), index=False)
        latex = df.to_latex(index=False, float_format="%.2f",
                             caption="Top-1 and Top-5 accuracy (\\%) across methods and datasets.",
                             label="tab:top1top5")
        with open(save_path, "w") as f:
            f.write(latex)
        print(f"Saved Top-1/Top-5 table: {save_path}")
    return df


def plot_per_class_accuracy(per_class_dict, classnames, method_name,
                             dataset_name, save_path=None):
    """Bar chart of per-class accuracy -- top-20 and bottom-20 if many classes."""
    classes = [classnames[i] for i in sorted(per_class_dict.keys())
               if per_class_dict.get(i) is not None]
    accs    = [per_class_dict[i] for i in sorted(per_class_dict.keys())
               if per_class_dict.get(i) is not None]

    if len(classes) > 40:
        sorted_pairs = sorted(zip(accs, classes), reverse=True)
        top20   = sorted_pairs[:20]
        bot20   = sorted_pairs[-20:]
        pairs   = top20 + bot20
        accs    = [p[0] for p in pairs]
        classes = [p[1] for p in pairs]
        title_suffix = " (Top-20 & Bottom-20 classes)"
    else:
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.4), 5))
    colors = ["steelblue" if a >= 50 else "coral" for a in accs]
    ax.bar(range(len(classes)), accs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{method_name} -- Per-Class Accuracy\n"
                 f"{dataset_name.upper()}{title_suffix}")
    ax.axhline(np.mean(accs), color="red", linestyle="--",
               label=f"Mean={np.mean(accs):.1f}%")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_training_curves(train_losses, val_accs, method_name, dataset_name,
                          save_path=None, val_top5_accs=None):
    """Plot training loss and validation accuracy curves. Optionally includes top-5."""
    n_plots = 3 if val_top5_accs else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))

    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{method_name} -- {dataset_name} -- Training Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_accs, "r-o", markersize=3, label="Val Top-1 Acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{method_name} -- {dataset_name} -- Val Top-1 Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    if val_top5_accs:
        axes[2].plot(epochs, val_top5_accs, "g-o", markersize=3, label="Val Top-5 Acc")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy (%)")
        axes[2].set_title(f"{method_name} -- {dataset_name} -- Val Top-5 Accuracy")
        axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_hyperparameter_sweep(param_name, param_values, accuracies,
                               dataset_name, save_path=None):
    """Plot accuracy vs. hyperparameter value."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_values, accuracies, "b-o", markersize=8, linewidth=2)
    for x, y in zip(param_values, accuracies):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Hyperparameter Sweep: {param_name} -- {dataset_name}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def compute_topk_accuracy(logits, labels, topk=(1, 5)):
    """Compute top-k accuracy from logits and labels tensors."""
    import torch
    maxk = max(topk)
    batch_size = labels.size(0)
    _, pred = logits.topk(min(maxk, logits.shape[1]), dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    results = {}
    for k in topk:
        k_eff = min(k, logits.shape[1])
        correct_k = correct[:k_eff].reshape(-1).float().sum(0)
        results[f"top{k}"] = (correct_k * 100.0 / batch_size).item()
    return results

