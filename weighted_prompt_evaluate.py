
# Evaluate Weighted Prompt Ensemble checkpoints across datasets.


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch

from common.datasets import get_dataset, get_dataloader, TEST_TRANSFORM, ALL_DATASETS
from common.clip_utils import load_clip_model, DEVICE
from common.metrics import save_json

from weighted_prompt_ensemble_model import WeightedPromptEnsemble


RESULTS_DIR = "results"


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()

    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def evaluate_checkpoint(dataset_name, clip_model):
    ckpt_path = f"{RESULTS_DIR}/checkpoints/weighted_prompt_{dataset_name}_best.pt"

    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Missing checkpoint: {ckpt_path}")
        return None

    _, _, test_samples, classnames = get_dataset(dataset_name)

    test_loader = get_dataloader(
        test_samples,
        TEST_TRANSFORM,
        batch_size=128,
        shuffle=False
    )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model = WeightedPromptEnsemble(
        clip_model=clip_model,
        classnames=classnames,
        templates=ckpt["templates"],
        device=DEVICE
    ).to(DEVICE)

    model.prompt_logits.data = ckpt["prompt_logits"]

    acc = evaluate(model, test_loader)

    print(f"{dataset_name:<20} Test Acc: {acc:.2f}%")

    return round(acc, 2)


def run(args):
    if args.dataset == "all":
        datasets = ALL_DATASETS
    else:
        datasets = [args.dataset]

    # Load CLIP ONCE and reuse across all datasets, instead of reloading the full ViT-B/16 backbone from disk on every iteration of the loop below.
    clip_model, _ = load_clip_model("ViT-B/16", DEVICE)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    results = {}

    for ds in datasets:
        acc = evaluate_checkpoint(ds, clip_model)

        if acc is not None:
            results[ds] = acc

    save_json(
        {
            "method": "Weighted Prompt Ensemble",
            "results": results
        },
        f"{RESULTS_DIR}/tables/weighted_prompt_all_results.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="all")

    args = parser.parse_args()

    run(args)
