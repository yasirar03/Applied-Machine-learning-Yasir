"""
common/val_reporter.py
=======================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from common.datasets import get_dataset, get_dataloader, TEST_TRANSFORM, ALL_DATASETS
from common.clip_utils import (load_clip_model, IMAGENET_TEMPLATES,
                                encode_text_prompts, DEVICE)
from common.metrics import (save_json, load_json, build_results_table,
                             plot_per_class_accuracy)

RESULTS_DIR = "results"


@torch.no_grad()
def evaluate_full(model_forward_fn, dataloader, n_classes, device=DEVICE):
    """
    Full evaluation: top-1, top-5, per-class accuracy.
    model_forward_fn: callable(images) -> logits (B, n_cls)
    """
    all_preds, all_labels = [], []
    for images, labels in tqdm(dataloader, desc="  Evaluating", leave=False):
        images = images.to(device)
        logits = model_forward_fn(images)
        all_preds.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    # Top-1
    top1_preds = logits.argmax(dim=1)
    top1_acc   = (top1_preds == labels).float().mean().item() * 100

    # Top-5
    top5_preds = logits.topk(min(5, logits.shape[1]), dim=1).indices
    top5_correct = top5_preds.eq(labels.unsqueeze(1)).any(dim=1)
    top5_acc = top5_correct.float().mean().item() * 100

    # Per-class accuracy
    per_class_acc = {}
    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        if mask.sum() == 0:
            per_class_acc[cls_idx] = None
        else:
            per_class_acc[cls_idx] = (
                (top1_preds[mask] == labels[mask]).float().mean().item() * 100
            )

    return {
        "top1": round(top1_acc, 4),
        "top5": round(top5_acc, 4),
        "per_class": per_class_acc,
    }


def report_zeroshot_both(clip_model, dataset_name):
    """Report zero-shot accuracy on val AND test splits, encoding the 80-template
    text classifier only ONCE and reusing it for both (previously this was
    recomputed twice - once per split - which is pure duplicated work since
    the text weights don't depend on the split)."""
    _, val_s, test_s, classnames = get_dataset(dataset_name)

    text_weights = encode_text_prompts(
        clip_model, classnames, IMAGENET_TEMPLATES, DEVICE
    ).float().to(DEVICE)  # (D, n_cls) - cast/move once, not per batch, not per split

    def forward_fn(images):
        img_feat = clip_model.encode_image(images)
        img_feat = F.normalize(img_feat.float(), dim=-1)
        return 100.0 * img_feat @ text_weights

    val_loader  = get_dataloader(val_s,  TEST_TRANSFORM, batch_size=128, shuffle=False)
    test_loader = get_dataloader(test_s, TEST_TRANSFORM, batch_size=128, shuffle=False)

    val_result  = evaluate_full(forward_fn, val_loader,  len(classnames))
    test_result = evaluate_full(forward_fn, test_loader, len(classnames))
    return val_result, test_result


def report_coop(clip_model, dataset_name, n_ctx, split="val"):
    """Report CoOp accuracy on val or test split."""
    ckpt_path = f"{RESULTS_DIR}/checkpoints/coop_{dataset_name}_nctx{n_ctx}_best.pt"
    if not os.path.exists(ckpt_path):
        return None
    from shiva.coop_model import CoOpCLIP

    _, val_s, test_s, classnames = get_dataset(dataset_name)
    samples = val_s if split == "val" else test_s
    loader  = get_dataloader(samples, TEST_TRANSFORM, batch_size=128, shuffle=False)

    model = CoOpCLIP(classnames=classnames, clip_model=clip_model, n_ctx=n_ctx).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.prompt_learner.ctx.data = ckpt["ctx"]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return evaluate_full(model, loader, len(classnames))


def report_cocoop(clip_model, dataset_name, n_ctx, split="val"):
    """Report CoCoOp accuracy on val or test split."""
    ckpt_path = f"{RESULTS_DIR}/checkpoints/cocoop_{dataset_name}_nctx{n_ctx}_best.pt"
    if not os.path.exists(ckpt_path):
        return None
    from aakash.cocoop_model import CoCoOpCLIP

    _, val_s, test_s, classnames = get_dataset(dataset_name)
    samples = val_s if split == "val" else test_s
    loader  = get_dataloader(samples, TEST_TRANSFORM, batch_size=64, shuffle=False)

    model = CoCoOpCLIP(classnames=classnames, clip_model=clip_model, n_ctx=n_ctx).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.prompt_learner.ctx.data = ckpt["ctx"]
    model.prompt_learner.meta_net.load_state_dict(ckpt["meta_net"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return evaluate_full(model, loader, len(classnames))


def report_kgcoop(clip_model, dataset_name, n_ctx, split="val"):
    """
    Report KgCoOp accuracy on val or test split.
    KgCoOp uses the same CoOpCLIP architecture as CoOp -- only the training
    loss differs (adds knowledge-guided regularisation). The checkpoint is
    saved under 'kgcoop_*' naming to distinguish it from CoOp checkpoints.
    """
    ckpt_path = f"{RESULTS_DIR}/checkpoints/kgcoop_{dataset_name}_nctx{n_ctx}_best.pt"
    if not os.path.exists(ckpt_path):
        return None
    from shiva.coop_model import CoOpCLIP

    _, val_s, test_s, classnames = get_dataset(dataset_name)
    samples = val_s if split == "val" else test_s
    loader  = get_dataloader(samples, TEST_TRANSFORM, batch_size=128, shuffle=False)

    model = CoOpCLIP(classnames=classnames, clip_model=clip_model, n_ctx=n_ctx).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.prompt_learner.ctx.data = ckpt["ctx"]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return evaluate_full(model, loader, len(classnames))


def run(args):
    os.makedirs(f"{RESULTS_DIR}/tables", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/plots",  exist_ok=True)

    clip_model, _ = load_clip_model("ViT-B/16", DEVICE)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # Use only the datasets specified (default: 5 trained datasets)
    datasets_to_eval = getattr(args, 'datasets',
                               ["flowers102", "oxfordpets", "food101",
                                "caltech101", "fgvcaircraft"])

    val_summary  = {}
    test_summary = {}

    methods = ["ZeroShot", f"CoOp_nctx{args.n_ctx}",
               f"CoCoOp_nctx{args.n_ctx}", f"KgCoOp_nctx{args.n_ctx}",
               "MaPLe", "LoRA", "Adapter"]
    for m in methods:
        val_summary[m]  = {}
        test_summary[m] = {}

    for ds in datasets_to_eval:
        print(f"\n{'='*55}")
        print(f"Dataset: {ds.upper()}")
        print(f"{'='*55}")

        _, _, _, classnames = get_dataset(ds)

        # Zero-shot: text weights encoded once, reused for val + test
        print("  [ZeroShot] Validation...")
        zs_val, zs_test = report_zeroshot_both(clip_model, ds)
        val_summary["ZeroShot"][ds]  = zs_val["top1"]
        test_summary["ZeroShot"][ds] = zs_test["top1"]
        print(f"    Val Top-1: {zs_val['top1']:.2f}%  "
              f"Val Top-5: {zs_val['top5']:.2f}%  "
              f"Test Top-1: {zs_test['top1']:.2f}%")

        # Per-class plot for zero-shot
        plot_per_class_accuracy(
            zs_val["per_class"], classnames, "ZeroShot", ds,
            save_path=f"{RESULTS_DIR}/plots/val_perclass_zeroshot_{ds}.png"
        )

        # CoOp val
        print(f"  [CoOp n_ctx={args.n_ctx}] Validation...")
        coop_val  = report_coop(clip_model, ds, args.n_ctx, split="val")
        coop_test = report_coop(clip_model, ds, args.n_ctx, split="test")
        if coop_val:
            val_summary[f"CoOp_nctx{args.n_ctx}"][ds]  = coop_val["top1"]
            test_summary[f"CoOp_nctx{args.n_ctx}"][ds] = coop_test["top1"]
            print(f"    Val Top-1: {coop_val['top1']:.2f}%  "
                  f"Val Top-5: {coop_val['top5']:.2f}%  "
                  f"Test Top-1: {coop_test['top1']:.2f}%")
            plot_per_class_accuracy(
                coop_val["per_class"], classnames,
                f"CoOp (n_ctx={args.n_ctx})", ds,
                save_path=f"{RESULTS_DIR}/plots/val_perclass_coop_{ds}.png"
            )
        else:
            print("    [SKIP] No checkpoint found")

        # CoCoOp val
        print(f"  [CoCoOp n_ctx={args.n_ctx}] Validation...")
        cocoop_val  = report_cocoop(clip_model, ds, args.n_ctx, split="val")
        cocoop_test = report_cocoop(clip_model, ds, args.n_ctx, split="test")
        if cocoop_val:
            val_summary[f"CoCoOp_nctx{args.n_ctx}"][ds]  = cocoop_val["top1"]
            test_summary[f"CoCoOp_nctx{args.n_ctx}"][ds] = cocoop_test["top1"]
            print(f"    Val Top-1: {cocoop_val['top1']:.2f}%  "
                  f"Test Top-1: {cocoop_test['top1']:.2f}%")
        else:
            print("    [SKIP] No checkpoint found")

        # KgCoOp val
        print(f"  [KgCoOp n_ctx={args.n_ctx}] Validation...")
        kg_val  = report_kgcoop(clip_model, ds, args.n_ctx, split="val")
        kg_test = report_kgcoop(clip_model, ds, args.n_ctx, split="test")
        if kg_val:
            val_summary[f"KgCoOp_nctx{args.n_ctx}"][ds]  = kg_val["top1"]
            test_summary[f"KgCoOp_nctx{args.n_ctx}"][ds] = kg_test["top1"]
            print(f"    Val Top-1: {kg_val['top1']:.2f}%  "
                  f"Test Top-1: {kg_test['top1']:.2f}%")
        else:
            print("    [SKIP] No checkpoint found")

        # -- Extra credit: MaPLe -----------------------------------------------
        maple_path = f"{RESULTS_DIR}/tables/aakash_maple_{ds}_nctx2.json"
        if os.path.exists(maple_path):
            d = load_json(maple_path)
            val_summary["MaPLe"][ds]  = d.get("best_val_acc", 0)
            test_summary["MaPLe"][ds] = d.get("test_acc", 0)
            print(f"  [MaPLe] Val: {d.get('best_val_acc',0):.2f}%  "
                  f"Test: {d.get('test_acc',0):.2f}%")

        # -- Extra credit: LoRA / Adapter --------------------------------------
        peft_path = f"{RESULTS_DIR}/tables/aakash_peft_{ds}.json"
        if os.path.exists(peft_path):
            d = load_json(peft_path)
            if "results" in d:
                if "LoRA" in d["results"]:
                    val_summary["LoRA"][ds]  = d["results"]["LoRA"]["val_acc"]
                    test_summary["LoRA"][ds] = d["results"]["LoRA"]["test_acc"]
                    print(f"  [LoRA]    Val: {d['results']['LoRA']['val_acc']:.2f}%  "
                          f"Test: {d['results']['LoRA']['test_acc']:.2f}%")
                if "Adapter" in d["results"]:
                    val_summary["Adapter"][ds]  = d["results"]["Adapter"]["val_acc"]
                    test_summary["Adapter"][ds] = d["results"]["Adapter"]["test_acc"]
                    print(f"  [Adapter] Val: {d['results']['Adapter']['val_acc']:.2f}%  "
                          f"Test: {d['results']['Adapter']['test_acc']:.2f}%")

    # -- Save and print tables -------------------------------------------------
    save_json({"val": val_summary, "test": test_summary},
              f"{RESULTS_DIR}/tables/VALIDATION_REPORT.json")

    # Filter to methods with results
    val_filled  = {m: v for m, v in val_summary.items()  if v}
    test_filled = {m: v for m, v in test_summary.items() if v}

    print("\n\n" + "="*70)
    print("VALIDATION ACCURACY TABLE")
    print("="*70)
    build_results_table(val_filled,
                        save_path=f"{RESULTS_DIR}/tables/VAL_accuracy_table.tex")

    print("\n\n" + "="*70)
    print("TEST ACCURACY TABLE")
    print("="*70)
    build_results_table(test_filled,
                        save_path=f"{RESULTS_DIR}/tables/TEST_accuracy_table.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ctx",    type=int, default=16)
    parser.add_argument("--shots",    type=int, default=16)
    parser.add_argument("--datasets", nargs="+",
                        default=["flowers102", "oxfordpets", "food101",
                                 "caltech101", "fgvcaircraft"],
                        help="Datasets to evaluate (default: the 5 trained datasets)")
    args = parser.parse_args()
    run(args)
