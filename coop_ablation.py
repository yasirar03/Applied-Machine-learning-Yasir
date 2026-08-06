# CoOp Ablation Sweep

# Satisfies the resit requirement: "develop components that can be evaluated through ablation studies" (context length, class-specific vs. shared context, and initialisation strategy are the three knobs CoOp itself exposes, per Zhou et al.'s own ablations).

# Runs a small grid on ONE dataset and produces comparison plots.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from types import SimpleNamespace

from common.metrics import save_json, plot_hyperparameter_sweep
from importlib import import_module

train_coop = import_module("04_train_coop").train_coop


def run_sweep(dataset, epochs, shots):
    results = {}

    # Ablation 1: context length
    print("\n" + "#" * 70)
    print("# ABLATION 1: Context length (n_ctx)")
    print("#" * 70)
    n_ctx_results = []
    for n_ctx in [4, 8, 16]:
        args = SimpleNamespace(dataset=dataset, shots=shots, epochs=epochs,
                                batch_size=32, lr=0.002, n_ctx=n_ctx,
                                ctx_init=None, csc=False, log_every=epochs)
        r = train_coop(args)
        n_ctx_results.append(r["test_acc"])
        results[f"n_ctx_{n_ctx}"] = r

    plot_hyperparameter_sweep(
        "n_ctx (context length)", [4, 8, 16], n_ctx_results, dataset,
        save_path=f"results/plots/ablation_{dataset}_n_ctx.png"
    )

    # Ablation 2: class-specific vs. shared context
    print("\n" + "#" * 70)
    print("# ABLATION 2: Class-specific context (CSC)")
    print("#" * 70)
    for csc in [False, True]:
        args = SimpleNamespace(dataset=dataset, shots=shots, epochs=epochs,
                                batch_size=32, lr=0.002, n_ctx=16,
                                ctx_init=None, csc=csc, log_every=epochs)
        r = train_coop(args)
        results[f"csc_{csc}"] = r

    # Ablation 3: initialisation strategy
    print("\n" + "#" * 70)
    print("# ABLATION 3: Context initialisation")
    print("#" * 70)
    for ctx_init in [None, "a photo of a"]:
        args = SimpleNamespace(dataset=dataset, shots=shots, epochs=epochs,
                                batch_size=32, lr=0.002, n_ctx=16,
                                ctx_init=ctx_init, csc=False, log_every=epochs)
        r = train_coop(args)
        key = f"init_{'random' if ctx_init is None else 'handcrafted'}"
        results[key] = r

    save_json(results, f"results/tables/coop_ablation_{dataset}.json")

    print("\n" + "=" * 70)
    print(f"ABLATION SUMMARY -- {dataset.upper()}")
    print("=" * 70)
    for k, v in results.items():
        print(f"{k:<20} Val: {v['best_val_acc']:.2f}%  Test: {v['test_acc']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dtd",
                         help="Run the sweep on ONE dataset (default: dtd - smallest/fastest)")
    parser.add_argument("--epochs", type=int, default=30,
                         help="Fewer epochs than a full CoOp run - this is a comparative "
                              "sweep, not a final-accuracy run")
    parser.add_argument("--shots", type=int, default=16)
    args = parser.parse_args()

    run_sweep(args.dataset, args.epochs, args.shots)
