# Compare Methods

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.metrics import load_json, save_json, plot_comparison_chart, build_results_table

RESULTS_DIR = "results"


def run(datasets, coop_tag=None):
    zero_shot = load_json(f"{RESULTS_DIR}/tables/zero_shot_all_results.json")["results"]
    zero_shot = {r["dataset"]: r for r in zero_shot}

    weighted = load_json(f"{RESULTS_DIR}/tables/weighted_prompt_all_results.json")["results"]

    results_dict = {
        "Zero-Shot (mean of templates)": {},
        "Zero-Shot (equal-weight ensemble)": {},
        "Weighted Prompt Ensemble (learned)": {},
    }
    if coop_tag:
        results_dict[f"CoOp ({coop_tag})"] = {}

    for ds in datasets:
        if ds not in zero_shot:
            print(f"[SKIP] No zero-shot results for {ds}")
            continue
        if ds not in weighted:
            print(f"[SKIP] No weighted-ensemble results for {ds}")
            continue

        results_dict["Zero-Shot (mean of templates)"][ds] = zero_shot[ds]["mean_acc"]
        results_dict["Zero-Shot (equal-weight ensemble)"][ds] = zero_shot[ds]["ensemble_acc"]
        results_dict["Weighted Prompt Ensemble (learned)"][ds] = weighted[ds]

        if coop_tag:
            coop_path = f"{RESULTS_DIR}/tables/coop_{ds}_{coop_tag}.json"
            if os.path.exists(coop_path):
                results_dict[f"CoOp ({coop_tag})"][ds] = load_json(coop_path)["test_acc"]
            else:
                print(f"[SKIP] No CoOp results for {ds} at {coop_path}")

  
    results_dict = {m: v for m, v in results_dict.items() if v}

    save_json(results_dict, f"{RESULTS_DIR}/tables/method_comparison.json")

    plot_comparison_chart(
        results_dict,
        save_path=f"{RESULTS_DIR}/plots/method_comparison.png",
        title="Method Comparison Across Datasets"
    )

    build_results_table(
        results_dict,
        save_path=f"{RESULTS_DIR}/tables/method_comparison_table.tex",
        title="Method Comparison"
    )


if __name__ == "__main__":
    
    DATASETS = ["caltech101", "oxfordpets", "dtd", "eurosat"]
    COOP_TAG = "nctx16"
    run(DATASETS, coop_tag=COOP_TAG)
