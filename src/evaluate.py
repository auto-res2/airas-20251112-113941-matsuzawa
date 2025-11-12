import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

PRIMARY_METRIC_STR = "(i) Exact-match on task dev sets; (ii) worst-case per-capability KL overshoot max_i(k_i/b_{cap(i)}); (iii) AUC-to-58 % GSM8K; (iv) drift-efficiency = ΔEM / Σ_i k_i."

# ------------------------------------------------------------------------------------------------------------------
#  CLI ARGUMENTS
# ------------------------------------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Cross-run evaluation & visualisation")
    p.add_argument("results_dir", type=str)
    p.add_argument("run_ids", type=str, help="JSON string list of WandB run IDs")
    return p.parse_args()

# ------------------------------------------------------------------------------------------------------------------
#  LOAD GLOBAL W&B CFG
# ------------------------------------------------------------------------------------------------------------------

def load_wandb_cfg() -> Dict[str, str]:
    cfg = OmegaConf.load(os.path.join("config", "config.yaml"))
    return {"entity": cfg.wandb.entity, "project": cfg.wandb.project}

# ------------------------------------------------------------------------------------------------------------------
#  PER-RUN EXPORT
# ------------------------------------------------------------------------------------------------------------------

def export_single_run(run: "wandb.apis.public.Run", out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    history_df = run.history()  # pandas DataFrame with ALL metrics
    summary = dict(run.summary)
    config = dict(run.config)

    # save comprehensive metrics -------------------------------------------------
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "summary": summary,
            "config": config,
            "history": history_df.to_dict("records")
        }, f, indent=2)
    print(metrics_path)

    # learning curve figure ------------------------------------------------------
    if not history_df.empty and "train_loss" in history_df.columns:
        plt.figure(figsize=(7, 4))
        sns.lineplot(x=history_df.index, y=history_df["train_loss"], label="train_loss")
        for col in ["exact_match_dev", "best_exact_match_dev"]:
            if col in history_df.columns:
                sns.lineplot(x=history_df.index, y=history_df[col], label=col)
        plt.legend(); plt.xlabel("step"); plt.title(run.id); plt.tight_layout()
        lc_path = os.path.join(out_dir, f"{run.id}_learning_curve.pdf")
        plt.savefig(lc_path); plt.close(); print(lc_path)

    # confusion matrix if predictions table present -----------------------------
    try:
        tbl_df = run.history(keys=["dev_predictions"]).tail(1)
        if not tbl_df.empty and isinstance(tbl_df.iloc[0]["dev_predictions"], wandb.Table):
            wb_tbl = tbl_df.iloc[0]["dev_predictions"]
            g, p = [], []
            for row in wb_tbl.data:
                g.append(row[1]); p.append(row[2])
            labels = sorted(set(g) | set(p))
            cm = confusion_matrix(g, p, labels=labels)
            ConfusionMatrixDisplay(cm, display_labels=labels).plot(xticks_rotation=45, cmap="Blues")
            plt.tight_layout()
            cm_path = os.path.join(out_dir, f"{run.id}_confusion_matrix.pdf")
            plt.savefig(cm_path); plt.close(); print(cm_path)
    except Exception:
        pass

# ------------------------------------------------------------------------------------------------------------------
#  SUMMARY AGGREGATION
# ------------------------------------------------------------------------------------------------------------------

def aggregate_summaries(summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    metrics = defaultdict(dict)
    for rid, summ in summaries.items():
        for k, v in summ.items():
            if isinstance(v, (float, int)):
                metrics[k][rid] = v

    primary_key = "best_exact_match_dev" if "best_exact_match_dev" in next(iter(summaries.values())) else "exact_match_dev"
    direction = "min" if any(t in primary_key for t in ["loss", "error", "perplex"]) else "max"

    best_prop = {"run_id": None, "value": -np.inf if direction == "max" else np.inf}
    best_base = {"run_id": None, "value": -np.inf if direction == "max" else np.inf}
    for rid, val in metrics[primary_key].items():
        if "proposed" in rid:
            if (direction == "max" and val > best_prop["value"]) or (direction == "min" and val < best_prop["value"]):
                best_prop = {"run_id": rid, "value": val}
        if any(t in rid for t in ["baseline", "comparative"]):
            if (direction == "max" and val > best_base["value"]) or (direction == "min" and val < best_base["value"]):
                best_base = {"run_id": rid, "value": val}

    gap = np.nan
    if best_prop["run_id"] and best_base["run_id"]:
        if direction == "max":
            gap = (best_prop["value"] - best_base["value"]) / (best_base["value"] + 1e-12) * 100
        else:
            gap = (best_base["value"] - best_prop["value"]) / (best_base["value"] + 1e-12) * 100

    return {
        "primary_metric": PRIMARY_METRIC_STR,
        "primary_key": primary_key,
        "metrics": metrics,
        "best_proposed": best_prop,
        "best_baseline": best_base,
        "gap": gap,
    }

# ------------------------------------------------------------------------------------------------------------------
#  CROSS-RUN FIGURES
# ------------------------------------------------------------------------------------------------------------------

def cross_run_figures(aggr: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    vals_dict = aggr["metrics"][aggr["primary_key"]]
    rids, vals = list(vals_dict.keys()), list(vals_dict.values())

    plt.figure(figsize=(max(6, 0.7 * len(rids)), 4))
    sns.barplot(x=rids, y=vals, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.ylabel(aggr["primary_key"]); plt.title("Cross-run comparison – primary metric"); plt.tight_layout()
    comp_path = os.path.join(out_dir, "comparison_primary_metric_bar_chart.pdf")
    plt.savefig(comp_path); plt.close(); print(comp_path)

    prop_vals = [v for k, v in vals_dict.items() if "proposed" in k]
    base_vals = [v for k, v in vals_dict.items() if any(t in k for t in ["baseline", "comparative"])]
    if len(prop_vals) >= 2 and len(base_vals) >= 2:
        t_stat, p_val = stats.ttest_ind(prop_vals, base_vals, equal_var=False)
        txt_path = os.path.join(out_dir, "ttest_primary_metric.txt")
        with open(txt_path, "w") as f:
            f.write(f"t = {t_stat:.4f}, p = {p_val:.4e}\n")
        print(txt_path)

# ------------------------------------------------------------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    res_dir = os.path.abspath(args.results_dir)
    os.makedirs(res_dir, exist_ok=True)

    run_ids: List[str] = json.loads(args.run_ids)
    wb_cfg = load_wandb_cfg()
    api = wandb.Api()

    summaries: Dict[str, Dict[str, Any]] = {}
    for rid in run_ids:
        run = api.run(f"{wb_cfg['entity']}/{wb_cfg['project']}/{rid}")
        single_out = os.path.join(res_dir, rid)
        export_single_run(run, single_out)
        summaries[rid] = dict(run.summary)

    aggr = aggregate_summaries(summaries)
    comp_dir = os.path.join(res_dir, "comparison"); os.makedirs(comp_dir, exist_ok=True)
    aggr_path = os.path.join(comp_dir, "aggregated_metrics.json")
    with open(aggr_path, "w") as f:
        json.dump(aggr, f, indent=2)
    print(aggr_path)

    cross_run_figures(aggr, comp_dir)

if __name__ == "__main__":
    main()