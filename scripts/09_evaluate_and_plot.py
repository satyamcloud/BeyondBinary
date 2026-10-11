"""
Recompute every reported result from the saved fold predictions and redraw the
figures. No GPU or network needed.

Usage:  python 09_evaluate_and_plot.py --results ../results --data ../data/hinglish_veracity_568.csv

Reads  results/muril_kfold_results.json, results/mbert_kfold_results.json
Writes results/model_comparison_results.csv and results/figures/01..05_*.png
Prints per-fold and pooled metrics, the paired t-test and the confusion matrices.
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="../results")
ap.add_argument("--data", default="../data/hinglish_veracity_568.csv")
args = ap.parse_args()
FIG = os.path.join(args.results, "figures")
os.makedirs(FIG, exist_ok=True)
sns.set_style("whitegrid")

muril = json.load(open(os.path.join(args.results, "muril_kfold_results.json")))
mbert = json.load(open(os.path.join(args.results, "mbert_kfold_results.json")))
f1_mu = [r["macro_f1"] for r in muril]
f1_mb = [r["macro_f1"] for r in mbert]
f1_maj = [f1_score(r["true"], [0] * len(r["true"]), average="macro") for r in muril]

table = pd.DataFrame({"fold": range(1, 6), "majority_baseline": f1_maj, "mbert": f1_mb, "muril": f1_mu})
table.to_csv(os.path.join(args.results, "model_comparison_results.csv"), index=False)
print(table.round(4).to_string(index=False))
for n, v in (("majority", f1_maj), ("mBERT", f1_mb), ("MuRIL", f1_mu)):
    print(f"{n}: mean {np.mean(v):.4f}  std (n=5) {np.std(v):.4f}")

t, p = stats.ttest_rel(f1_mu, f1_mb)
diffs = np.array(f1_mu) - np.array(f1_mb)
print(f"paired t-test t={t:.3f} p={p:.3f}; mean diff {diffs.mean():+.3f}; "
      f"range {diffs.min():+.3f}..{diffs.max():+.3f}; MuRIL wins {(diffs > 0).sum()}/5")


def pooled(res):
    t = sum([r["true"] for r in res], [])
    p = sum([r["preds"] for r in res], [])
    return confusion_matrix(t, p, labels=[0, 1]), precision_recall_fscore_support(t, p), f1_score(t, p, average="macro")

cms = {}
for name, res in (("MuRIL", muril), ("mBERT", mbert)):
    cm, prf, mf = pooled(res)
    cms[name] = cm
    print(f"\n{name} pooled confusion matrix {cm.tolist()} macro-F1 {mf:.3f}")
    print("  precision", prf[0].round(3), "recall", prf[1].round(3), "F1", prf[2].round(3))

# Fig 1 dataset overview
df = pd.read_csv(args.data)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
lc = df["binary_label"].value_counts()
ax[0].bar(lc.index, lc.values, color=["#d62728", "#2ca02c"]); ax[0].set_title("Class Distribution", fontweight="bold")
for i, v in enumerate(lc.values):
    ax[0].text(i, v + 5, str(v), ha="center", fontweight="bold")
df.groupby(["source", "binary_label"]).size().unstack(fill_value=0).plot(
    kind="bar", stacked=True, ax=ax[1], color=["#d62728", "#2ca02c"])
ax[1].set_title("Dataset Composition by Source", fontweight="bold"); ax[1].set_xlabel(""); ax[1].tick_params(axis="x", rotation=0)
ax[2].hist(df["text"].str.len(), bins=30, color="#1f77b4", edgecolor="black", alpha=0.7)
ax[2].set_title("Text Length Distribution", fontweight="bold"); ax[2].set_xlabel("Characters"); ax[2].set_ylabel("Frequency")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "01_dataset_overview.png"), bbox_inches="tight"); plt.close()

# Fig 2 macro-F1 bars
fig, ax = plt.subplots(figsize=(7, 5.5))
means = [np.mean(f1_maj), np.mean(f1_mb), np.mean(f1_mu)]
stds = [np.std(f1_maj), np.std(f1_mb), np.std(f1_mu)]
bars = ax.bar(["Majority\nBaseline", "mBERT\n(generic)", "MuRIL\n(code-mix)"], means, yerr=stds, capsize=8,
              color=["#888888", "#ff7f0e", "#2ca02c"], edgecolor="black", linewidth=1.2)
ax.set_ylabel("Macro-F1 Score", fontweight="bold"); ax.set_ylim(0, 0.85)
ax.set_title("Model Comparison: Macro-F1 (5-fold CV, mean ± std)", fontweight="bold")
for b, m, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width() / 2, m + s + 0.02, f"{m:.3f}", ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "02_macro_f1_comparison.png"), bbox_inches="tight"); plt.close()

# Fig 3 per-fold
folds = np.arange(1, 6)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(folds, f1_mb, "o-", label="mBERT", color="#ff7f0e", linewidth=2, markersize=9)
ax.plot(folds, f1_mu, "s-", label="MuRIL", color="#2ca02c", linewidth=2, markersize=9)
ax.axhline(np.mean(f1_maj), linestyle="--", color="#888888", label="Majority Baseline")
ax.set_xlabel("Fold", fontweight="bold"); ax.set_ylabel("Macro-F1", fontweight="bold"); ax.set_xticks(folds); ax.legend()
ax.set_title("Per-Fold Macro-F1: MuRIL vs mBERT (paired)", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "03_per_fold_comparison.png"), bbox_inches="tight"); plt.close()

# Fig 4 confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, (name, cm) in zip(ax, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=a, cbar=False,
                xticklabels=["False", "Not-Clearly-False"], yticklabels=["False", "Not-Clearly-False"],
                annot_kws={"fontsize": 13, "fontweight": "bold"})
    a.set_title(f"{name} — Aggregated Confusion Matrix\n(across 5 folds)", fontweight="bold")
    a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "04_confusion_matrices.png"), bbox_inches="tight"); plt.close()

# Fig 5 per-fold advantage
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(folds, diffs, color=["#2ca02c" if d > 0 else "#d62728" for d in diffs], edgecolor="black")
ax.axhline(0, color="black"); ax.axhline(diffs.mean(), linestyle="--", color="blue", label=f"Mean diff = {diffs.mean():.3f}")
ax.set_xlabel("Fold", fontweight="bold"); ax.set_ylabel("Macro-F1 Difference (MuRIL - mBERT)", fontweight="bold")
ax.set_xticks(folds); ax.legend(); ax.set_title("Per-Fold Advantage of MuRIL over mBERT", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "05_muril_advantage_per_fold.png"), bbox_inches="tight"); plt.close()
print("\nfigures written to", FIG)
