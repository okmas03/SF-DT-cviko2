"""Grafy z results/*.csv: confusion matice a vplyv k na kNN."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RESULTS = Path(__file__).resolve().parent.parent / "results"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

BEST_K = 5  # najlepšie F1 v oboch variantoch (vybrané príznaky)
CM_MODELS = ["GNB (všetky príznaky)", f"kNN k={BEST_K} (selected)"]
CM_TITLES = ["Gaussian Naive Bayes", f"kNN, k = {BEST_K}, vybrané príznaky"]
VARIANT_NAMES = {"A": "Variant A (bez Sunshine/Cloud/Evap.)", "B": "Variant B (všetky stĺpce)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": TEXT,
    "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def plot_confusion_matrices():
    cms = pd.read_csv(RESULTS / "confusion_matrices.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for r, variant in enumerate(["A", "B"]):
        for c, model in enumerate(CM_MODELS):
            ax = axes[r, c]
            row = cms[(cms.variant == variant) & (cms.model == model)].iloc[0]
            cm = np.array([[row.TN, row.FP], [row.FN, row.TP]])
            pct = cm / cm.sum(axis=1, keepdims=True)  # podiel v rámci skutočnej triedy
            ax.imshow(pct, cmap=BLUES, vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    color = "#ffffff" if pct[i, j] > 0.55 else TEXT
                    ax.text(j, i, f"{cm[i, j]:,}".replace(",", " ") + f"\n{pct[i, j]:.1%}",
                            ha="center", va="center", color=color, fontsize=12)
            ax.set_xticks([0, 1], ["No", "Yes"])
            ax.set_yticks([0, 1], ["No", "Yes"])
            ax.set_xlabel("Predikcia")
            ax.set_ylabel("Skutočnosť")
            ax.set_title(f"{CM_TITLES[c]}\n{VARIANT_NAMES[variant]}", fontsize=11, color=TEXT)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xticks(np.arange(-.5, 2), minor=True)
            ax.set_yticks(np.arange(-.5, 2), minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=3)
            ax.tick_params(which="both", length=0)
    fig.suptitle("Confusion matrix – súčet cez 10 foldov (% = podiel v riadku)",
                 fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_k_curve():
    m = pd.read_csv(RESULTS / "metrics.csv")
    knn = m[m.model.str.startswith("kNN")].copy()
    knn["k"] = knn.model.str.extract(r"k=(\d+)").astype(int)
    knn["features"] = knn.model.str.extract(r"\((\w+)\)")
    series = [("A", "all"), ("A", "selected"), ("B", "all"), ("B", "selected")]
    labels = {"all": "všetky príznaky", "selected": "vybrané príznaky"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, metric, title in zip(axes, ["accuracy", "f1"], ["Accuracy", "F1 (trieda Yes)"]):
        for color, (v, fs) in zip(SERIES, series):
            d = knn[(knn.variant == v) & (knn.features == fs)].sort_values("k")
            ax.plot(d.k, d[metric], color=color, linewidth=2, marker="o", markersize=5,
                    label=f"{v}: {labels[fs]}")
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 11, 21, 51, 101], ["1", "5", "11", "21", "51", "101"])
        ax.minorticks_off()
        ax.set_xlabel("k (logaritmická os)")
        ax.set_title(title, loc="left", fontsize=12, color=TEXT)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, frameon=False, fontsize=10, ncol=4, loc="lower center")
    fig.suptitle("kNN: vplyv k (priemer cez 10 foldov)", fontsize=13, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(RESULTS / "knn_k.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_confusion_matrices()
    plot_k_curve()
    print(f"Grafy uložené do {RESULTS}/")