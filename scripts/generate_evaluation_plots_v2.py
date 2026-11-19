"""
Generate Publication-Quality Clinical Verification Evaluation Plots (v2 — Synthea + Gemini).
Reads:
  - results/ablations_v2.json
Outputs (in results/plots_v2/):
  - confusion_matrix_v2.png
  - ablation_study_v2.png
  - date_shuffle_sensitivity_v2.png
  - calibration_reliability_v2.png
  - candidate_pool_sweep_v2.png
  - evaluation_dashboard_v2.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOTS_DIR = Path("results/plots_v2")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})

SHORT_CLASSES = ["TRANS_SUPP", "HIST_SUPP", "CLIN_INFR", "CLIN_CONF", "STALE", "HALLUC"]


def plot_confusion_matrix(data: dict, output_path: Path):
    conf_mat = np.array(data["full_model_metrics"]["confusion_matrix"], dtype=float)
    norm_mat = conf_mat / np.maximum(conf_mat.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    cax = ax.imshow(norm_mat, cmap="Blues", vmin=0, vmax=1.0)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Frequency (Recall)", rotation=270, labelpad=15, weight="semibold")

    for i in range(len(SHORT_CLASSES)):
        for j in range(len(SHORT_CLASSES)):
            raw_cnt = int(conf_mat[i, j])
            frac = norm_mat[i, j]
            text_color = "white" if frac > 0.55 else "black"
            ax.text(j, i, f"{raw_cnt}\n({frac:.1%})", ha="center", va="center",
                    color=text_color, fontsize=9.5, fontweight="semibold")

    ax.set_xticks(range(len(SHORT_CLASSES)))
    ax.set_yticks(range(len(SHORT_CLASSES)))
    ax.set_xticklabels(SHORT_CLASSES, rotation=35, ha="right", weight="semibold")
    ax.set_yticklabels(SHORT_CLASSES, weight="semibold")
    ax.set_xlabel("Predicted Verification Label", labelpad=10)
    ax.set_ylabel("True Clinical Label", labelpad=10)
    ax.set_title(
        f"ClinNoteGuard v2 Verifier Test Confusion Matrix\n"
        f"({data['test_patients']} Held-Out Synthea+Gemini Patients, N={data['test_claims']} Claims)"
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[+] Saved: {output_path}")


def plot_ablation_study(data: dict, output_path: Path):
    abls = data["ablations"]
    short_map = {
        "A0 Majority Class": "A0\nMajority",
        "A1 MLP Base": "A1\nMLP Base",
        "A2 MLP Recency": "A2\nMLP Recency",
        "A3 Transf (No GNN)": "A3\nTransf-Only\n(No GNN)",
        "A6 Full (GNN+Time)": "A6 Full\n(GNN+Time)",
        "A7 EHR-Only": "A7\nEHR-Only\n(No Transc.)",
        "A8 No Time Feat": "A8\nNo Time\nFeatures",
    }
    configs = [short_map.get(a["config"], a["config"]) for a in abls]
    f1_scores = [a["macro_f1"] for a in abls]
    accuracies = [a["accuracy"] for a in abls]

    x = np.arange(len(configs))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    rects1 = ax.bar(x - width / 2, f1_scores, width, label="Macro-F1", color="#1f77b4", edgecolor="#0b3c61", alpha=0.92)
    rects2 = ax.bar(x + width / 2, accuracies, width, label="Accuracy", color="#2ca02c", edgecolor="#145214", alpha=0.92)

    ax.set_ylabel("Performance Score")
    ax.set_title("System Architecture Ablation Study v2 (170 Synthea + Gemini Patients)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, weight="semibold", fontsize=9.5)
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)

    # Highlight A6 Full (Best) & A8 No-Time drop
    best_idx = 4
    ax.annotate(
        f"Best Model (A6 GNN+Time)\nF1: {f1_scores[best_idx]:.3f} (+{f1_scores[best_idx]-f1_scores[3]:.3f} vs No-GNN)",
        xy=(best_idx - width / 2, f1_scores[best_idx]),
        xytext=(best_idx - 0.8, 1.07),
        arrowprops=dict(facecolor="#d62728", shrink=0.06, width=1.6, headwidth=6),
        ha="center", fontweight="bold", color="#d62728", fontsize=9.5,
    )
    no_time_idx = 6
    ax.annotate(
        f"Time Removed: -{f1_scores[best_idx]-f1_scores[no_time_idx]:.3f} F1",
        xy=(no_time_idx - width / 2, f1_scores[no_time_idx]),
        xytext=(no_time_idx - 0.15, 1.03),
        arrowprops=dict(facecolor="#e76f51", shrink=0.06, width=1.4, headwidth=5),
        ha="center", fontweight="bold", color="#b83b1d", fontsize=9,
    )

    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="semibold")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[+] Saved: {output_path}")


def plot_date_shuffle_sensitivity(data: dict, output_path: Path):
    ds = data["date_shuffle_sensitivity"]
    labels = ["HISTORY_SUPPORTED", "STALE"]
    original_f1 = [ds["original_history_f1"], ds["original_stale_f1"]]
    shuffled_f1 = [ds["shuffled_history_f1"], ds["shuffled_stale_f1"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=300)
    r1 = ax.bar(x - width / 2, original_f1, width, label="Original Encounter Dates", color="#2b5c8f", edgecolor="#142c47")
    r2 = ax.bar(x + width / 2, shuffled_f1, width, label="Shuffled Encounter Dates", color="#e76f51", edgecolor="#94331a")

    ax.set_ylabel("F1 Score")
    ax.set_title("Date-Shuffle Sensitivity Test v2\n(Impact of Shuffling Synthea Encounter Dates)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, weight="semibold")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    for idx_b, name in enumerate(labels):
        drop = original_f1[idx_b] - shuffled_f1[idx_b]
        ax.annotate(
            f"Drop: -{drop:.3f} F1",
            xy=(idx_b + width / 2, shuffled_f1[idx_b]),
            xytext=(idx_b + width / 2 + 0.02, shuffled_f1[idx_b] + 0.14),
            arrowprops=dict(facecolor="#d62728", shrink=0.06, width=1.5, headwidth=6),
            ha="center", fontweight="bold", color="#d62728", fontsize=9,
        )

    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.annotate(f"{h:.4f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="semibold")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[+] Saved: {output_path}")


def plot_calibration_reliability(data: dict, output_path: Path):
    """
    Thesis-grade Reliability Diagram with:
      - Top panel: Full [0, 1] reliability bars & curve (M=10 uniform ECE bins) + 95% CIs
      - Bottom panel: Histogram of prediction counts per bin (log/linear with exact n labels)
    """
    cal = data["calibration"]
    m_bins = cal.get("num_bins", 10)
    ub = cal["uncalibrated_bins"]
    cb = cal["calibrated_bins"]
    ece_drop_pct = (cal["ece_uncalibrated"] - cal["ece_calibrated"]) / max(cal["ece_uncalibrated"], 1e-9) * 100.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 7.5), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    centers = np.array(cb["bin_centers"])
    width = 0.08

    # Plot empirical accuracy bars for populated bins (Temp-Scaled) + 95% CI
    pop_idx = [i for i, v in enumerate(cb["empirical_acc"]) if v is not None]
    pop_x = [centers[i] for i in pop_idx]
    pop_y = [cb["empirical_acc"][i] for i in pop_idx]
    pop_ci = [cb["ci95"][i] for i in pop_idx]
    pop_n = [cb["counts"][i] for i in pop_idx]

    ax1.bar(pop_x, pop_y, width=width, color="#2a9d8f", alpha=0.65, edgecolor="#14524a",
            yerr=pop_ci, capsize=4, label=f"Temp-Scaled T={cal['temperature']:.2f} (ECE={cal['ece_calibrated']:.4f}, M={m_bins} bins)")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration (Ideal)")

    # Overlay uncalibrated populated points
    u_idx = [i for i, v in enumerate(ub["conf_centers"]) if v is not None]
    ax1.plot([ub["conf_centers"][i] for i in u_idx], [ub["empirical_acc"][i] for i in u_idx],
             "o-", color="#e76f51", linewidth=2, markersize=6.5,
             label=f"Uncalibrated (ECE={cal['ece_uncalibrated']:.4f})")

    ax1.annotate(
        f"M = {m_bins} Uniform Bins\nECE: {cal['ece_uncalibrated']:.4f} → {cal['ece_calibrated']:.4f} (-{ece_drop_pct:.1f}%)",
        xy=(0.04, 0.74), xycoords="axes fraction",
        fontweight="bold", color="#1b6b61", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eefaf8", edgecolor="#2a9d8f", alpha=0.95),
    )

    ax1.set_ylabel("Empirical Accuracy (±95% CI)")
    ax1.set_title(f"Confidence Calibration Reliability & Sample Distribution (Full [0–1] Range, M={m_bins} Bins)")
    ax1.set_xlim(0.0, 1.02)
    ax1.set_ylim(0.0, 1.08)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8.8)

    # Bottom panel: Histogram of prediction counts per bin
    counts = cb["counts"]
    bars_h = ax2.bar(centers, counts, width=width, color="#457b9d", edgecolor="#1d3557", alpha=0.88)
    ax2.set_xlabel("Predicted Confidence Bin [0.0 – 1.0]")
    ax2.set_ylabel("Sample Count (N)")
    ax2.set_yscale("log")
    ax2.set_ylim(0.8, max(counts) * 2.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for b, cnt in zip(bars_h, counts):
        if cnt > 0:
            ax2.annotate(f"n={cnt}", xy=(b.get_x() + b.get_width() / 2, cnt),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                         fontsize=8, fontweight="semibold")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[+] Saved: {output_path}")


def plot_retrieval_metrics(data: dict, output_path: Path):
    """Plots Top-K Accuracy (Acc@K) and Mean Reciprocal Rank (MRR) on held-out test claims."""
    ret = data["retrieval_metrics"]
    ks = [1, 3, 5, 10, 15]
    accs = [ret["acc_at_k"][str(k)] for k in ks]
    mrr = ret["mrr"]

    fig, ax = plt.subplots(figsize=(7.8, 5.5), dpi=300)
    ax.plot(ks, accs, "o-", color="#1f77b4", linewidth=2.4, markersize=8, label="Top-K Retrieval Accuracy (Acc@K)")
    ax.axhline(mrr, color="#e76f51", linestyle="--", linewidth=2, label=f"Mean Reciprocal Rank (MRR = {mrr:.4f})")

    for k, a in zip(ks, accs):
        ax.annotate(f"{a:.3f}", (k, a), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, fontweight="semibold")

    ax.set_xlabel("Retrieval Cutoff (Top-K)")
    ax.set_ylabel("Score")
    ax.set_title(f"Information Retrieval Performance (Acc@K & MRR)\n(N={ret['evaluated_claims_with_gold']} Test Claims with Gold Evidence)")
    ax.set_xticks(ks)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[+] Saved: {output_path}")


def plot_per_class_f1(data: dict, output_path: Path):
    """Plots Precision, Recall, and F1 across all 6 verification classes."""
    per_cls = data["full_model_metrics"]["per_class"]
    classes = data["classes"]
    short_names = ["TRANS\nSUPP", "HIST\nSUPP", "CLIN\nINFR", "CLIN\nCONF", "STALE", "HALLUC"]
    p_vals = [per_cls[c]["precision"] for c in classes]
    r_vals = [per_cls[c]["recall"] for c in classes]
    f_vals = [per_cls[c]["f1"] for c in classes]
    supps = [per_cls[c]["support"] for c in classes]

    x = np.arange(len(classes))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)
    ax.bar(x - w, p_vals, w, label="Precision", color="#457b9d", edgecolor="#1d3557")
    ax.bar(x, r_vals, w, label="Recall", color="#2a9d8f", edgecolor="#14524a")
    b_f1 = ax.bar(x + w, f_vals, w, label="F1 Score", color="#e76f51", edgecolor="#94331a")

    for b, f in zip(b_f1, f_vals):
        ax.annotate(f"{f:.3f}", xy=(b.get_x() + b.get_width() / 2, f),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="semibold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(short_names, supps)], fontweight="semibold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.7, 1.08)
    ax.set_title("Per-Class Verification Performance (Precision, Recall, F1 — Test Set N=575)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[+] Saved: {output_path}")


def plot_candidate_pool_sweep(data: dict, output_path: Path):
    ks = [r["k"] for r in data["k_sweep"]]
    macro_f1 = [r["macro_f1"] for r in data["k_sweep"]]
    test_acc = [r["accuracy"] for r in data["k_sweep"]]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    ax.plot(ks, macro_f1, "o-", label="Macro-F1", color="#1f77b4", linewidth=2.2, markersize=7)
    ax.plot(ks, test_acc, "s-", label="Test Accuracy", color="#2ca02c", linewidth=2.2, markersize=7)

    ax.set_xlabel("Candidate Pool Size (Top-K)")
    ax.set_ylabel("Metric Value")
    ax.set_title("Performance vs. Candidate Pool Size (Top-K) v2")
    ax.set_xticks(ks)
    ax.set_ylim(0.80, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

    for k, f1 in zip(ks, macro_f1):
        ax.annotate(f"{f1:.3f}", (k, f1), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[+] Saved: {output_path}")


def plot_evaluation_dashboard(data: dict, output_path: Path):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    plt.suptitle(
        f"ClinNoteGuard v2 Verification Evaluation Dashboard\n"
        f"(Synthea + Gemini Longitudinal EHR — {data['test_patients']} Held-Out Patients, N={data['test_claims']} Claims)",
        y=0.98, fontsize=15,
    )

    # Panel A: Confusion Matrix
    conf_mat = np.array(data["full_model_metrics"]["confusion_matrix"], dtype=float)
    norm_mat = conf_mat / np.maximum(conf_mat.sum(axis=1, keepdims=True), 1.0)
    cax = axs[0, 0].imshow(norm_mat, cmap="Blues", vmin=0, vmax=1.0)
    fig.colorbar(cax, ax=axs[0, 0], fraction=0.046, pad=0.04)
    for i in range(len(SHORT_CLASSES)):
        for j in range(len(SHORT_CLASSES)):
            raw_cnt = int(conf_mat[i, j])
            frac = norm_mat[i, j]
            text_color = "white" if frac > 0.55 else "black"
            axs[0, 0].text(j, i, f"{raw_cnt}\n({frac:.0%})", ha="center", va="center", color=text_color, fontsize=8)
    axs[0, 0].set_xticks(range(len(SHORT_CLASSES)))
    axs[0, 0].set_yticks(range(len(SHORT_CLASSES)))
    axs[0, 0].set_xticklabels(SHORT_CLASSES, rotation=35, ha="right", fontsize=9)
    axs[0, 0].set_yticklabels(SHORT_CLASSES, fontsize=9)
    axs[0, 0].set_title("(A) Normalized Test Confusion Matrix")
    axs[0, 0].set_xlabel("Predicted Label")
    axs[0, 0].set_ylabel("True Label")

    # Panel B: Ablation Study
    abls = data["ablations"]
    short_labels = ["A0\nMajority", "A1\nMLP", "A2\nMLP+Rec", "A3\nTransf-Only", "A6\nGNN+Time", "A7\nEHR-Only", "A8\nNo-Time"]
    f1_scores = [a["macro_f1"] for a in abls]
    colors = ["#888888", "#2b5c8f", "#e76f51", "#457b9d", "#2a9d8f", "#6c757d", "#e76f51"]
    bars = axs[0, 1].bar(short_labels[:len(abls)], f1_scores, color=colors[:len(abls)], edgecolor="black", alpha=0.88)
    axs[0, 1].set_ylabel("Macro-F1 Score")
    axs[0, 1].set_ylim(0, 1.18)
    axs[0, 1].set_title("(B) Architectural Ablation Study (Macro-F1)")
    axs[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars:
        h = b.get_height()
        axs[0, 1].annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                           textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="semibold")
    axs[0, 1].annotate(
        f"Best A6: {f1_scores[4]:.3f}\n(+{f1_scores[4]-f1_scores[3]:.3f} vs No-GNN)",
        xy=(4, f1_scores[4]), xytext=(2.8, 1.07),
        arrowprops=dict(facecolor="#d62728", shrink=0.06, width=1.4, headwidth=5),
        ha="center", fontweight="bold", color="#d62728", fontsize=8.5,
    )

    # Panel C: Date-Shuffle Drop
    ds = data["date_shuffle_sensitivity"]
    labels = ["HISTORY_SUPPORTED", "STALE"]
    orig = [ds["original_history_f1"], ds["original_stale_f1"]]
    shuff = [ds["shuffled_history_f1"], ds["shuffled_stale_f1"]]
    x = np.arange(len(labels))
    w = 0.35
    r1c = axs[1, 0].bar(x - w / 2, orig, w, label="Original Dates", color="#2b5c8f")
    r2c = axs[1, 0].bar(x + w / 2, shuff, w, label="Shuffled Dates", color="#e76f51")
    axs[1, 0].set_ylabel("F1 Score")
    axs[1, 0].set_ylim(0, 1.12)
    axs[1, 0].set_title("(C) Date-Shuffle Sensitivity Test")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels, weight="semibold")
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
    for idx_b in range(len(labels)):
        drop = orig[idx_b] - shuff[idx_b]
        axs[1, 0].annotate(
            f"-{drop:.3f} F1",
            xy=(idx_b + w / 2, shuff[idx_b]),
            xytext=(idx_b + w / 2 + 0.02, shuff[idx_b] + 0.14),
            arrowprops=dict(facecolor="#d62728", shrink=0.06, width=1.3, headwidth=5),
            ha="center", fontweight="bold", color="#d62728", fontsize=8.5,
        )
    for r in list(r1c) + list(r2c):
        h = r.get_height()
        axs[1, 0].annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                           textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    # Panel D: Calibration Curve (Full [0-1] M=10 Bins with 95% CI & Sample Counts)
    cal = data["calibration"]
    m_bins = cal.get("num_bins", 10)
    ub = cal["uncalibrated_bins"]
    cb = cal["calibrated_bins"]
    ece_drop_pct = (cal["ece_uncalibrated"] - cal["ece_calibrated"]) / max(cal["ece_uncalibrated"], 1e-9) * 100.0
    pop_idx = [i for i, v in enumerate(cb["empirical_acc"]) if v is not None]
    pop_x = [cb["bin_centers"][i] for i in pop_idx]
    pop_y = [cb["empirical_acc"][i] for i in pop_idx]
    pop_ci = [cb["ci95"][i] for i in pop_idx]
    pop_n = [cb["counts"][i] for i in pop_idx]
    axs[1, 1].bar(pop_x, pop_y, width=0.08, color="#2a9d8f", alpha=0.65, edgecolor="#14524a",
                  yerr=pop_ci, capsize=3, label=f"Temp-Scaled T={cal['temperature']:.2f} (ECE={cal['ece_calibrated']:.4f})")
    axs[1, 1].plot([0, 1], [0, 1], "k--", label="Ideal Calibration", alpha=0.75)
    u_idx = [i for i, v in enumerate(ub["conf_centers"]) if v is not None]
    axs[1, 1].plot([ub["conf_centers"][i] for i in u_idx], [ub["empirical_acc"][i] for i in u_idx],
                   "o-", label=f"Uncalibrated (ECE={cal['ece_uncalibrated']:.4f})", color="#e76f51", linewidth=2)
    for bx, by, bn in zip(pop_x, pop_y, pop_n):
        axs[1, 1].annotate(f"n={bn}", xy=(bx, max(0.05, by - 0.12)), ha="center", fontsize=7.5, fontweight="bold", color="white")
    axs[1, 1].annotate(
        f"M={m_bins} Uniform Bins\nECE -{ece_drop_pct:.1f}%",
        xy=(0.05, 0.65), xycoords="axes fraction",
        fontweight="bold", color="#1b6b61", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eefaf8", edgecolor="#2a9d8f", alpha=0.95),
    )
    axs[1, 1].set_xlim(0.0, 1.02)
    axs[1, 1].set_ylim(0.0, 1.08)
    axs[1, 1].set_xlabel(f"Confidence (Full [0–1] Range, M={m_bins} Bins)")
    axs[1, 1].set_ylabel("Empirical Accuracy (±95% CI)")
    axs[1, 1].set_title("(D) Confidence Calibration Reliability Curve")
    axs[1, 1].legend(loc="upper left", fontsize=8.5)
    axs[1, 1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[+] Saved: {output_path}")


if __name__ == "__main__":
    with open("results/ablations_v2.json", "r", encoding="utf-8") as f:
        ablation_data = json.load(f)

    plot_confusion_matrix(ablation_data, PLOTS_DIR / "confusion_matrix_v2.png")
    plot_ablation_study(ablation_data, PLOTS_DIR / "ablation_study_v2.png")
    plot_date_shuffle_sensitivity(ablation_data, PLOTS_DIR / "date_shuffle_sensitivity_v2.png")
    plot_calibration_reliability(ablation_data, PLOTS_DIR / "calibration_reliability_v2.png")
    plot_retrieval_metrics(ablation_data, PLOTS_DIR / "retrieval_metrics_v2.png")
    plot_per_class_f1(ablation_data, PLOTS_DIR / "per_class_f1_v2.png")
    plot_candidate_pool_sweep(ablation_data, PLOTS_DIR / "candidate_pool_sweep_v2.png")
    plot_evaluation_dashboard(ablation_data, PLOTS_DIR / "evaluation_dashboard_v2.png")
    print("\n[+] All v2 evaluation plots generated successfully in results/plots_v2/!")
