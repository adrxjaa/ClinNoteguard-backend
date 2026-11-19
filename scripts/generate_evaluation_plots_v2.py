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
    configs = [a["config"] for a in abls]
    f1_scores = [a["macro_f1"] for a in abls]
    accuracies = [a["accuracy"] for a in abls]

    x = np.arange(len(configs))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    rects1 = ax.bar(x - width / 2, f1_scores, width, label="Macro-F1", color="#1f77b4", edgecolor="#0b3c61", alpha=0.9)
    ax.bar(x + width / 2, accuracies, width, label="Accuracy", color="#2ca02c", edgecolor="#145214", alpha=0.9)

    ax.set_ylabel("Performance Score")
    ax.set_title("System Architecture Ablation Study v2 (Synthea + Gemini)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", weight="semibold")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)

    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

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

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    r1 = ax.bar(x - width / 2, original_f1, width, label="Original Encounter Dates", color="#2b5c8f", edgecolor="#142c47")
    r2 = ax.bar(x + width / 2, shuffled_f1, width, label="Shuffled Encounter Dates", color="#e76f51", edgecolor="#94331a")

    ax.set_ylabel("F1 Score")
    ax.set_title("Date-Shuffle Sensitivity Test v2\n(Impact of Shuffling Synthea Encounter Dates)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, weight="semibold")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    drop_hist = original_f1[0] - shuffled_f1[0]
    ax.annotate(
        f"Drop: -{drop_hist:.4f} F1",
        xy=(0 + width / 2, shuffled_f1[0]),
        xytext=(0 + 0.32, min(0.95, shuffled_f1[0] + 0.22)),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", color="#d62728", fontsize=9.5,
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
    cal = data["calibration"]
    ub = cal["uncalibrated_bins"]
    cb = cal["calibrated_bins"]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)
    ax.plot(ub["conf_centers"], ub["empirical_acc"], "o-",
            label=f"Uncalibrated (ECE = {cal['ece_uncalibrated']:.4f})", color="#e76f51", linewidth=2)
    ax.plot(cb["conf_centers"], cb["empirical_acc"], "s-",
            label=f"Temp Scaled T={cal['temperature']:.2f} (ECE = {cal['ece_calibrated']:.4f})",
            color="#2a9d8f", linewidth=2.2)

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy (Fraction of Positives)")
    ax.set_title("Confidence Calibration Reliability Diagram v2\n(Synthea + Gemini Held-Out Test Set)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)

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
    configs = [a["config"].split()[0] for a in abls]
    f1_scores = [a["macro_f1"] for a in abls]
    colors = ["#888888", "#2b5c8f", "#e76f51", "#457b9d", "#2a9d8f", "#888888", "#e76f51"]
    bars = axs[0, 1].bar(configs, f1_scores, color=colors[:len(configs)], edgecolor="black", alpha=0.85)
    axs[0, 1].set_ylabel("Macro-F1 Score")
    axs[0, 1].set_ylim(0, 1.1)
    axs[0, 1].set_title("(B) Architectural Ablation Study (Macro-F1)")
    axs[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars:
        h = b.get_height()
        axs[0, 1].annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                           textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    # Panel C: Date-Shuffle Drop
    ds = data["date_shuffle_sensitivity"]
    labels = ["HISTORY_SUPPORTED", "STALE"]
    orig = [ds["original_history_f1"], ds["original_stale_f1"]]
    shuff = [ds["shuffled_history_f1"], ds["shuffled_stale_f1"]]
    x = np.arange(len(labels))
    w = 0.35
    axs[1, 0].bar(x - w / 2, orig, w, label="Original Dates", color="#2b5c8f")
    axs[1, 0].bar(x + w / 2, shuff, w, label="Shuffled Dates", color="#e76f51")
    axs[1, 0].set_ylabel("F1 Score")
    axs[1, 0].set_ylim(0, 1.05)
    axs[1, 0].set_title("(C) Date-Shuffle Sensitivity Test")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels)
    axs[1, 0].legend()
    axs[1, 0].grid(axis="y", linestyle="--", alpha=0.4)

    # Panel D: Calibration Curve
    cal = data["calibration"]
    ub = cal["uncalibrated_bins"]
    cb = cal["calibrated_bins"]
    axs[1, 1].plot([0, 1], [0, 1], "k--", label="Ideal Calibration", alpha=0.7)
    axs[1, 1].plot(ub["conf_centers"], ub["empirical_acc"], "o-",
                   label=f"Uncalibrated (ECE = {cal['ece_uncalibrated']:.4f})", color="#e76f51")
    axs[1, 1].plot(cb["conf_centers"], cb["empirical_acc"], "s-",
                   label=f"Temp-Scaled (ECE = {cal['ece_calibrated']:.4f})", color="#2a9d8f")
    axs[1, 1].set_xlabel("Confidence")
    axs[1, 1].set_ylabel("Empirical Accuracy")
    axs[1, 1].set_title("(D) Confidence Calibration Reliability Curve")
    axs[1, 1].legend()
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
    plot_candidate_pool_sweep(ablation_data, PLOTS_DIR / "candidate_pool_sweep_v2.png")
    plot_evaluation_dashboard(ablation_data, PLOTS_DIR / "evaluation_dashboard_v2.png")
    print("\n[+] All v2 evaluation plots generated successfully in results/plots_v2/!")
