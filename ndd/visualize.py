"""Plots for the README: example heatmaps and the robustness curve."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_examples(images, masks, maps, labels, path, n=4):
    """Rows: input, ground truth, anomaly heatmap overlay. Shows defective samples."""
    idx = np.flatnonzero(labels == 1)[:n]
    fig, axes = plt.subplots(3, len(idx), figsize=(3 * len(idx), 9))
    axes = np.atleast_2d(axes).reshape(3, len(idx))
    for col, i in enumerate(idx):
        axes[0, col].imshow(images[i], cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(masks[i], cmap="gray")
        axes[2, col].imshow(images[i], cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(maps[i], cmap="jet", alpha=0.45)
    for row, title in enumerate(["Input (noisy)", "Ground truth", "Anomaly map"]):
        axes[row, 0].set_ylabel(title, fontsize=12)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_robustness(rows, path, metric="image_auroc"):
    """AUROC vs. noise level, one line per (detector, denoiser) pair."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    combos = sorted({(r["detector"], r["denoise"]) for r in rows})
    for det, den in combos:
        pts = sorted((r["noise_level"], r[metric]) for r in rows if r["detector"] == det and r["denoise"] == den)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=f"{det} + {den}")
    ax.set_xlabel("Noise level")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
