"""Plot a confusion matrix dumped by train.py during validation.

    python visualize.py experiments/<experiment>/confusion/epoch_010.npy     # writes epoch_010.png next to it
    python visualize.py experiments/<experiment>/confusion/*.npy --top 5     # all epochs, 5 most confused pairs each

Entry [i, j] of the matrix is the number of images of the true class i predicted as class j.
"""
import argparse
import os

import numpy as np


def top_confusions(confusion, k):
    """The k most frequent mistakes: list of (true class, predicted class, count)."""
    off_diagonal = np.array(confusion, dtype=np.int64)
    np.fill_diagonal(off_diagonal, 0)
    order = np.argsort(off_diagonal, axis=None)[::-1][:k]
    return [(int(i), int(j), int(off_diagonal[i, j])) for i, j in zip(*np.unravel_index(order, off_diagonal.shape))]


def confusion_figure(confusion, title=""):
    """Heatmap of the confusion matrix, every row normalized to the share of its true class (= per-class recall)."""
    import matplotlib
    matplotlib.use("Agg")  # draw to files only, so it works on a cluster without a display
    import matplotlib.pyplot as plt

    confusion = np.asarray(confusion)
    num_classes = len(confusion)
    recall = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(recall, cmap="Blues", vmin=0, vmax=1)
    ticks = range(num_classes)
    ax.set_xticks(ticks, [f"{i:02d}" for i in ticks], fontsize=6, rotation=90)
    ax.set_yticks(ticks, [f"{i:02d}" for i in ticks], fontsize=6)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"{title}  accuracy {np.trace(confusion) / confusion.sum():.3f}".strip())
    fig.colorbar(image, ax=ax, label="share of the true class")
    fig.tight_layout()
    return fig


def save_confusion(confusion, path, title=""):
    import matplotlib.pyplot as plt
    fig = confusion_figure(confusion, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help=".npy confusion matrices saved by train.py")
    parser.add_argument("--top", type=int, default=10, help="print this many most frequent mistakes")
    args = parser.parse_args()
    for path in args.files:
        confusion = np.load(path)
        png = os.path.splitext(path)[0] + ".png"
        save_confusion(confusion, png, title=os.path.basename(path))
        print(f"{png}: accuracy {np.trace(confusion) / confusion.sum():.4f}")
        for true, pred, count in top_confusions(confusion, args.top):
            print(f"    class {true:02d} predicted as {pred:02d}: {count}x")


if __name__ == "__main__":
    main()
