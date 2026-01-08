import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

sns.set_context("paper")  # paper-friendly font sizes
sns.set_style("whitegrid")  # clean style


def plot_auroc(y_true, y_score, save_path="auroc.png", pos_label=1):
    """
    Plots and saves an AUROC (ROC curve) figure.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted scores or probabilities for the positive class.
    save_path : str
        Filepath to save the figure (e.g. "auroc.png").
    pos_label : int
        Label of the positive class (default=1).
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(3.5, 3.5), dpi=300)
    plt.plot(fpr, tpr, lw=2, label=f"AUROC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# def plot_auprc(y_true, y_score, save_path="auprc.png", pos_label=1):
#     """
#     Plots and saves an AUPRC (Precision-Recall curve) figure.
#
#     Parameters
#     ----------
#     y_true : array-like
#         True binary labels.
#     y_score : array-like
#         Predicted scores or probabilities for the positive class.
#     save_path : str
#         Filepath to save the figure (e.g. "auprc.png").
#     pos_label : int
#         Label of the positive class (default=1).
#     """
#     y_true = np.array(y_true)
#     y_score = np.array(y_score)
#
#     precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
#     avg_prec = average_precision_score(y_true, y_score, pos_label=pos_label)
#
#     plt.figure(figsize=(3.5, 3.5), dpi=300)
#     plt.plot(recall, precision, lw=2, label=f"AUPRC = {avg_prec:.3f}")
#
#     # Baseline = proportion of positives
#     baseline = (y_true == pos_label).astype(int).mean()
#     plt.hlines(baseline, 0, 1, colors="k", linestyles="--", lw=1,
#                label=f"Baseline = {baseline:.2f}")
#
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Precision-Recall Curve")
#     plt.legend(loc="lower left", frameon=True)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()


def plot_auprc(y_true, y_score, save_path="auprc.png", pos_label=1):
    """
    Plots and saves an AUPRC (Precision-Recall curve) figure.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted scores or probabilities for the positive class.
    save_path : str
        Filepath to save the figure (e.g. "auprc.png").
    pos_label : int
        Label of the positive class (default=1).
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    avg_prec = average_precision_score(y_true, y_score, pos_label=pos_label)

    # Baseline = prevalence of positives
    baseline = (y_true == pos_label).astype(int).mean()

    plt.figure(figsize=(3.5, 3.5), dpi=300)
    plt.plot(recall, precision, lw=2, label=f"AUPRC = {avg_prec:.3f}")
    plt.hlines(baseline, 0, 1, colors="k", linestyles="--", lw=1,
               label=f"Baseline (prevalence = {baseline:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Pos prevalence = {baseline:.2f})")
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()