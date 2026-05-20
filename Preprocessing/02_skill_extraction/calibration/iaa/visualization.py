"""Confusion matrix and error distribution plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .normalization import normalize_category

LABEL_CLASSES = ["required_skill", "preferred_skill"]


def plot_label_confusion(
    matched_pairs: list[tuple], output_path: str | Path
) -> None:
    """2×2 confusion matrix for label agreement (llm vs human_a)."""
    labels_a = [p[0].get("label", "") for p in matched_pairs]
    labels_b = [p[1].get("label", "") for p in matched_pairs]

    present = sorted(set(labels_a + labels_b))
    cm = confusion_matrix(labels_a, labels_b, labels=present)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present,
        yticklabels=present,
        ax=ax,
    )
    ax.set_xlabel("Annotator B (predicted)")
    ax.set_ylabel("Annotator A (reference)")
    ax.set_title("Label Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_category_confusion(
    matched_pairs: list[tuple], output_path: str | Path
) -> None:
    """N×N confusion matrix for category agreement (llm vs human_a)."""
    cats_a = [normalize_category(p[0].get("category", "")) for p in matched_pairs]
    cats_b = [normalize_category(p[1].get("category", "")) for p in matched_pairs]

    all_cats = sorted(set(cats_a + cats_b))
    cm = confusion_matrix(cats_a, cats_b, labels=all_cats)

    n = len(all_cats)
    fig_size = max(8, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=all_cats,
        yticklabels=all_cats,
        ax=ax,
    )
    ax.set_xlabel("Annotator B (predicted)")
    ax.set_ylabel("Annotator A (reference)")
    ax.set_title("Category Confusion Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_error_distribution(
    error_cases: list[dict], output_path: str | Path
) -> None:
    """Bar chart of error type frequency."""
    if not error_cases:
        return

    counts = pd.Series([e["error_type"] for e in error_cases]).value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Error type")
    ax.set_ylabel("Count")
    ax.set_title("IAA Error Distribution")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
