"""EDA report: generates 5 exploratory charts for transaction anomaly data.

Saves PNG figures to reports/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def load_data() -> pd.DataFrame:
    """Load clean.parquet for EDA.

    Returns:
        Cleaned transactions DataFrame.
    """
    return pd.read_parquet(PROCESSED_DIR / "clean.parquet")


def plot_amount_distribution(df: pd.DataFrame) -> None:
    """Chart 1: Distribution of transaction amounts by anomaly label.

    Args:
        df: Cleaned transactions DataFrame.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Transaction Amount Distribution by Anomaly Label", fontsize=14, fontweight="bold")

    for ax, label, color in zip(axes, [0, 1], ["steelblue", "tomato"]):
        subset = df[df["is_anomaly"] == label]["amount"]
        ax.hist(subset, bins=10, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"{'Normal' if label == 0 else 'Anomaly'} Transactions")
        ax.set_xlabel("Amount ($)")
        ax.set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_amount_distribution.png", dpi=150)
    plt.close(fig)
    print("[EDA] Saved 01_amount_distribution.png")


def plot_anomaly_rate_by_category(df: pd.DataFrame) -> None:
    """Chart 2: Anomaly rate by transaction category.

    Args:
        df: Cleaned transactions DataFrame.
    """
    rates = df.groupby("category")["is_anomaly"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tomato" if v > 0 else "steelblue" for v in rates.values]
    rates.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Anomaly Rate by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Anomaly Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_anomaly_rate_by_category.png", dpi=150)
    plt.close(fig)
    print("[EDA] Saved 02_anomaly_rate_by_category.png")


def plot_amount_boxplot(df: pd.DataFrame) -> None:
    """Chart 3: Box plot of amounts split by anomaly.

    Args:
        df: Cleaned transactions DataFrame.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    df["label"] = df["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
    sns.boxplot(data=df, x="label", y="amount", palette={"Normal": "steelblue", "Anomaly": "tomato"}, ax=ax)
    ax.set_title("Transaction Amount: Normal vs Anomaly", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Amount ($)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_amount_boxplot.png", dpi=150)
    plt.close(fig)
    print("[EDA] Saved 03_amount_boxplot.png")


def plot_merchant_anomaly_rate(df: pd.DataFrame) -> None:
    """Chart 4: Anomaly rate per merchant.

    Args:
        df: Cleaned transactions DataFrame.
    """
    rates = df.groupby("merchant_id")["is_anomaly"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tomato" if v > 0 else "steelblue" for v in rates.values]
    rates.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Anomaly Rate by Merchant", fontsize=14, fontweight="bold")
    ax.set_xlabel("Merchant ID")
    ax.set_ylabel("Anomaly Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_merchant_anomaly_rate.png", dpi=150)
    plt.close(fig)
    print("[EDA] Saved 04_merchant_anomaly_rate.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Chart 5: Correlation matrix of numeric features.

    Args:
        df: Cleaned transactions DataFrame.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("[EDA] Saved 05_correlation_heatmap.png")


def run() -> None:
    """Execute all 5 EDA charts."""
    df = load_data()
    plot_amount_distribution(df)
    plot_anomaly_rate_by_category(df)
    plot_amount_boxplot(df)
    plot_merchant_anomaly_rate(df)
    plot_correlation_heatmap(df)
    print(f"[STAGE 2B] EDA complete — 5 charts saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
