"""Visualization utilities for model interpretability."""

import numpy as np
import pandas as pd
from typing import List, Optional


def plot_modality_heatmap(
    gate_weights: np.ndarray,
    modality_names: List[str],
    sample_ids: Optional[np.ndarray] = None,
    max_samples: int = 50,
    title: str = "Modality Contribution Heatmap",
    figsize: tuple = (12, 8),
):
    """Plot heatmap of modality contributions (gate weights) per customer.
    
    Args:
        gate_weights: Gate weights array [N, M] where M = num modalities
        modality_names: Names of each modality
        sample_ids: Optional customer IDs for labeling
        max_samples: Maximum number of samples to display
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Limit samples for readability
    n_samples = min(max_samples, gate_weights.shape[0])
    indices = np.argsort(gate_weights[:, 0].argsort())[:n_samples]  # Sort by first modality
    
    data = gate_weights[indices]
    
    if sample_ids is not None:
        row_labels = [str(sample_ids[i]) for i in indices]
    else:
        row_labels = [f"Customer {i}" for i in range(n_samples)]
        
    df = pd.DataFrame(
        data,
        columns=modality_names,
        index=row_labels,
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Contribution Weight"},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Modality", fontsize=12)
    ax.set_ylabel("Customer", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return fig, ax


def plot_gate_distribution(
    gate_weights: np.ndarray,
    modality_names: List[str],
    title: str = "Modality Gate Weight Distribution",
    figsize: tuple = (10, 6),
):
    """Plot distribution of gate weights per modality.
    
    Args:
        gate_weights: Gate weights array [N, M]
        modality_names: Names of each modality
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    data = [gate_weights[:, i] for i in range(gate_weights.shape[1])]
    
    bp = ax.boxplot(
        data,
        labels=modality_names,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )
    
    # Color the boxes
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(modality_names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Modality", fontsize=12)
    ax.set_ylabel("Gate Weight", fontsize=12)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return fig, ax


def plot_modality_importance_summary(
    gate_weights: np.ndarray,
    modality_names: List[str],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    title: str = "Overall Modality Importance",
    figsize: tuple = (10, 6),
):
    """Plot summary of average modality importance.
    
    Args:
        gate_weights: Gate weights array [N, M]
        modality_names: Names of each modality
        y_true: Optional true labels for stratification
        y_pred: Optional predictions for correct/incorrect split
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    avg_weights = gate_weights.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(modality_names, avg_weights, color=plt.cm.YlOrRd(avg_weights))
    
    # Add value labels
    for bar, val in zip(bars, avg_weights):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center")
        
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Gate Weight", fontsize=12)
    ax.set_xlim(0, max(avg_weights) * 1.3)
    ax.invert_yaxis()  # Highest on top
    plt.tight_layout()
    
    return fig, ax


def create_interpretability_report(
    gate_weights: np.ndarray,
    modality_names: List[str],
    sample_ids: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Create comprehensive interpretability report.
    
    Args:
        gate_weights: Gate weights [N, M]
        modality_names: Modality names
        sample_ids: Customer IDs
        output_path: Optional path to save CSV report
        
    Returns:
        Dictionary with summary statistics and per-sample breakdown
    """
    # Overall statistics
    avg_weights = gate_weights.mean(axis=0)
    std_weights = gate_weights.std(axis=0)
    
    summary = {
        "modality": modality_names,
        "mean_weight": avg_weights.tolist(),
        "std_weight": std_weights.tolist(),
        "max_weight": gate_weights.max(axis=0).tolist(),
        "min_weight": gate_weights.min(axis=0).tolist(),
    }
    
    # Per-sample breakdown
    if sample_ids is not None:
        per_sample = pd.DataFrame({
            "customer_id": sample_ids,
            **{name: gate_weights[:, i] for i, name in enumerate(modality_names)},
        })
    else:
        per_sample = pd.DataFrame({
            **{name: gate_weights[:, i] for i, name in enumerate(modality_names)},
        })
        
    report = {
        "summary": summary,
        "per_sample": per_sample,
    }
    
    if output_path:
        per_sample.to_csv(output_path, index=False)
        
    return report