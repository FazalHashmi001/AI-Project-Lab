"""
visualizer.py
All Matplotlib charts used in the application.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_score_gauge(score, category, save_path=None):
    """
    Draw a horizontal bar gauge showing resume score (0-10).
    Color: red (<4), orange (4-6), green (>6).
    """
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')

    # Background bar
    ax.barh(0, 10, color='#3a3a5c', height=0.5)

    # Score bar color
    if score < 4:
        bar_color = '#e74c3c'
        label = 'Low Match'
    elif score < 7:
        bar_color = '#f39c12'
        label = 'Moderate Match'
    else:
        bar_color = '#2ecc71'
        label = 'Strong Match'

    ax.barh(0, score, color=bar_color, height=0.5)

    ax.set_xlim(0, 10)
    ax.set_yticks([])
    ax.set_xticks(range(0, 11))
    ax.tick_params(colors='white')
    ax.set_title(f'Resume Score: {score}/10  |  Category: {category}  |  {label}',
                 color='white', fontsize=11, pad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    return fig


def plot_metrics_bar(metrics, save_path=None):
    """
    Bar chart of Accuracy, Precision, Recall, F1.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')

    names = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]
    colors = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71']

    bars = ax.bar(names, values, color=colors, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val:.1f}%', ha='center', color='white', fontsize=10)

    ax.set_ylim(0, 110)
    ax.set_ylabel('Score (%)', color='white')
    ax.set_title('Model Evaluation Metrics', color='white', fontsize=13)
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3a5c')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    return fig


def plot_category_distribution(categories, save_path=None):
    """
    Horizontal bar chart showing resume counts per job category.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')

    labels = list(categories.keys())
    counts = list(categories.values())

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.barh(labels, counts, color=colors)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(cnt), va='center', color='white', fontsize=9)

    ax.set_xlabel('Number of Resumes', color='white')
    ax.set_title('Dataset Category Distribution', color='white', fontsize=12)
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3a5c')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    return fig