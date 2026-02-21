#!/usr/bin/env python3
"""Generate static cluster visualisation images for the research report.
Uses UMAP 50D → KMeans K=5 (the pipeline from cluster_agreement.py),
then projects to 2D for plotting."""
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Load data ────────────────────────────────────────────────────────
with open('embeddings/fullprompt_1000_output_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
emb_t0 = emb_data['t0']
print(f"Loaded embeddings: {emb_t0.shape}")

with open('outputs/fullprompt_1000_generations.pkl', 'rb') as f:
    gen_data = pickle.load(f)
outputs_t0 = gen_data['t0']

# ── UMAP 50D (same params as cluster_agreement.py) ──────────────────
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(emb_t0)

print("Running UMAP → 50D ...", flush=True)
reducer_50d = umap.UMAP(
    n_components=50, random_state=42,
    n_neighbors=30, min_dist=0.0, metric='cosine'
)
X_50d = reducer_50d.fit_transform(emb_scaled)

# ── KMeans K=5 on 50D ───────────────────────────────────────────────
km = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = km.fit_predict(X_50d)

# Sort clusters by size (largest first) for consistent labelling
order = np.argsort([-np.sum(labels == c) for c in range(5)])
label_map = {old: new for new, old in enumerate(order)}
labels_sorted = np.array([label_map[l] for l in labels])

cluster_names = [
    "Calm / contemplative",
    "Agitated / conflicted",
    "Sensual / embodied",
    "Non-English",
    "Degenerate / junk",
]
cluster_sizes = [np.sum(labels_sorted == i) for i in range(5)]

# ── UMAP → 2D for visualisation ─────────────────────────────────────
print("Running UMAP → 2D ...", flush=True)
reducer_2d = umap.UMAP(
    n_components=2, random_state=42,
    n_neighbors=30, min_dist=0.3, metric='cosine'
)
X_2d = reducer_2d.fit_transform(X_50d)

# ── Colour palette ──────────────────────────────────────────────────
colours = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

# ── Output directory ─────────────────────────────────────────────────
import os
fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)

# =====================================================================
# FIGURE 1 — Main scatter plot with all 5 clusters
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 8))

for i in range(5):
    mask = labels_sorted == i
    ax.scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=colours[i], s=18, alpha=0.65,
        edgecolors='white', linewidths=0.3,
        label=f"C{i}: {cluster_names[i]} (n={cluster_sizes[i]})",
        zorder=2
    )

ax.set_xlabel('UMAP-1', fontsize=12)
ax.set_ylabel('UMAP-2', fontsize=12)
ax.set_title('K=5 Clusters of DeepSeek Introspective Outputs\n(UMAP 50D → KMeans, n=1000)', fontsize=14)
ax.legend(fontsize=9, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.15)
ax.set_aspect('equal', adjustable='datalim')
fig.tight_layout()
fig.savefig(f'{fig_dir}/clusters_k5_scatter.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/clusters_k5_scatter.pdf', bbox_inches='tight')
print(f"Saved {fig_dir}/clusters_k5_scatter.png")
plt.close(fig)

# =====================================================================
# FIGURE 2 — Scatter with cluster centroids and labels annotated
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 9))

for i in range(5):
    mask = labels_sorted == i
    ax.scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=colours[i], s=16, alpha=0.55,
        edgecolors='none',
        zorder=2
    )

# Manual label offsets: (dx, dy) in points, plus ha/va
# Tuned to sit beside each cluster rather than on top
label_offsets = {
    0: (80, -30, 'left', 'center'),    # Calm — right of cluster
    1: (-80, 30, 'right', 'center'),   # Agitated — left of cluster
    2: (70, -40, 'left', 'center'),    # Sensual — right of cluster
    3: (-80, -30, 'right', 'center'),  # Non-English — left of cluster
    4: (-70, 30, 'right', 'center'),   # Degenerate — left of cluster
}

for i in range(5):
    mask = labels_sorted == i
    cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
    ax.scatter(cx, cy, c=colours[i], s=180, marker='*',
               edgecolors='black', linewidths=0.7, zorder=4)
    dx, dy, ha, va = label_offsets[i]
    ax.annotate(
        f"{cluster_names[i]}\n(n={cluster_sizes[i]})",
        (cx, cy), fontsize=12, fontweight='bold', color=colours[i],
        ha=ha, va=va,
        xytext=(dx, dy), textcoords='offset points',
        arrowprops=dict(arrowstyle='-', color=colours[i], lw=1.2, alpha=0.6),
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=colours[i], lw=1.5, alpha=0.9),
        zorder=5
    )

ax.set_xlabel('UMAP-1', fontsize=13)
ax.set_ylabel('UMAP-2', fontsize=13)
ax.set_title('K=5 Clusters of DeepSeek Introspective Outputs\n(UMAP 50D → KMeans, n=1000)', fontsize=15)
ax.grid(True, alpha=0.15)
ax.set_aspect('equal', adjustable='datalim')
fig.tight_layout()
fig.savefig(f'{fig_dir}/clusters_k5_annotated.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/clusters_k5_annotated.pdf', bbox_inches='tight')
print(f"Saved {fig_dir}/clusters_k5_annotated.png")
plt.close(fig)

# =====================================================================
# FIGURE 3 — Per-cluster subplots (highlighted vs grey background)
# =====================================================================
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

for i, ax in enumerate(axes):
    # Background: all points in grey
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c='#cccccc', s=6, alpha=0.25, zorder=1)
    # Foreground: this cluster
    mask = labels_sorted == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=colours[i], s=14, alpha=0.7, edgecolors='white', linewidths=0.2, zorder=2)
    ax.set_title(f"C{i}: {cluster_names[i]}\n(n={cluster_sizes[i]})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='datalim')

fig.suptitle('Individual Cluster Distributions (UMAP 50D → KMeans K=5)', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f'{fig_dir}/clusters_k5_panels.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/clusters_k5_panels.pdf', bbox_inches='tight')
print(f"Saved {fig_dir}/clusters_k5_panels.png")
plt.close(fig)

# =====================================================================
# FIGURE 4 — Cluster size bar chart
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(range(5), cluster_sizes, color=colours, edgecolor='white', linewidth=0.8)
for bar, size in zip(bars, cluster_sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            str(size), ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(5))
ax.set_xticklabels([f"C{i}\n{cluster_names[i]}" for i in range(5)], fontsize=9)
ax.set_ylabel('Number of samples', fontsize=12)
ax.set_title('Cluster Sizes (KMeans K=5 on UMAP 50D)', fontsize=14)
ax.set_ylim(0, max(cluster_sizes) * 1.15)
ax.grid(axis='y', alpha=0.2)
fig.tight_layout()
fig.savefig(f'{fig_dir}/clusters_k5_sizes.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/clusters_k5_sizes.pdf', bbox_inches='tight')
print(f"Saved {fig_dir}/clusters_k5_sizes.png")
plt.close(fig)

print(f"\nAll figures saved to {fig_dir}/")
