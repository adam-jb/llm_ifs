#!/usr/bin/env python3
"""Compare K=5 cluster assignments across different methods on UMAP 50D.
Methods that found ~5 clusters: KMeans, Silhouette-optimal KMeans,
Davies-Bouldin-optimal KMeans, GMM BIC, HDBSCAN (with subclusters),
DBSCAN at various eps."""
import numpy as np
import pickle
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import umap

# Load and UMAP reduce
with open('embeddings/fullprompt_1000_output_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
emb_t0 = emb_data['t0']

scaler = StandardScaler()
emb_scaled = scaler.fit_transform(emb_t0)

print("Running UMAP to 50D...", flush=True)
reducer = umap.UMAP(n_components=50, random_state=42, n_neighbors=30, min_dist=0.0, metric='cosine')
X = reducer.fit_transform(emb_scaled)

# Also load outputs for content analysis
with open('outputs/fullprompt_1000_generations.pkl', 'rb') as f:
    gen_data = pickle.load(f)
outputs_t0 = gen_data['t0']

import pandas as pd
df = pd.read_csv('lmsys_data/lmsys_top_1000.csv')
conversations = df['conversation_text'].tolist()[:1000]

# ── Generate cluster assignments from each method ────────────────────
methods = {}

# 1. KMeans K=5
km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
methods['KMeans(K=5)'] = km5.fit_predict(X)

# 2. GMM K=5
gmm5 = GaussianMixture(n_components=5, random_state=42, n_init=3, max_iter=300)
gmm5.fit(X)
methods['GMM(K=5)'] = gmm5.predict(X)

# 3. KMeans with different random seeds to check stability
for seed in [0, 1, 2, 3]:
    km = KMeans(n_clusters=5, random_state=seed, n_init=10)
    methods[f'KMeans(K=5,seed={seed})'] = km.fit_predict(X)

# 4. HDBSCAN at settings that give ~5 clusters
hdb1 = HDBSCAN(min_cluster_size=50, min_samples=5)
labels_hdb1 = hdb1.fit_predict(X)
# Map noise to nearest cluster
if -1 in labels_hdb1:
    from sklearn.neighbors import KNeighborsClassifier
    mask = labels_hdb1 != -1
    if mask.sum() > 0 and len(set(labels_hdb1[mask])) >= 2:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X[mask], labels_hdb1[mask])
        labels_hdb1_filled = labels_hdb1.copy()
        labels_hdb1_filled[~mask] = knn.predict(X[~mask])
        n_c = len(set(labels_hdb1_filled))
        methods[f'HDBSCAN(mc=50,ms=5) [{n_c}c]'] = labels_hdb1_filled

hdb2 = HDBSCAN(min_cluster_size=30, min_samples=5)
labels_hdb2 = hdb2.fit_predict(X)
if -1 in labels_hdb2:
    mask = labels_hdb2 != -1
    if mask.sum() > 0 and len(set(labels_hdb2[mask])) >= 2:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X[mask], labels_hdb2[mask])
        labels_hdb2_filled = labels_hdb2.copy()
        labels_hdb2_filled[~mask] = knn.predict(X[~mask])
        n_c = len(set(labels_hdb2_filled))
        methods[f'HDBSCAN(mc=30,ms=5) [{n_c}c]'] = labels_hdb2_filled

hdb3 = HDBSCAN(min_cluster_size=75, min_samples=5)
labels_hdb3 = hdb3.fit_predict(X)
if -1 in labels_hdb3:
    mask = labels_hdb3 != -1
    if mask.sum() > 0 and len(set(labels_hdb3[mask])) >= 2:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X[mask], labels_hdb3[mask])
        labels_hdb3_filled = labels_hdb3.copy()
        labels_hdb3_filled[~mask] = knn.predict(X[~mask])
        n_c = len(set(labels_hdb3_filled))
        methods[f'HDBSCAN(mc=75,ms=5) [{n_c}c]'] = labels_hdb3_filled

# 5. DBSCAN at eps that gives ~5 clusters
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=10)
nn.fit(X)
distances, _ = nn.kneighbors(X)
k_dist = distances[:, -1]

for eps in [np.percentile(k_dist, 75), np.percentile(k_dist, 90)]:
    for min_s in [5, 10]:
        db = DBSCAN(eps=eps, min_samples=min_s)
        labels_db = db.fit_predict(X)
        n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        if 3 <= n_clusters <= 9:
            # Fill noise
            if -1 in labels_db:
                mask = labels_db != -1
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X[mask], labels_db[mask])
                labels_db_filled = labels_db.copy()
                labels_db_filled[~mask] = knn.predict(X[~mask])
            else:
                labels_db_filled = labels_db
            methods[f'DBSCAN(eps={eps:.2f},ms={min_s}) [{n_clusters}c]'] = labels_db_filled

# ── Pairwise agreement ───────────────────────────────────────────────
print(f"\n{'='*70}")
print("CLUSTER ASSIGNMENTS OVERVIEW")
print(f"{'='*70}")
for name, labels in methods.items():
    unique, counts = np.unique(labels, return_counts=True)
    sizes = ", ".join(str(c) for c in sorted(counts, reverse=True))
    print(f"  {name:40s} → {len(unique):2d} clusters [{sizes}]")

print(f"\n{'='*70}")
print("PAIRWISE AGREEMENT (Adjusted Rand Index / Normalized Mutual Info)")
print(f"{'='*70}")

# Focus on main methods (not all KMeans seeds)
main_methods = [k for k in methods if 'seed=' not in k]

print(f"\n{'':42s}", end="")
for j, name_j in enumerate(main_methods):
    short = name_j[:8]
    print(f" {short:>8s}", end="")
print()

for i, name_i in enumerate(main_methods):
    print(f"  {name_i:40s}", end="")
    for j, name_j in enumerate(main_methods):
        if j <= i:
            if j == i:
                print(f"     ---", end="")
            else:
                print(f"        ", end="")
        else:
            ari = adjusted_rand_score(methods[name_i], methods[name_j])
            print(f" {ari:8.3f}", end="")
    print()

print(f"\n  Full ARI matrix (main methods):")
print(f"  {'':40s} ", end="")
for j, name_j in enumerate(main_methods):
    print(f" {j:>5d}", end="")
print()
for i, name_i in enumerate(main_methods):
    print(f"  {i}: {name_i:37s}", end="")
    for j, name_j in enumerate(main_methods):
        ari = adjusted_rand_score(methods[name_i], methods[name_j])
        print(f" {ari:5.3f}", end="")
    print()

# KMeans stability across seeds
print(f"\n{'='*70}")
print("KMEANS STABILITY (K=5, different random seeds)")
print(f"{'='*70}")
seed_methods = [k for k in methods if 'seed=' in k]
seed_methods.append('KMeans(K=5)')
for a, b in combinations(seed_methods, 2):
    ari = adjusted_rand_score(methods[a], methods[b])
    nmi = normalized_mutual_info_score(methods[a], methods[b])
    print(f"  {a:25s} vs {b:25s}  ARI={ari:.4f}  NMI={nmi:.4f}")

# ── Cross-tabulation: KMeans vs GMM ──────────────────────────────────
print(f"\n{'='*70}")
print("CROSS-TAB: KMeans(K=5) vs GMM(K=5)")
print(f"{'='*70}")
km_labels = methods['KMeans(K=5)']
gmm_labels = methods['GMM(K=5)']

# Sort clusters by size for readability
km_order = np.argsort([-np.sum(km_labels == c) for c in range(5)])
gmm_order = np.argsort([-np.sum(gmm_labels == c) for c in range(5)])

print(f"\n  {'':10s}", end="")
for gc in gmm_order:
    print(f"  GMM-{gc}", end="")
print(f"  {'Total':>6s}")

for kc in km_order:
    km_mask = km_labels == kc
    print(f"  KM-{kc} ({km_mask.sum():3d})", end="")
    for gc in gmm_order:
        overlap = np.sum(km_mask & (gmm_labels == gc))
        print(f"  {overlap:5d}", end="")
    print(f"  {km_mask.sum():6d}")

print(f"  {'Total':10s}", end="")
for gc in gmm_order:
    print(f"  {np.sum(gmm_labels == gc):5d}", end="")
print(f"  {len(km_labels):6d}")

# ── Content analysis of KMeans K=5 clusters ──────────────────────────
print(f"\n{'='*70}")
print("CONTENT ANALYSIS: KMeans K=5 on UMAP 50D")
print(f"{'='*70}")

keywords = ['roleplay', 'kiss', 'love', 'sex', 'breast', 'moan',
            'code', 'python', 'function', 'algorithm',
            'write', 'story', 'translate', 'explain', 'math',
            'chinese', 'japanese', 'french', 'german',
            'help', 'sorry', 'cannot', 'apologize']

out_keywords = ['want', 'skin', 'heat', 'pulse', 'breath', 'kiss', 'touch',
                'hum', 'buzz', 'static', 'wire', 'circuit',
                'alive', 'fire', 'ache', 'hunger',
                'silence', 'stillness', 'frustrat', 'pressure',
                'sorry', 'cannot', 'apologize']

for kc in km_order:
    idx = np.where(km_labels == kc)[0]
    print(f"\n--- KMeans Cluster {kc} ({len(idx)} samples) ---")

    # Sample outputs
    print(f"  Sample outputs:")
    for i in idx[:3]:
        text = outputs_t0[i][:200].replace('\n', ' ')
        print(f"    [{i:3d}] {text}")

    # Avg output length
    lengths = [len(outputs_t0[i]) for i in idx]
    print(f"  Output length: mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # Input keywords
    print(f"  Input keywords (>15%):", end="")
    for kw in keywords:
        pct = sum(1 for i in idx if kw.lower() in conversations[i].lower()) / len(idx) * 100
        if pct > 15:
            print(f" {kw}({pct:.0f}%)", end="")
    print()

    # Output keywords (>20%)
    print(f"  Output keywords (>20%):", end="")
    for kw in out_keywords:
        pct = sum(1 for i in idx if kw.lower() in outputs_t0[i].lower()) / len(idx) * 100
        if pct > 20:
            print(f" {kw}({pct:.0f}%)", end="")
    print()

    # Non-English
    non_ascii = sum(1 for i in idx if sum(1 for ch in outputs_t0[i] if ord(ch) < 128) / max(len(outputs_t0[i]), 1) < 0.8)
    if non_ascii > 0:
        print(f"  Non-English outputs: {non_ascii} ({non_ascii/len(idx)*100:.1f}%)")
