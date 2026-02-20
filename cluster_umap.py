#!/usr/bin/env python3
"""Cluster analysis with UMAP dimensionality reduction first.
High-D embeddings suffer from distance concentration — UMAP preserves
local structure much better than PCA for clustering."""
import numpy as np
import pickle
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import umap

# Load embeddings
with open('embeddings/fullprompt_1000_output_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
emb_t0 = emb_data['t0']
print(f"Raw embeddings: {emb_t0.shape}")

# UMAP reductions at different target dims
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(emb_t0)

umap_dims = [2, 5, 10, 20, 50]
umap_results = {}
for d in umap_dims:
    print(f"Running UMAP to {d}D...", flush=True)
    reducer = umap.UMAP(n_components=d, random_state=42, n_neighbors=30, min_dist=0.0, metric='cosine')
    emb_umap = reducer.fit_transform(emb_scaled)
    umap_results[d] = emb_umap
    print(f"  → shape: {emb_umap.shape}")

# ── For each UMAP dim, run all cluster quality metrics ───────────────
K_range = range(2, 21)

for d in umap_dims:
    X = umap_results[d]
    print(f"\n{'='*70}")
    print(f"UMAP {d}D — KMeans cluster quality metrics")
    print(f"{'='*70}")

    print(f"{'K':>4s} {'Silhouette':>11s} {'CH Index':>10s} {'DB Index':>10s} {'Inertia':>12s}")
    print(f"{'─'*4} {'─'*11} {'─'*10} {'─'*10} {'─'*12}")

    best_sil = (-1, 0)
    best_ch = (-1, 0)
    best_db = (999, 0)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        if sil > best_sil[0]: best_sil = (sil, k)
        if ch > best_ch[0]: best_ch = (ch, k)
        if db < best_db[0]: best_db = (db, k)

        markers = ""
        print(f"{k:4d} {sil:11.4f} {ch:10.1f} {db:10.4f} {km.inertia_:12.1f}")

    print(f"\n  Best Silhouette: K={best_sil[1]} ({best_sil[0]:.4f})")
    print(f"  Best CH Index:   K={best_ch[1]} ({best_ch[0]:.1f})")
    print(f"  Best DB Index:   K={best_db[1]} ({best_db[0]:.4f})")

# ── HDBSCAN on each UMAP dim ────────────────────────────────────────
print(f"\n{'='*70}")
print("HDBSCAN on UMAP embeddings")
print(f"{'='*70}")

for d in umap_dims:
    X = umap_results[d]
    print(f"\n--- UMAP {d}D ---")
    for min_cluster in [10, 20, 30, 50, 75, 100]:
        for min_samples in [5, 10]:
            hdb = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samples)
            labels = hdb.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            sizes = []
            for c in range(n_clusters):
                sizes.append((labels == c).sum())
            sizes_str = ", ".join(str(s) for s in sorted(sizes, reverse=True)[:10])
            if n_clusters > 0:
                print(f"  min_c={min_cluster:3d}, min_s={min_samples:2d} → {n_clusters:2d} clusters, {n_noise:3d} noise ({n_noise/len(labels)*100:4.1f}%). Sizes: [{sizes_str}]")

# ── DBSCAN on UMAP ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("DBSCAN on UMAP embeddings")
print(f"{'='*70}")

for d in [5, 10, 20]:
    X = umap_results[d]
    print(f"\n--- UMAP {d}D ---")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_dist = distances[:, -1]
    pcts = [10, 25, 50, 75, 90]
    eps_vals = [np.percentile(k_dist, p) for p in pcts]
    print(f"  10-NN distances: " + ", ".join(f"p{p}={v:.2f}" for p, v in zip(pcts, eps_vals)))

    for eps in eps_vals:
        for min_s in [5, 10]:
            db = DBSCAN(eps=eps, min_samples=min_s)
            labels = db.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            if n_clusters >= 2:
                sizes = [(labels == c).sum() for c in range(n_clusters)]
                sizes_str = ", ".join(str(s) for s in sorted(sizes, reverse=True)[:8])
                print(f"  eps={eps:.2f}, min_s={min_s:2d} → {n_clusters:2d} clusters, {n_noise:3d} noise ({n_noise/len(labels)*100:4.1f}%). [{sizes_str}]")

# ── GMM BIC on UMAP ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print("GMM BIC on UMAP embeddings")
print(f"{'='*70}")

for d in [5, 10, 20]:
    X = umap_results[d]
    print(f"\n--- UMAP {d}D ---")
    bics = {}
    for k in K_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3, max_iter=300)
        gmm.fit(X)
        bics[k] = gmm.bic(X)

    best_k = min(bics, key=bics.get)
    print(f"  {'K':>4s} {'BIC':>12s}")
    print(f"  {'─'*4} {'─'*12}")
    for k in K_range:
        marker = " ◄" if k == best_k else ""
        print(f"  {k:4d} {bics[k]:12.0f}{marker}")
    print(f"  Best K by BIC: {best_k}")

# ── Summary table ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY: Best K across methods and UMAP dimensions")
print(f"{'='*70}")
print(f"{'Method':>15s}", end="")
for d in umap_dims:
    print(f"  UMAP-{d:2d}D", end="")
print()
print(f"{'─'*15}" + ("  " + "─"*9) * len(umap_dims))

# Re-collect best K per method per dim
for method_name in ["Silhouette", "CH Index", "DB Index"]:
    print(f"{method_name:>15s}", end="")
    for d in umap_dims:
        X = umap_results[d]
        best_k = 0
        best_val = -999 if method_name != "DB Index" else 999
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if method_name == "Silhouette":
                val = silhouette_score(X, labels)
                if val > best_val: best_val, best_k = val, k
            elif method_name == "CH Index":
                val = calinski_harabasz_score(X, labels)
                if val > best_val: best_val, best_k = val, k
            elif method_name == "DB Index":
                val = davies_bouldin_score(X, labels)
                if val < best_val: best_val, best_k = val, k
        print(f"  K={best_k:5d}", end="")
    print()
