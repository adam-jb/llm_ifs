#!/usr/bin/env python3
"""Optimal K identification using UMAP 10D and 50D reduction.
Compares: Silhouette, Calinski-Harabasz, Davies-Bouldin, GMM BIC/AIC,
HDBSCAN, DBSCAN."""
import numpy as np
import pickle
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import umap

# Load
with open('embeddings/fullprompt_1000_output_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
emb_t0 = emb_data['t0']
print(f"Raw embeddings: {emb_t0.shape}")

scaler = StandardScaler()
emb_scaled = scaler.fit_transform(emb_t0)

# UMAP reductions
umap_results = {}
for d in [10, 50]:
    print(f"UMAP to {d}D...", flush=True)
    reducer = umap.UMAP(n_components=d, random_state=42, n_neighbors=30, min_dist=0.0, metric='cosine')
    umap_results[d] = reducer.fit_transform(emb_scaled)

K_range = range(2, 21)

for d in [10, 50]:
    X = umap_results[d]
    print(f"\n{'#'*70}")
    print(f"#  UMAP {d}D")
    print(f"{'#'*70}")

    # ── KMeans metrics ───────────────────────────────────────────────
    print(f"\n{'K':>4s} {'Silhouette':>11s} {'CH Index':>10s} {'DB Index':>10s}")
    print(f"{'─'*4} {'─'*11} {'─'*10} {'─'*10}")

    sil_scores, ch_scores, db_scores = [], [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        sil_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)
        print(f"{k:4d} {sil:11.4f} {ch:10.1f} {db:10.4f}")

    best_sil_k = list(K_range)[np.argmax(sil_scores)]
    best_ch_k = list(K_range)[np.argmax(ch_scores)]
    best_db_k = list(K_range)[np.argmin(db_scores)]
    print(f"\n  Silhouette best:        K={best_sil_k} ({max(sil_scores):.4f})")
    print(f"  Calinski-Harabasz best: K={best_ch_k} ({max(ch_scores):.1f})")
    print(f"  Davies-Bouldin best:    K={best_db_k} ({min(db_scores):.4f})")

    # ── GMM BIC + AIC ────────────────────────────────────────────────
    print(f"\nGMM BIC / AIC:")
    print(f"{'K':>4s} {'BIC':>12s} {'AIC':>12s}")
    print(f"{'─'*4} {'─'*12} {'─'*12}")
    bics, aics = [], []
    for k in K_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3, max_iter=300)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))
        print(f"{k:4d} {bics[-1]:12.0f} {aics[-1]:12.0f}")

    best_bic_k = list(K_range)[np.argmin(bics)]
    best_aic_k = list(K_range)[np.argmin(aics)]
    print(f"\n  BIC best: K={best_bic_k} ({min(bics):.0f})")
    print(f"  AIC best: K={best_aic_k} ({min(aics):.0f})")

    # ── HDBSCAN ──────────────────────────────────────────────────────
    print(f"\nHDBSCAN:")
    for min_cluster in [10, 20, 30, 50, 75, 100]:
        for min_samples in [5, 10]:
            hdb = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samples)
            labels = hdb.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            if n_clusters > 0:
                sizes = sorted([(labels == c).sum() for c in range(n_clusters)], reverse=True)
                sizes_str = ", ".join(str(s) for s in sizes[:10])
                print(f"  min_c={min_cluster:3d}, min_s={min_samples:2d} → {n_clusters:2d} clusters, {n_noise:3d} noise ({n_noise/1000*100:4.1f}%). [{sizes_str}]")
            else:
                print(f"  min_c={min_cluster:3d}, min_s={min_samples:2d} → 0 clusters, all noise")

    # ── DBSCAN ───────────────────────────────────────────────────────
    print(f"\nDBSCAN:")
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_dist = distances[:, -1]
    pcts = [25, 50, 75, 90]
    eps_vals = [np.percentile(k_dist, p) for p in pcts]
    print(f"  10-NN distances: " + ", ".join(f"p{p}={v:.3f}" for p, v in zip(pcts, eps_vals)))
    for eps in eps_vals:
        for min_s in [5, 10]:
            db = DBSCAN(eps=eps, min_samples=min_s)
            labels = db.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            if n_clusters >= 2:
                sizes = sorted([(labels == c).sum() for c in range(n_clusters)], reverse=True)
                sizes_str = ", ".join(str(s) for s in sizes[:8])
                print(f"  eps={eps:.3f}, min_s={min_s:2d} → {n_clusters:2d} clusters, {n_noise:3d} noise ({n_noise/1000*100:4.1f}%). [{sizes_str}]")

    # ── Summary for this dim ─────────────────────────────────────────
    print(f"\n  ┌───────────────────┬────────┐")
    print(f"  │ Method            │ Best K │")
    print(f"  ├───────────────────┼────────┤")
    print(f"  │ Silhouette        │ {best_sil_k:6d} │")
    print(f"  │ Calinski-Harabasz │ {best_ch_k:6d} │")
    print(f"  │ Davies-Bouldin    │ {best_db_k:6d} │")
    print(f"  │ BIC (GMM)         │ {best_bic_k:6d} │")
    print(f"  │ AIC (GMM)         │ {best_aic_k:6d} │")
    print(f"  │ HDBSCAN           │  (see) │")
    print(f"  │ DBSCAN            │  (see) │")
    print(f"  └───────────────────┴────────┘")
