#!/usr/bin/env python3
"""Find optimal number of clusters using multiple methods on the
full expression (temp=0) output embeddings."""
import numpy as np
import pickle
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load embeddings
with open('embeddings/fullprompt_1000_output_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
emb_t0 = emb_data['t0']
print(f"Embeddings shape: {emb_t0.shape}")

# PCA reduce for methods that struggle with high-D
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(emb_t0)
pca50 = PCA(n_components=50, random_state=42)
emb_pca50 = pca50.fit_transform(emb_scaled)
print(f"PCA 50D variance explained: {pca50.explained_variance_ratio_.sum():.3f}")

pca20 = PCA(n_components=20, random_state=42)
emb_pca20 = pca20.fit_transform(emb_scaled)
print(f"PCA 20D variance explained: {pca20.explained_variance_ratio_.sum():.3f}")

K_range = range(2, 21)

# ── Method 1: Elbow (inertia) ──────────────────────────────────────
print("\n" + "="*70)
print("METHOD 1: ELBOW (KMeans inertia)")
print("="*70)
inertias = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(emb_pca50)
    inertias.append(km.inertia_)

# Compute rate of change to find elbow
deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
delta_ratios = [deltas[i] / deltas[i+1] if deltas[i+1] > 0 else 0 for i in range(len(deltas)-1)]

print(f"{'K':>4s} {'Inertia':>12s} {'Δ Inertia':>12s} {'Δ ratio':>8s}")
print(f"{'─'*4} {'─'*12} {'─'*12} {'─'*8}")
for i, k in enumerate(K_range):
    d = f"{deltas[i]:.0f}" if i < len(deltas) else ""
    r = f"{delta_ratios[i-1]:.2f}" if 0 < i <= len(delta_ratios) else ""
    print(f"{k:4d} {inertias[i]:12.0f} {d:>12s} {r:>8s}")

# ── Method 2: Silhouette Score ──────────────────────────────────────
print("\n" + "="*70)
print("METHOD 2: SILHOUETTE SCORE (higher = better)")
print("="*70)
sil_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(emb_pca50)
    sil = silhouette_score(emb_pca50, labels)
    sil_scores.append(sil)

print(f"{'K':>4s} {'Silhouette':>12s} {'':>4s}")
print(f"{'─'*4} {'─'*12} {'─'*4}")
best_sil_k = list(K_range)[np.argmax(sil_scores)]
for i, k in enumerate(K_range):
    marker = " ◄" if k == best_sil_k else ""
    print(f"{k:4d} {sil_scores[i]:12.4f}{marker}")
print(f"\nBest K by silhouette: {best_sil_k}")

# ── Method 3: Calinski-Harabasz Index ───────────────────────────────
print("\n" + "="*70)
print("METHOD 3: CALINSKI-HARABASZ INDEX (higher = better)")
print("="*70)
ch_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(emb_pca50)
    ch = calinski_harabasz_score(emb_pca50, labels)
    ch_scores.append(ch)

print(f"{'K':>4s} {'CH Index':>12s}")
print(f"{'─'*4} {'─'*12}")
best_ch_k = list(K_range)[np.argmax(ch_scores)]
for i, k in enumerate(K_range):
    marker = " ◄" if k == best_ch_k else ""
    print(f"{k:4d} {ch_scores[i]:12.1f}{marker}")
print(f"\nBest K by Calinski-Harabasz: {best_ch_k}")

# ── Method 4: Davies-Bouldin Index ──────────────────────────────────
print("\n" + "="*70)
print("METHOD 4: DAVIES-BOULDIN INDEX (lower = better)")
print("="*70)
db_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(emb_pca50)
    db = davies_bouldin_score(emb_pca50, labels)
    db_scores.append(db)

print(f"{'K':>4s} {'DB Index':>12s}")
print(f"{'─'*4} {'─'*12}")
best_db_k = list(K_range)[np.argmin(db_scores)]
for i, k in enumerate(K_range):
    marker = " ◄" if k == best_db_k else ""
    print(f"{k:4d} {db_scores[i]:12.4f}{marker}")
print(f"\nBest K by Davies-Bouldin: {best_db_k}")

# ── Method 5: BIC (Gaussian Mixture) ────────────────────────────────
print("\n" + "="*70)
print("METHOD 5: BIC / AIC (Gaussian Mixture Models, PCA 20D)")
print("="*70)
bics = []
aics = []
for k in K_range:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=3, max_iter=200)
    gmm.fit(emb_pca20)
    bics.append(gmm.bic(emb_pca20))
    aics.append(gmm.aic(emb_pca20))

print(f"{'K':>4s} {'BIC':>14s} {'AIC':>14s}")
print(f"{'─'*4} {'─'*14} {'─'*14}")
best_bic_k = list(K_range)[np.argmin(bics)]
best_aic_k = list(K_range)[np.argmin(aics)]
for i, k in enumerate(K_range):
    bic_m = " ◄" if k == best_bic_k else ""
    aic_m = " ◄" if k == best_aic_k else ""
    print(f"{k:4d} {bics[i]:14.0f}{bic_m:3s} {aics[i]:14.0f}{aic_m}")
print(f"\nBest K by BIC: {best_bic_k}")
print(f"Best K by AIC: {best_aic_k}")

# ── Method 6: DBSCAN (automatic) ────────────────────────────────────
print("\n" + "="*70)
print("METHOD 6: DBSCAN (density-based, auto K, PCA 50D)")
print("="*70)
# Try different eps values
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=10)
nn.fit(emb_pca50)
distances, _ = nn.kneighbors(emb_pca50)
k_dist = np.sort(distances[:, -1])
print(f"10-NN distance stats: mean={k_dist.mean():.2f}, median={np.median(k_dist):.2f}, p25={np.percentile(k_dist,25):.2f}, p75={np.percentile(k_dist,75):.2f}")

for eps in [np.percentile(k_dist, 25), np.percentile(k_dist, 50), np.percentile(k_dist, 75), np.percentile(k_dist, 90)]:
    for min_samples in [5, 10, 20]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(emb_pca50)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"  eps={eps:.2f}, min_samples={min_samples:2d} → {n_clusters:2d} clusters, {n_noise:3d} noise points ({n_noise/len(labels)*100:.1f}%)")

# ── Method 7: HDBSCAN ───────────────────────────────────────────────
print("\n" + "="*70)
print("METHOD 7: HDBSCAN (hierarchical density, PCA 50D)")
print("="*70)
for min_cluster in [20, 50, 100, 150]:
    hdb = HDBSCAN(min_cluster_size=min_cluster, min_samples=5)
    labels = hdb.fit_predict(emb_pca50)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    sizes = []
    for c in range(n_clusters):
        sizes.append((labels == c).sum())
    sizes_str = ", ".join(str(s) for s in sorted(sizes, reverse=True)[:8])
    print(f"  min_cluster={min_cluster:3d} → {n_clusters:2d} clusters, {n_noise:3d} noise ({n_noise/len(labels)*100:.1f}%). Sizes: [{sizes_str}]")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Silhouette:        K = {best_sil_k}")
print(f"  Calinski-Harabasz: K = {best_ch_k}")
print(f"  Davies-Bouldin:    K = {best_db_k}")
print(f"  BIC (GMM):         K = {best_bic_k}")
print(f"  AIC (GMM):         K = {best_aic_k}")
print(f"  DBSCAN:            see above (varies with params)")
print(f"  HDBSCAN:           see above (varies with min_cluster_size)")
