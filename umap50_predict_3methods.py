#!/usr/bin/env python3
"""Predict UMAP 50D clusters from input embeddings.
3 clustering methods: KMeans(K=5), GMM(K=5), DBSCAN(eps=0.37,ms=10).
20 random 50/50 hold-out splits, logistic regression."""
import numpy as np
import pickle
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import umap

N_SAMPLES = 1000

# Load input embeddings
with open('embeddings/conversation_embeddings.pkl', 'rb') as f:
    full_conv_embeddings = pickle.load(f)[:N_SAMPLES]
with open('embeddings/first_portion_embeddings.pkl', 'rb') as f:
    first_embeddings = pickle.load(f)[:N_SAMPLES]
with open('embeddings/last_portion_embeddings.pkl', 'rb') as f:
    last_embeddings = pickle.load(f)[:N_SAMPLES]

# Load and UMAP reduce output embeddings
with open('embeddings/fullprompt_1000_output_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)

scaler_out = StandardScaler()

results = []

for temp_label, temp_key in [("temp=0", "t0"), ("temp=0.7", "t07")]:
    emb = emb_data[temp_key]
    emb_scaled = scaler_out.fit_transform(emb)

    print(f"UMAP 50D on {temp_label}...", flush=True)
    reducer = umap.UMAP(n_components=50, random_state=42, n_neighbors=30, min_dist=0.0, metric='cosine')
    X_umap = reducer.fit_transform(emb_scaled)

    # Generate cluster labels from 3 methods
    cluster_methods = {}

    # 1. KMeans K=5
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_methods['KMeans(K=5)'] = km.fit_predict(X_umap)

    # 2. GMM K=5
    gmm = GaussianMixture(n_components=5, random_state=42, n_init=3, max_iter=300)
    gmm.fit(X_umap)
    cluster_methods['GMM(K=5)'] = gmm.predict(X_umap)

    # 3. DBSCAN(eps=0.37, ms=10) — fill noise with KNN
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(X_umap)
    distances, _ = nn.kneighbors(X_umap)
    k_dist = distances[:, -1]
    eps_75 = np.percentile(k_dist, 75)

    db = DBSCAN(eps=eps_75, min_samples=10)
    labels_db = db.fit_predict(X_umap)
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise = (labels_db == -1).sum()

    # Fill noise
    if -1 in labels_db and n_clusters_db >= 2:
        mask = labels_db != -1
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_umap[mask], labels_db[mask])
        labels_db_filled = labels_db.copy()
        labels_db_filled[~mask] = knn.predict(X_umap[~mask])
    else:
        labels_db_filled = labels_db

    unique_db, counts_db = np.unique(labels_db_filled, return_counts=True)
    sizes_str = ", ".join(str(c) for c in sorted(counts_db, reverse=True))
    cluster_methods[f'DBSCAN(eps=p75,ms=10) [{n_clusters_db}c, {n_noise}noise]'] = labels_db_filled

    print(f"\n{'='*70}")
    print(f"  {temp_label} — UMAP 50D")
    print(f"{'='*70}")

    for method_name, labels in cluster_methods.items():
        unique, counts = np.unique(labels, return_counts=True)
        n_k = len(unique)
        majority_pct = counts.max() / counts.sum() * 100
        sizes = ", ".join(str(c) for c in sorted(counts, reverse=True))

        print(f"\n  {method_name} — {n_k} clusters [{sizes}]")
        print(f"  {'Input':30s} {'Kappa mean':>10s} {'std':>6s} {'range':>16s}")
        print(f"  {'-'*30} {'-'*10} {'-'*6} {'-'*16}")

        for input_name, X in [
            ("Full conv (3072D)", full_conv_embeddings),
            ("Full conv (PCA 20D)", full_conv_embeddings),
            ("First 8k (3072D)", first_embeddings),
            ("First 8k (PCA 20D)", first_embeddings),
            ("Last 8k (PCA 20D)", last_embeddings),
            ("First+Last (PCA 20D)", np.hstack([first_embeddings, last_embeddings])),
        ]:
            use_pca = "PCA" in input_name
            kappas = []
            for seed in range(20):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels, test_size=0.5, random_state=seed, stratify=labels
                )
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_train)
                X_te = sc.transform(X_test)
                if use_pca:
                    pca = PCA(n_components=20, random_state=42)
                    X_tr = pca.fit_transform(X_tr)
                    X_te = pca.transform(X_te)
                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
                clf.fit(X_tr, y_train)
                preds = clf.predict(X_te)
                kappas.append(cohen_kappa_score(y_test, preds))

            mean_k = np.mean(kappas)
            std_k = np.std(kappas)
            print(f"  {input_name:30s} {mean_k:10.4f} {std_k:6.4f} [{np.min(kappas):.3f}-{np.max(kappas):.3f}]")
            results.append({
                'temperature': temp_label, 'cluster_method': method_name,
                'n_clusters': n_k, 'input': input_name,
                'kappa_mean': mean_k, 'kappa_std': std_k,
                'majority_pct': majority_pct,
            })

# Summary
print(f"\n{'='*70}")
print("SUMMARY: Best Kappa per (temperature, cluster method)")
print(f"{'='*70}")

import pandas as pd
results_df = pd.DataFrame(results)

# Compact comparison table
print(f"\n  {'Method':45s} {'temp=0':>10s} {'temp=0.7':>10s}")
print(f"  {'-'*45} {'-'*10} {'-'*10}")

for method in results_df['cluster_method'].unique():
    for temp in ["temp=0", "temp=0.7"]:
        sub = results_df[(results_df['cluster_method'] == method) & (results_df['temperature'] == temp)]
        if len(sub) > 0:
            best = sub['kappa_mean'].max()
            best_input = sub.loc[sub['kappa_mean'].idxmax(), 'input']
        else:
            best = 0
    # Print once per method
    t0_sub = results_df[(results_df['cluster_method'] == method) & (results_df['temperature'] == "temp=0")]
    t07_sub = results_df[(results_df['cluster_method'] == method) & (results_df['temperature'] == "temp=0.7")]
    t0_best = t0_sub['kappa_mean'].max() if len(t0_sub) > 0 else 0
    t07_best = t07_sub['kappa_mean'].max() if len(t07_sub) > 0 else 0
    print(f"  {method:45s} {t0_best:10.4f} {t07_best:10.4f}")

results_df.to_csv('embeddings/umap50_3methods_prediction.csv', index=False)
print(f"\nResults saved to embeddings/umap50_3methods_prediction.csv")
