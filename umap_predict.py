#!/usr/bin/env python3
"""Redo input→output cluster prediction using UMAP-reduced output embeddings
for clustering (10D and 50D), then test prediction with various input reps."""
import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

N_SAMPLES = 1000

# Load input embeddings
with open('embeddings/conversation_embeddings.pkl', 'rb') as f:
    full_conv_embeddings = pickle.load(f)[:N_SAMPLES]
with open('embeddings/first_portion_embeddings.pkl', 'rb') as f:
    first_embeddings = pickle.load(f)[:N_SAMPLES]
with open('embeddings/last_portion_embeddings.pkl', 'rb') as f:
    last_embeddings = pickle.load(f)[:N_SAMPLES]

# Load output embeddings
with open('embeddings/fullprompt_1000_output_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
emb_t0 = emb_data['t0']
emb_t07 = emb_data['t07']

print(f"Output embeddings: {emb_t0.shape}")
print(f"Input embeddings: {full_conv_embeddings.shape}")

# UMAP reduce output embeddings
scaler_out = StandardScaler()
emb_t0_scaled = scaler_out.fit_transform(emb_t0)

scaler_out07 = StandardScaler()
emb_t07_scaled = scaler_out07.fit_transform(emb_t07)

umap_configs = [10, 50]
umap_embs = {}

for d in umap_configs:
    print(f"\nRunning UMAP to {d}D on temp=0 outputs...", flush=True)
    reducer = umap.UMAP(n_components=d, random_state=42, n_neighbors=30, min_dist=0.0, metric='cosine')
    umap_embs[('t0', d)] = reducer.fit_transform(emb_t0_scaled)

    print(f"Running UMAP to {d}D on temp=0.7 outputs...", flush=True)
    reducer07 = umap.UMAP(n_components=d, random_state=42, n_neighbors=30, min_dist=0.0, metric='cosine')
    umap_embs[('t07', d)] = reducer07.fit_transform(emb_t07_scaled)

# Cluster and predict
results = []

for umap_d in umap_configs:
    for temp_label, temp_key in [("temp=0", "t0"), ("temp=0.7", "t07")]:
        output_emb = umap_embs[(temp_key, umap_d)]

        print(f"\n{'='*60}")
        print(f"  UMAP {umap_d}D — {temp_label}")
        print(f"{'='*60}")

        for K in [2, 5, 10, 15]:
            km = KMeans(n_clusters=K, random_state=42, n_init=10)
            labels = km.fit_predict(output_emb)

            unique, counts = np.unique(labels, return_counts=True)
            majority_pct = counts.max() / counts.sum() * 100
            sizes = ", ".join(str(c) for c in sorted(counts, reverse=True))

            print(f"\n  K={K} (sizes: [{sizes}], majority: {majority_pct:.1f}%)")
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
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_train)
                    X_te = scaler.transform(X_test)
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
                    'umap_dim': umap_d, 'temperature': temp_label, 'K': K,
                    'input': input_name, 'kappa_mean': mean_k, 'kappa_std': std_k,
                    'majority_pct': majority_pct,
                })

# Summary
print("\n" + "="*70)
print("SUMMARY: Best input per (umap_dim, temperature, K)")
print("="*70)

results_df = pd.DataFrame(results)
for umap_d in umap_configs:
    print(f"\n  UMAP {umap_d}D:")
    for temp in results_df['temperature'].unique():
        for K in results_df['K'].unique():
            sub = results_df[
                (results_df['umap_dim'] == umap_d) &
                (results_df['temperature'] == temp) &
                (results_df['K'] == K)
            ]
            best = sub.loc[sub['kappa_mean'].idxmax()]
            print(f"    {temp:10s} K={K:2d}  best={best['input']:30s}  Kappa={best['kappa_mean']:.4f}")

# Comparison table: UMAP vs PCA (raw 3072D clustering)
print("\n" + "="*70)
print("COMPARISON: Best Kappa — PCA clustering vs UMAP clustering")
print("="*70)
# Load PCA results for comparison
pca_df = pd.read_csv('embeddings/fullprompt_1000_results.csv')

print(f"\n  {'':30s} {'PCA clust':>10s} {'UMAP 10D':>10s} {'UMAP 50D':>10s}")
print(f"  {'':30s} {'─'*10} {'─'*10} {'─'*10}")
for temp in ["temp=0", "temp=0.7"]:
    for K in [2, 5, 10, 15]:
        # PCA (raw clustering on 3072D output embeddings)
        pca_sub = pca_df[(pca_df['temperature'] == temp) & (pca_df['K'] == K)]
        pca_best = pca_sub['kappa_mean'].max()

        label = f"{temp} K={K}"
        line = f"  {label:30s} {pca_best:10.4f}"
        for ud in umap_configs:
            usub = results_df[
                (results_df['umap_dim'] == ud) &
                (results_df['temperature'] == temp) &
                (results_df['K'] == K)
            ]
            ubest = usub['kappa_mean'].max()
            line += f" {ubest:10.4f}"
        print(line)

results_df.to_csv('embeddings/umap_cluster_prediction_results.csv', index=False)
print(f"\nResults saved to embeddings/umap_cluster_prediction_results.csv")
