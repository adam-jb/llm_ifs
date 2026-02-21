#!/usr/bin/env python3
"""Test UMAP reduction on INPUT embeddings for prediction.
Previously only used PCA or raw 3072D on the input side.
Now try UMAP 50D, 20D, 10D on input embeddings too."""
import numpy as np
import pickle
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

# Load output embeddings, UMAP 50D, cluster with KMeans K=5
with open('embeddings/fullprompt_1000_output_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
emb_t0 = emb_data['t0']

scaler_out = StandardScaler()
emb_t0_scaled = scaler_out.fit_transform(emb_t0)

print("UMAP 50D on outputs...", flush=True)
reducer_out = umap.UMAP(n_components=50, random_state=42, n_neighbors=30, min_dist=0.0, metric='cosine')
out_umap = reducer_out.fit_transform(emb_t0_scaled)

km = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = km.fit_predict(out_umap)
unique, counts = np.unique(labels, return_counts=True)
print(f"Cluster sizes: {sorted(counts, reverse=True)}")

# UMAP reduce input embeddings
print("\nUMAP reducing input embeddings...", flush=True)
input_sets = {
    'Full conv': full_conv_embeddings,
    'First 8k': first_embeddings,
    'Last 8k': last_embeddings,
    'First+Last': np.hstack([first_embeddings, last_embeddings]),
}

# Pre-compute UMAP reductions for each input set
umap_input_cache = {}
for name, X in input_sets.items():
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    for d in [10, 20, 50]:
        print(f"  UMAP {d}D on {name}...", flush=True)
        reducer = umap.UMAP(n_components=d, random_state=42, n_neighbors=30, min_dist=0.1, metric='cosine')
        umap_input_cache[(name, d)] = reducer.fit_transform(X_scaled)

# Run predictions
print(f"\n{'='*70}")
print("INPUT REPRESENTATION COMPARISON (temp=0, KMeans K=5 on UMAP 50D outputs)")
print(f"{'='*70}")

results = []

def test_input(input_name, X):
    kappas = []
    for seed in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.5, random_state=seed, stratify=labels
        )
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_te = sc.transform(X_test)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        kappas.append(cohen_kappa_score(y_test, preds))
    mean_k = np.mean(kappas)
    std_k = np.std(kappas)
    print(f"  {input_name:40s} {mean_k:10.4f} {std_k:6.4f} [{np.min(kappas):.3f}-{np.max(kappas):.3f}]")
    results.append({'input': input_name, 'kappa_mean': mean_k, 'kappa_std': std_k})

print(f"\n  {'Input':40s} {'Kappa mean':>10s} {'std':>6s} {'range':>16s}")
print(f"  {'-'*40} {'-'*10} {'-'*6} {'-'*16}")

# Raw 3072D
for name, X in input_sets.items():
    test_input(f"{name} (3072D raw)", X)

# PCA 20D
for name, X in input_sets.items():
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    pca = PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    test_input(f"{name} (PCA 20D)", X_pca)

# UMAP various dims
for d in [10, 20, 50]:
    for name in input_sets:
        test_input(f"{name} (UMAP {d}D)", umap_input_cache[(name, d)])

# Sort and show top 10
print(f"\n{'='*70}")
print("TOP 10 INPUT REPRESENTATIONS")
print(f"{'='*70}")
import pandas as pd
rdf = pd.DataFrame(results).sort_values('kappa_mean', ascending=False)
print(f"\n  {'Input':40s} {'Kappa':>8s} {'std':>6s}")
print(f"  {'-'*40} {'-'*8} {'-'*6}")
for _, row in rdf.head(10).iterrows():
    print(f"  {row['input']:40s} {row['kappa_mean']:8.4f} {row['kappa_std']:6.4f}")

rdf.to_csv('embeddings/input_umap_comparison.csv', index=False)
print(f"\nSaved to embeddings/input_umap_comparison.csv")
