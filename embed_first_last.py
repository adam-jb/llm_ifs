#!/usr/bin/env python3
"""
Generate OpenAI text-embedding-3-large embeddings for the FIRST and LAST
portions of each conversation, then test whether they predict output clusters
better than full-conversation embeddings.

Uses batch API calls + concurrent threads for speed.
"""
import pandas as pd
import numpy as np
import pickle
import os
import openai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

OPENAI_API_KEY = "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"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('lmsys_data/lmsys_top_1000.csv')
conversations = df['conversation_text'].tolist()
print(f"Loaded {len(conversations)} conversations")

with open('embeddings/output_embeddings.pkl', 'rb') as f:
    output_embeddings = pickle.load(f)
print(f"Output embeddings shape: {output_embeddings.shape}")

with open('embeddings/conversation_embeddings.pkl', 'rb') as f:
    full_conv_embeddings = pickle.load(f)
print(f"Full conversation embeddings shape: {full_conv_embeddings.shape}")

# ── Extract first / last portions ──────────────────────────────────────
FIRST_CHARS = 8000   # ~2000 tokens
LAST_CHARS  = 8000

first_texts = [conv[:FIRST_CHARS] for conv in conversations]
last_texts  = [conv[-LAST_CHARS:] if len(conv) > LAST_CHARS else conv for conv in conversations]

print(f"\nFirst-portion lengths: min={min(len(t) for t in first_texts)}, "
      f"max={max(len(t) for t in first_texts)}, "
      f"mean={np.mean([len(t) for t in first_texts]):.0f}")
print(f"Last-portion lengths:  min={min(len(t) for t in last_texts)}, "
      f"max={max(len(t) for t in last_texts)}, "
      f"mean={np.mean([len(t) for t in last_texts]):.0f}")

# ── Parallel batch embedding ──────────────────────────────────────────
BATCH_SIZE = 100      # texts per API call (OpenAI supports up to 2048)
MAX_WORKERS = 10      # concurrent API calls

def embed_one_batch(batch_texts, model="text-embedding-3-large"):
    """Single API call that embeds a list of texts."""
    for attempt in range(5):
        try:
            resp = client.embeddings.create(input=batch_texts, model=model)
            return [d.embedding for d in resp.data]
        except openai.RateLimitError:
            wait = 2 ** attempt
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except openai.APIError as e:
            wait = 2 ** attempt
            print(f"  API error: {e}, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Failed after 5 retries")

def embed_parallel(texts, label):
    """Embed all texts using batched, concurrent API calls."""
    # Split into batches
    batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    all_embeddings = [None] * len(batches)

    print(f"  {label}: {len(texts)} texts in {len(batches)} batches, {MAX_WORKERS} workers")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(embed_one_batch, batch): idx
            for idx, batch in enumerate(batches)
        }
        with tqdm(total=len(batches), desc=f"  {label}") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                all_embeddings[idx] = future.result()
                pbar.update(1)

    # Flatten
    flat = []
    for batch_result in all_embeddings:
        flat.extend(batch_result)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/sec)")
    return np.array(flat)

# ── Generate embeddings ────────────────────────────────────────────────
print("\n" + "="*70)
print("GENERATING OPENAI EMBEDDINGS FOR FIRST/LAST PORTIONS")
print("="*70)

first_path = 'embeddings/first_portion_embeddings.pkl'
last_path  = 'embeddings/last_portion_embeddings.pkl'

if os.path.exists(first_path):
    print("Loading cached first-portion embeddings...")
    with open(first_path, 'rb') as f:
        first_embeddings = pickle.load(f)
else:
    first_embeddings = embed_parallel(first_texts, "First 8k chars")
    with open(first_path, 'wb') as f:
        pickle.dump(first_embeddings, f)

if os.path.exists(last_path):
    print("Loading cached last-portion embeddings...")
    with open(last_path, 'rb') as f:
        last_embeddings = pickle.load(f)
else:
    last_embeddings = embed_parallel(last_texts, "Last 8k chars")
    with open(last_path, 'wb') as f:
        pickle.dump(last_embeddings, f)

print(f"\nFirst-portion embeddings shape: {first_embeddings.shape}")
print(f"Last-portion embeddings shape:  {last_embeddings.shape}")

# ── Cluster output embeddings ──────────────────────────────────────────
print("\n" + "="*70)
print("CLUSTERING OUTPUT EMBEDDINGS")
print("="*70)

results = []

for K in [2, 5, 10, 15]:
    print(f"\n--- K = {K} ---")
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = km.fit_predict(output_embeddings)

    unique, counts = np.unique(labels, return_counts=True)
    majority_class_pct = counts.max() / counts.sum() * 100
    print(f"  Cluster sizes: {dict(zip(unique, counts))}")
    print(f"  Majority class: {majority_class_pct:.1f}%")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()

    for name, X in [
        ("Full conversation (3072D)", full_conv_embeddings),
        ("First 8k chars (3072D)",    first_embeddings),
        ("Last 8k chars (3072D)",     last_embeddings),
        ("First + Last concat (6144D)", np.hstack([first_embeddings, last_embeddings])),
    ]:
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                                 solver='lbfgs', multi_class='multinomial')
        preds = cross_val_predict(clf, X_scaled, labels, cv=cv)
        kappa = cohen_kappa_score(labels, preds)
        acc = (preds == labels).mean()

        print(f"  {name:40s}  Kappa={kappa:.4f}  Acc={acc:.3f}")
        results.append({
            'K': K, 'input': name, 'kappa': kappa, 'accuracy': acc,
            'majority_pct': majority_class_pct
        })

# ── Summary table ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

results_df = pd.DataFrame(results)
for K in results_df['K'].unique():
    print(f"\n  K = {K}")
    subset = results_df[results_df['K'] == K].sort_values('kappa', ascending=False)
    print(f"  {'Input':45s} {'Kappa':>8s} {'Acc':>6s}")
    print(f"  {'-'*45} {'-'*8} {'-'*6}")
    for _, row in subset.iterrows():
        print(f"  {row['input']:45s} {row['kappa']:8.4f} {row['accuracy']:6.3f}")

results_df.to_csv('embeddings/first_last_embedding_results.csv', index=False)
print(f"\nResults saved to embeddings/first_last_embedding_results.csv")
