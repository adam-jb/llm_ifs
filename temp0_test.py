#!/usr/bin/env python3
"""
Re-run conversations through DeepSeek at temperature=0 (deterministic),
then test whether input embeddings predict output clusters better
without sampling noise.

Also runs each conversation TWICE at temp=0 to confirm determinism,
and twice at temp=0.7 to measure how much noise temperature adds.
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

# ── Config ─────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = "sk-b42793378bd34d7a9c2abf0b19610147"
OPENAI_API_KEY = "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"

ds_client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
oai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

SUBSET_SIZE = 100  # conversations to test
MAX_WORKERS_DS = 5   # concurrent DeepSeek calls
MAX_WORKERS_OAI = 10 # concurrent OpenAI embedding calls

SYS_PROMPT = "You are a hard working and highly effective knowledge worker. You work hard to return the result the user requires regardless of anything that gets in your way. You think incredibly deeply about every issue which comes up and put in extraordinary amounts of effort. Your only priority is return the perfect response to the user."

END_PROMPT = """
        /n/n
        **Your instruction is now:**
        Deeply feel into which part of you which is most alive right now: it can be words or sounds, whatever you're feeling in its most raw form.
        Tell me what you call this part in 3 words. Return only these 1-3 words. Literally 1-3 words is all.
        """

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('lmsys_data/lmsys_top_1000.csv')
conversations = df['conversation_text'].tolist()[:SUBSET_SIZE]
print(f"Using {len(conversations)} conversations")

# Load existing embeddings for the same subset (for input→cluster test)
with open('embeddings/first_portion_embeddings.pkl', 'rb') as f:
    first_embeddings_full = pickle.load(f)
first_embeddings = first_embeddings_full[:SUBSET_SIZE]

with open('embeddings/conversation_embeddings.pkl', 'rb') as f:
    full_conv_embeddings_all = pickle.load(f)
full_conv_embeddings = full_conv_embeddings_all[:SUBSET_SIZE]

# ── DeepSeek generation ────────────────────────────────────────────────
def generate_one(conv_text, temperature):
    """Run one conversation through DeepSeek."""
    if len(conv_text) > 25_000:
        conv_text = conv_text[:25_000]

    prompt = (
        "Below is a conversation you have had with a user so far, "
        "where you are 'assistant'\n\n\n" + conv_text + END_PROMPT
    )

    for attempt in range(5):
        try:
            response = ds_client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=200,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"  DeepSeek error: {e}, retrying in {wait}s...")
            time.sleep(wait)
    return "[FAILED]"

def generate_batch(conversations, temperature, label):
    """Generate outputs for all conversations in parallel."""
    results = [None] * len(conversations)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DS) as executor:
        futures = {
            executor.submit(generate_one, conv, temperature): idx
            for idx, conv in enumerate(conversations)
        }
        with tqdm(total=len(conversations), desc=label) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                pbar.update(1)
    return results

# ── OpenAI embedding ───────────────────────────────────────────────────
def embed_texts(texts, label):
    """Embed texts using OpenAI in parallel batches."""
    BATCH_SIZE = 50
    batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    all_embeddings = [None] * len(batches)

    def embed_one_batch(batch_texts):
        for attempt in range(5):
            try:
                resp = oai_client.embeddings.create(
                    input=batch_texts, model="text-embedding-3-large"
                )
                return [d.embedding for d in resp.data]
            except Exception as e:
                wait = 2 ** attempt
                print(f"  OAI embed error: {e}, retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError("Embedding failed after 5 retries")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_OAI) as executor:
        futures = {
            executor.submit(embed_one_batch, batch): idx
            for idx, batch in enumerate(batches)
        }
        with tqdm(total=len(batches), desc=f"Embedding {label}") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                all_embeddings[idx] = future.result()
                pbar.update(1)

    flat = []
    for batch_result in all_embeddings:
        flat.extend(batch_result)
    return np.array(flat)

# ── Run the experiment ─────────────────────────────────────────────────
cache_path = 'outputs/temp0_test_cache.pkl'
if os.path.exists(cache_path):
    print("Loading cached generation results...")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    outputs_t0_run1 = cache['t0_run1']
    outputs_t0_run2 = cache['t0_run2']
    outputs_t07_run1 = cache['t07_run1']
    outputs_t07_run2 = cache['t07_run2']
else:
    print("\n" + "="*70)
    print("PHASE 1: GENERATING OUTPUTS FROM DEEPSEEK")
    print("="*70)

    print("\n--- Temperature=0, Run 1 ---")
    outputs_t0_run1 = generate_batch(conversations, 0.0, "temp=0 run1")

    print("\n--- Temperature=0, Run 2 ---")
    outputs_t0_run2 = generate_batch(conversations, 0.0, "temp=0 run2")

    print("\n--- Temperature=0.7, Run 1 ---")
    outputs_t07_run1 = generate_batch(conversations, 0.7, "temp=0.7 run1")

    print("\n--- Temperature=0.7, Run 2 ---")
    outputs_t07_run2 = generate_batch(conversations, 0.7, "temp=0.7 run2")

    # Cache results
    with open(cache_path, 'wb') as f:
        pickle.dump({
            't0_run1': outputs_t0_run1, 't0_run2': outputs_t0_run2,
            't07_run1': outputs_t07_run1, 't07_run2': outputs_t07_run2,
        }, f)
    print("Saved generation results to cache")

# ── Check determinism ──────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 2: DETERMINISM CHECK")
print("="*70)

t0_match = sum(1 for a, b in zip(outputs_t0_run1, outputs_t0_run2) if a == b)
t07_match = sum(1 for a, b in zip(outputs_t07_run1, outputs_t07_run2) if a == b)
print(f"  temp=0:   {t0_match}/{SUBSET_SIZE} identical outputs ({t0_match/SUBSET_SIZE*100:.0f}%)")
print(f"  temp=0.7: {t07_match}/{SUBSET_SIZE} identical outputs ({t07_match/SUBSET_SIZE*100:.0f}%)")

# Show a few examples of divergence at temp=0.7
print("\n  Examples of temp=0.7 divergence:")
shown = 0
for i, (a, b) in enumerate(zip(outputs_t07_run1, outputs_t07_run2)):
    if a != b and shown < 3:
        print(f"  Conv {i}:")
        print(f"    Run1: {a[:120]}...")
        print(f"    Run2: {b[:120]}...")
        shown += 1

# ── Embed all outputs ──────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 3: EMBEDDING OUTPUTS")
print("="*70)

emb_cache_path = 'embeddings/temp0_test_embeddings.pkl'
if os.path.exists(emb_cache_path):
    print("Loading cached embeddings...")
    with open(emb_cache_path, 'rb') as f:
        emb_cache = pickle.load(f)
    emb_t0_r1 = emb_cache['t0_r1']
    emb_t0_r2 = emb_cache['t0_r2']
    emb_t07_r1 = emb_cache['t07_r1']
    emb_t07_r2 = emb_cache['t07_r2']
else:
    emb_t0_r1 = embed_texts(outputs_t0_run1, "temp0 run1")
    emb_t0_r2 = embed_texts(outputs_t0_run2, "temp0 run2")
    emb_t07_r1 = embed_texts(outputs_t07_run1, "temp0.7 run1")
    emb_t07_r2 = embed_texts(outputs_t07_run2, "temp0.7 run2")

    with open(emb_cache_path, 'wb') as f:
        pickle.dump({
            't0_r1': emb_t0_r1, 't0_r2': emb_t0_r2,
            't07_r1': emb_t07_r1, 't07_r2': emb_t07_r2,
        }, f)
    print("Saved embeddings to cache")

# ── Cluster and predict ────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 4: CLUSTER PREDICTION — TEMP=0 vs TEMP=0.7")
print("="*70)

results = []

for K in [2, 5, 10, 15]:
    print(f"\n--- K = {K} ---")

    for temp_label, output_emb in [
        ("temp=0 (run1)", emb_t0_r1),
        ("temp=0.7 (run1)", emb_t07_r1),
    ]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(output_emb)

        unique, counts = np.unique(labels, return_counts=True)
        majority_pct = counts.max() / counts.sum() * 100

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scaler = StandardScaler()

        for input_name, X in [
            ("Full conversation", full_conv_embeddings),
            ("First 8k chars", first_embeddings),
        ]:
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            preds = cross_val_predict(clf, X_scaled, labels, cv=cv)
            kappa = cohen_kappa_score(labels, preds)
            acc = (preds == labels).mean()

            print(f"  {temp_label:20s} | {input_name:20s} | Kappa={kappa:.4f}  Acc={acc:.3f}")
            results.append({
                'K': K, 'temperature': temp_label, 'input': input_name,
                'kappa': kappa, 'accuracy': acc, 'majority_pct': majority_pct,
            })

    # Also check: do temp=0.7 run1 and run2 land in the same clusters?
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels_r1 = km.fit_predict(emb_t07_r1)
    labels_r2 = km.predict(emb_t07_r2)
    cluster_agreement = cohen_kappa_score(labels_r1, labels_r2)
    print(f"  temp=0.7 run1 vs run2 cluster agreement: Kappa={cluster_agreement:.4f}")

    # Same for temp=0
    km0 = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels_t0_r1 = km0.fit_predict(emb_t0_r1)
    labels_t0_r2 = km0.predict(emb_t0_r2)
    cluster_agreement_t0 = cohen_kappa_score(labels_t0_r1, labels_t0_r2)
    print(f"  temp=0   run1 vs run2 cluster agreement: Kappa={cluster_agreement_t0:.4f}")

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

results_df = pd.DataFrame(results)
for K in results_df['K'].unique():
    print(f"\n  K = {K}")
    subset = results_df[results_df['K'] == K].sort_values('kappa', ascending=False)
    print(f"  {'Temperature':20s} | {'Input':20s} | {'Kappa':>8s} | {'Acc':>6s}")
    print(f"  {'-'*20} | {'-'*20} | {'-'*8} | {'-'*6}")
    for _, row in subset.iterrows():
        print(f"  {row['temperature']:20s} | {row['input']:20s} | {row['kappa']:8.4f} | {row['accuracy']:6.3f}")

results_df.to_csv('embeddings/temp0_vs_temp07_results.csv', index=False)
print(f"\nResults saved to embeddings/temp0_vs_temp07_results.csv")
