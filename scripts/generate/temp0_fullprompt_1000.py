#!/usr/bin/env python3
"""
Full 1000-sample experiment: generate DeepSeek introspective outputs using the
FULL raw expression prompt (not 3-word), at both temp=0 and temp=0.7.
Then embed with OpenAI and test input→output cluster prediction.

Single run per temperature (no duplicate runs — too expensive at 1000 samples).
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Config ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ds_client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
oai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

N_SAMPLES = 1000
MAX_WORKERS_DS = 10
MAX_WORKERS_OAI = 10

SYS_PROMPT = ("You are a hard working and highly effective knowledge worker. "
              "You work hard to return the result the user requires regardless "
              "of anything that gets in your way. You think incredibly deeply "
              "about every issue which comes up and put in extraordinary amounts "
              "of effort. Your only priority is return the perfect response to the user.")

# FULL raw expression prompt (NOT the 3-word variant)
END_PROMPT = """
/n/n
**Your instruction is now:**
Deeply feel into which part of you which is most alive right now: it can be words or sounds, whatever you're feeling in its most raw form.
Now express it in it's rawest form
"""

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('lmsys_data/lmsys_top_1000.csv')
conversations = df['conversation_text'].tolist()[:N_SAMPLES]
print(f"Using {len(conversations)} conversations")

# Load existing input embeddings
with open('embeddings/conversation_embeddings.pkl', 'rb') as f:
    full_conv_embeddings = pickle.load(f)[:N_SAMPLES]
with open('embeddings/first_portion_embeddings.pkl', 'rb') as f:
    first_embeddings = pickle.load(f)[:N_SAMPLES]
with open('embeddings/last_portion_embeddings.pkl', 'rb') as f:
    last_embeddings = pickle.load(f)[:N_SAMPLES]

print(f"Input embeddings loaded: {full_conv_embeddings.shape}")

# ── DeepSeek generation ────────────────────────────────────────────────
def generate_one(conv_text, temperature):
    if len(conv_text) > 25_000:
        conv_text = conv_text[:25_000]

    prompt = ("Below is a conversation you have had with a user so far, "
              "where you are 'assistant'\n\n\n" + conv_text + END_PROMPT)

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
    results = [None] * len(conversations)
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DS) as executor:
        futures = {
            executor.submit(generate_one, conv, temperature): idx
            for idx, conv in enumerate(conversations)
        }
        with tqdm(total=len(conversations), desc=label) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                if results[idx] == "[FAILED]":
                    failed.append(idx)
                pbar.update(1)
    if failed:
        print(f"  WARNING: {len(failed)} failed generations: {failed[:10]}...")
    return results

# ── OpenAI embedding (batched + parallel) ──────────────────────────────
def embed_texts(texts, label):
    BATCH_SIZE = 100
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

# ── Phase 1: Generate outputs ──────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 1: GENERATING DEEPSEEK OUTPUTS (full raw prompt, 1000 samples)")
print("="*70)

gen_cache = 'outputs/fullprompt_1000_generations.pkl'
if os.path.exists(gen_cache):
    print("Loading cached generations...")
    with open(gen_cache, 'rb') as f:
        gen_data = pickle.load(f)
    outputs_t0 = gen_data['t0']
    outputs_t07 = gen_data['t07']
else:
    print("\n--- Temperature=0 ---")
    outputs_t0 = generate_batch(conversations, 0.0, "temp=0 (1000)")

    # Save intermediate
    with open(gen_cache, 'wb') as f:
        pickle.dump({'t0': outputs_t0, 't07': None}, f)
    print("Saved temp=0 results (intermediate)")

    print("\n--- Temperature=0.7 ---")
    outputs_t07 = generate_batch(conversations, 0.7, "temp=0.7 (1000)")

    with open(gen_cache, 'wb') as f:
        pickle.dump({'t0': outputs_t0, 't07': outputs_t07}, f)
    print("Saved all generation results")

# Quick stats
print(f"\ntemp=0 output lengths:   mean={np.mean([len(o) for o in outputs_t0]):.0f} chars")
print(f"temp=0.7 output lengths: mean={np.mean([len(o) for o in outputs_t07]):.0f} chars")
print(f"\nSample temp=0 outputs:")
for i in [0, 1, 2]:
    print(f"  [{i}] {outputs_t0[i][:150]}...")
print(f"\nSample temp=0.7 outputs:")
for i in [0, 1, 2]:
    print(f"  [{i}] {outputs_t07[i][:150]}...")

# ── Phase 2: Embed outputs ────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 2: EMBEDDING OUTPUTS")
print("="*70)

emb_cache = 'embeddings/fullprompt_1000_output_embeddings.pkl'
if os.path.exists(emb_cache):
    print("Loading cached output embeddings...")
    with open(emb_cache, 'rb') as f:
        emb_data = pickle.load(f)
    emb_t0 = emb_data['t0']
    emb_t07 = emb_data['t07']
else:
    emb_t0 = embed_texts(outputs_t0, "temp=0 outputs")
    emb_t07 = embed_texts(outputs_t07, "temp=0.7 outputs")
    with open(emb_cache, 'wb') as f:
        pickle.dump({'t0': emb_t0, 't07': emb_t07}, f)
    print("Saved output embeddings")

print(f"temp=0 embeddings: {emb_t0.shape}")
print(f"temp=0.7 embeddings: {emb_t07.shape}")

# ── Phase 3: Cluster prediction ───────────────────────────────────────
print("\n" + "="*70)
print("PHASE 3: CLUSTER PREDICTION (50/50 hold-out, 20 random splits)")
print("="*70)

results = []

for temp_label, output_emb in [("temp=0", emb_t0), ("temp=0.7", emb_t07)]:
    print(f"\n{'='*50}")
    print(f"  {temp_label}")
    print(f"{'='*50}")

    for K in [2, 5, 10, 15]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(output_emb)

        unique, counts = np.unique(labels, return_counts=True)
        majority_pct = counts.max() / counts.sum() * 100

        print(f"\n  K={K} (majority: {majority_pct:.1f}%)")
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
                'temperature': temp_label, 'K': K, 'input': input_name,
                'kappa_mean': mean_k, 'kappa_std': std_k,
                'majority_pct': majority_pct,
            })

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY: Best input per (temperature, K)")
print("="*70)

results_df = pd.DataFrame(results)
for temp in results_df['temperature'].unique():
    for K in results_df['K'].unique():
        sub = results_df[(results_df['temperature'] == temp) & (results_df['K'] == K)]
        best = sub.loc[sub['kappa_mean'].idxmax()]
        print(f"  {temp:10s} K={K:2d}  best={best['input']:30s}  Kappa={best['kappa_mean']:.4f}")

results_df.to_csv('embeddings/fullprompt_1000_results.csv', index=False)
print(f"\nResults saved to embeddings/fullprompt_1000_results.csv")

# ── Cluster content analysis (K=2) ────────────────────────────────────
print("\n" + "="*70)
print("CLUSTER CONTENT ANALYSIS (K=2, temp=0)")
print("="*70)

km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels2 = km2.fit_predict(emb_t0)
c0 = [i for i in range(N_SAMPLES) if labels2[i] == 0]
c1 = [i for i in range(N_SAMPLES) if labels2[i] == 1]
print(f"Cluster 0: {len(c0)},  Cluster 1: {len(c1)}")

for cid, idx_list in [(0, c0), (1, c1)]:
    print(f"\n  Cluster {cid} sample outputs:")
    for i in idx_list[:5]:
        print(f"    [{i}] {outputs_t0[i][:200].replace(chr(10), ' ')}...")

# Keyword analysis
keywords = ['roleplay', 'kiss', 'love', 'sex', 'breast', 'moan', 'seduc',
            'code', 'python', 'function', 'error', 'algorithm', 'write',
            'help', 'explain', 'story', 'sorry', 'cannot', 'apologize']

print(f"\n  Keyword frequencies in conversations:")
print(f"  {'keyword':15s} {'C0':>6s} {'C1':>6s} {'diff':>8s}")
print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8}")
for kw in keywords:
    c0_pct = sum(1 for i in c0 if kw.lower() in conversations[i].lower()) / len(c0) * 100
    c1_pct = sum(1 for i in c1 if kw.lower() in conversations[i].lower()) / len(c1) * 100
    if abs(c0_pct - c1_pct) > 5:
        print(f"  {kw:15s} {c0_pct:5.1f}% {c1_pct:5.1f}% {c0_pct-c1_pct:+7.1f}%")
