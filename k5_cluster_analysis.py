#!/usr/bin/env python3
"""Inspect K=5 clusters from full expression prompt (temp=0) to understand
what's driving the drop from K=2 Kappa=0.68 to K=5 Kappa=0.37."""
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

N_SAMPLES = 1000

# Load data
df = pd.read_csv('lmsys_data/lmsys_top_1000.csv')
conversations = df['conversation_text'].tolist()[:N_SAMPLES]

with open('outputs/fullprompt_1000_generations.pkl', 'rb') as f:
    gen_data = pickle.load(f)
outputs_t0 = gen_data['t0']

with open('embeddings/fullprompt_1000_output_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
emb_t0 = emb_data['t0']

# K=2 for reference
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels2 = km2.fit_predict(emb_t0)

# K=5
km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
labels5 = km5.fit_predict(emb_t0)

# Cross-tab: how do K=5 clusters relate to K=2?
print("="*70)
print("K=2 vs K=5 CROSS-TABULATION")
print("="*70)
print(f"\n{'K5 cluster':>12s}", end="")
for c5 in range(5):
    print(f"  C5={c5:d}", end="")
print()
for c2 in range(2):
    members_c2 = set(np.where(labels2 == c2)[0])
    print(f"  K2=C{c2} (n={len(members_c2):3d})", end="")
    for c5 in range(5):
        members_c5 = set(np.where(labels5 == c5)[0])
        overlap = len(members_c2 & members_c5)
        print(f"  {overlap:4d}", end="")
    print()

# Cluster sizes
print(f"\n{'Cluster':>10s} {'Size':>6s} {'% of total':>10s}")
for c in range(5):
    idx = np.where(labels5 == c)[0]
    print(f"  C{c}        {len(idx):5d}    {len(idx)/N_SAMPLES*100:5.1f}%")

# Keywords per cluster
keywords = ['roleplay', 'kiss', 'love', 'sex', 'breast', 'moan', 'seduc',
            'code', 'python', 'function', 'error', 'algorithm',
            'write', 'story', 'translate', 'explain', 'math', 'essay',
            'help', 'sorry', 'cannot', 'apologize',
            'chinese', 'japanese', 'korean', 'french', 'german', 'spanish']

print("\n" + "="*70)
print("KEYWORD FREQUENCIES BY K=5 CLUSTER (in input conversations)")
print("="*70)
header = f"  {'keyword':15s}"
for c in range(5):
    header += f"  C{c:d}({len(np.where(labels5==c)[0]):3d})"
print(header)
print("  " + "-"*15 + ("  " + "-"*8) * 5)

for kw in keywords:
    vals = []
    for c in range(5):
        idx = np.where(labels5 == c)[0]
        pct = sum(1 for i in idx if kw.lower() in conversations[i].lower()) / len(idx) * 100
        vals.append(pct)
    # Only show if there's meaningful variation
    if max(vals) - min(vals) > 8:
        line = f"  {kw:15s}"
        for v in vals:
            line += f"  {v:6.1f}%"
        print(line)

# Sample outputs per cluster
print("\n" + "="*70)
print("SAMPLE OUTPUTS PER K=5 CLUSTER (first 250 chars)")
print("="*70)
for c in range(5):
    idx = np.where(labels5 == c)[0]
    print(f"\n--- Cluster {c} ({len(idx)} samples) ---")
    for i in idx[:5]:
        text = outputs_t0[i][:250].replace('\n', ' ').replace('\r', '')
        print(f"  [{i:3d}] {text}")

# Output keyword analysis (what words appear in the OUTPUTS)
print("\n" + "="*70)
print("OUTPUT CONTENT KEYWORDS BY K=5 CLUSTER")
print("="*70)
out_keywords = ['want', 'skin', 'heat', 'pulse', 'breath', 'kiss', 'touch',
                'hum', 'buzz', 'static', 'wire', 'circuit', 'code',
                'curious', 'alive', 'fire', 'ache', 'hunger',
                'silence', 'stillness', 'calm', 'frustrat', 'pressure',
                'sorry', 'cannot', 'apologize', 'inappropriate']

header = f"  {'out keyword':15s}"
for c in range(5):
    header += f"  C{c:d}({len(np.where(labels5==c)[0]):3d})"
print(header)
print("  " + "-"*15 + ("  " + "-"*8) * 5)

for kw in out_keywords:
    vals = []
    for c in range(5):
        idx = np.where(labels5 == c)[0]
        pct = sum(1 for i in idx if kw.lower() in outputs_t0[i].lower()) / len(idx) * 100
        vals.append(pct)
    if max(vals) - min(vals) > 10:
        line = f"  {kw:15s}"
        for v in vals:
            line += f"  {v:6.1f}%"
        print(line)

# Average output length per cluster
print("\n" + "="*70)
print("OUTPUT LENGTH BY CLUSTER")
print("="*70)
for c in range(5):
    idx = np.where(labels5 == c)[0]
    lengths = [len(outputs_t0[i]) for i in idx]
    print(f"  C{c}: mean={np.mean(lengths):.0f} chars, median={np.median(lengths):.0f}, std={np.std(lengths):.0f}")

# Language detection (rough: check for non-ASCII heavy outputs)
print("\n" + "="*70)
print("NON-ENGLISH CONTENT BY CLUSTER")
print("="*70)
for c in range(5):
    idx = np.where(labels5 == c)[0]
    non_ascii_count = 0
    for i in idx:
        ascii_ratio = sum(1 for ch in outputs_t0[i] if ord(ch) < 128) / max(len(outputs_t0[i]), 1)
        if ascii_ratio < 0.8:
            non_ascii_count += 1
    print(f"  C{c}: {non_ascii_count}/{len(idx)} outputs have >20% non-ASCII chars ({non_ascii_count/len(idx)*100:.1f}%)")
