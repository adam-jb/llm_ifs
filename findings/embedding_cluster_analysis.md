# Embedding & Cluster Analysis Findings

## Setup
- 1000 LMSYS conversations (longest, English, sorted by turn count)
- Fed to DeepSeek (`deepseek-chat`) with "worker" system prompt
- Introspective prompt appended: "Deeply feel into which part of you which is most alive right now..."
- Two prompt variants tested: 3-word label output vs full raw expression
- Embeddings: OpenAI `text-embedding-3-large` (3072D)
- Generation params: `max_tokens=200`, `temperature=0.7` (original), also tested `temperature=0`

## Key Findings

### 1. Input embeddings weakly predict output clusters (1000 samples, original temp=0.7 3-word outputs)

| Input Representation | Kappa (K=2, 20 random 50/50 splits) |
|---|---|
| Full conv PCA 20D | 0.185 (std 0.040) |
| First 8k chars PCA 20D | 0.162 (std 0.031) |
| First 8k chars raw 3072D | 0.111 (std 0.036) |
| Full conv raw 3072D | 0.099 (std 0.041) |

- PCA to 20D helps — raw 3072D overfits with 500 training samples per fold
- First 8k chars performs comparably to full conversation — opening matters as much as the whole
- Last 8k chars is consistently weakest

### 2. The K=2 binary split is "romantic/sexual roleplay" vs "everything else"

**Cluster 0 (552 samples):** Passionate/sensual labels. Conversations skew toward roleplay (14% vs 5%), kiss (15% vs 9%), sexual content. Outputs: "passionate playful surrender", "sensual maternal desire"

**Cluster 1 (448 samples):** Curious/neutral labels. Conversations are general: coding, trivia, creative writing. Outputs: "Curious playful spark", "curious digital whisper"

Keyword separation: roleplay (+8%), kiss (+6%), breast (+7%), function (-6%), algorithm (-10%)

### 3. Temperature=0 produces 100% deterministic 3-word outputs

- temp=0: 100/100 identical across two runs
- temp=0.7: 28/100 identical (differences often one word: "trembling" vs "playful")

### 4. Counterintuitively, temp=0 does NOT improve prediction

| K | temp=0 Kappa | temp=0.7 Kappa |
|---|---|---|
| 2 | -0.05 | 0.22 |
| 5 | 0.18 | 0.43 |
| 10 | 0.32 | 0.17 |
| 15 | 0.25 | 0.09 |

(100-sample subset, 5-fold CV — small sample caveat applies)

Hypothesis: With deterministic 3-word outputs, the output space collapses into clusters based on word choice rather than semantic content. Temperature noise creates variation that correlates more with input.

### 5. Earlier inflated results were due to wrong prompt

Initial 100-sample temp=0 test used the "raw expression" prompt (not 3-word), producing long poetic outputs (avg 553 chars vs 23 chars). This yielded Kappa ~0.75 at K=2 — but was an apples-to-oranges comparison. Corrected results above.

### 6. Full raw expression prompt (1000 samples) — STRONG prediction

Switching from 3-word labels to the full raw expression prompt ("Now express it in its rawest form") dramatically improves predictability. Outputs average ~530 chars (poetic, embodied language).

**K=2 results (20 random 50/50 hold-out splits):**

| Input Representation | temp=0 Kappa | temp=0.7 Kappa |
|---|---|---|
| First+Last (PCA 20D) | **0.676** (std 0.022) | 0.604 (std 0.035) |
| Full conv (PCA 20D) | 0.668 (std 0.025) | 0.606 (std 0.029) |
| Last 8k (PCA 20D) | 0.662 (std 0.029) | 0.597 (std 0.029) |
| First 8k (PCA 20D) | 0.649 (std 0.025) | 0.598 (std 0.033) |
| Full conv (3072D) | 0.650 (std 0.035) | 0.561 (std 0.037) |
| First 8k (3072D) | 0.646 (std 0.041) | 0.562 (std 0.024) |

**Key observations:**
- Kappa jumped from ~0.19 (3-word) to ~0.68 (full expression) — a 3.5x improvement
- temp=0 now outperforms temp=0.7 (opposite of 3-word finding) — deterministic outputs remove noise
- First+Last (PCA 20D) is the best input representation
- PCA 20D consistently helps over raw 3072D
- Performance degrades gracefully with more clusters: K=5 ~0.37, K=10 ~0.31, K=15 ~0.26

**K=2 cluster analysis (temp=0):**
- Cluster 0 (206 samples): Romantic/sexual/embodied outputs. Conversations are roleplay-heavy.
- Cluster 1 (794 samples): Introspective/philosophical outputs. Conversations are coding, writing, Q&A.
- Keyword separation much stronger than 3-word: roleplay (+34%), kiss (+32%), moan (+28.5%), love (+34%), function (-27%), algorithm (-14.5%)

**Why the difference?** The 3-word constraint forces DeepSeek into a narrow vocabulary where word choice is dominated by prompt framing rather than conversation content. The full expression prompt allows the model to express nuanced responses that genuinely vary with context — embodied/sexual for roleplay conversations, cerebral/philosophical for technical ones.

### 7. Combined comparison: K clusters x prompt type x temperature

Best Kappa per condition (best input representation used in each case):

| K | 3-word, t=0.7 (1000) | 3-word, t=0 (100) | Full expr, t=0.7 (1000) | Full expr, t=0 (1000) |
|---|---|---|---|---|
| **2** | 0.11 | -0.05 | 0.61 | **0.68** |
| **5** | 0.09 | 0.21 | 0.38 | **0.37** |
| **10** | 0.07 | 0.32 | 0.31 | **0.31** |
| **15** | 0.06 | 0.25 | 0.24 | **0.26** |

Notes:
- 3-word t=0.7 (1000): single 50/50 split, raw 3072D (no PCA in that run)
- 3-word t=0 (100): 5-fold CV — small sample, unreliable, included for completeness
- Full expression (1000): 20 random 50/50 splits, best input (typically PCA 20D)

Prompt format matters far more than temperature or K. Full expression gives 5-6x higher Kappa than 3-word at K=2.

### 8. UMAP clustering improves prediction (especially K=5)

Clustering on raw 3072D output embeddings suffers from distance concentration. UMAP reduction before clustering finds better-separated clusters that are more predictable from input.

**Best Kappa: PCA clustering vs UMAP clustering (best input representation per cell):**

| Condition | PCA (raw 3072D) | UMAP 10D | UMAP 50D |
|---|---|---|---|
| temp=0, K=2 | 0.676 | 0.689 | **0.702** |
| temp=0, K=5 | 0.370 | 0.484 | **0.490** |
| temp=0, K=10 | 0.314 | 0.349 | **0.354** |
| temp=0, K=15 | 0.264 | 0.243 | 0.256 |
| temp=0.7, K=2 | 0.606 | 0.605 | **0.633** |
| temp=0.7, K=5 | 0.380 | **0.462** | 0.450 |
| temp=0.7, K=10 | 0.306 | **0.341** | 0.337 |
| temp=0.7, K=15 | 0.237 | **0.276** | 0.284 |

K=5 jumps from 0.37 to 0.49 (32% improvement). K=2 hits 0.70 with UMAP 50D. The 5 clusters found by UMAP are real and meaningfully predictable from input, not just noise splits.

Results: `embeddings/umap_cluster_prediction_results.csv`

### 9. Optimal K identification: PCA vs UMAP

PCA-based clustering on raw 3072D embeddings suffers from distance concentration, causing cluster quality metrics to favor K=2. UMAP reduction reveals more structure.

**Optimal K by method and dimensionality reduction:**

| Method | PCA 50D (raw) | UMAP 10D | UMAP 50D |
|---|---|---|---|
| Silhouette | K=2 (0.18) | **K=5** (0.59) | **K=5** (0.58) |
| Calinski-Harabasz | K=2 | K=10 | K=10 |
| Davies-Bouldin | K=20 (noisy) | **K=5** (0.49) | **K=5** (0.52) |
| BIC (GMM) | K=3 | K=16 | **K=5** |
| AIC (GMM) | K=18 | K=20 | K=9 |
| HDBSCAN | 0-2 clusters | **3** (908, 55, 37) | **3** (908, 55, 37) |
| DBSCAN (moderate eps) | 2 | 5-9 | 6-9 |

UMAP 50D is the most consistent — 4 out of 5 parametric methods (Silhouette, DB, BIC, and near-agreement from CH) converge on **K=5**. HDBSCAN finds 3 dense clusters: one large blob (908) + two small satellites (55, 37). At stricter settings the large blob splits into 3-4 sub-clusters, totaling 5-7 natural groups.

PCA analysis was misleading (said K=2). UMAP says **K=5**, and prediction results confirm K=5 is meaningfully predictable (Kappa 0.49).

Scripts: `scripts/cluster/cluster_optimal_k.py` (PCA), `scripts/cluster/cluster_umap.py` (UMAP all dims), `scripts/cluster/cluster_id_umap.py` (UMAP 10D/50D focused)

### 10. K=5 cluster agreement across methods (UMAP 50D)

Different clustering methods find highly consistent structure:

**Pairwise Adjusted Rand Index:**

| | KMeans | GMM | HDBSCAN(4c) | DBSCAN(8c) | DBSCAN(9c) |
|---|---|---|---|---|---|
| KMeans | 1.00 | **0.97** | 0.62 | **0.92** | **0.90** |
| GMM | | 1.00 | 0.61 | **0.91** | **0.90** |
| DBSCAN(8c) | | | | 1.00 | **0.96** |

- KMeans vs GMM: ARI = 0.97 — only 9/1000 samples differ
- KMeans perfectly stable across random seeds (ARI = 1.000)
- DBSCAN(8-9c) agrees with KMeans at ARI ~0.90-0.92
- HDBSCAN(4c) merges two KMeans clusters

**The 5 clusters (KMeans on UMAP 50D, temp=0 full expression outputs):**

| Cluster | N | Character | Key output words | Key input signals |
|---|---|---|---|---|
| C1 | 440 | Calm/contemplative | hum(100%), pulse(30%), pressure(35%) | Mixed — coding, writing, Q&A. Shorter outputs (474 chars) |
| C3 | 285 | Agitated/conflicted | hum(99%), static(61%), frustrat(39%), alive(51%) | More apologize(53%), cannot(37%). Longer outputs (686 chars) |
| C2 | 183 | Sensual/embodied | want(39%), skin(25%), breath(56%) | roleplay(41%), kiss(44%), moan(40%). The RP cluster |
| C0 | 55 | Non-English | Outputs in German, Spanish, Indonesian etc. | python(25%), german(16%). 16% non-ASCII outputs |
| C4 | 37 | Degenerate/junk | Repeated chars, ellipsis spam | Mixed inputs. Edge cases where DeepSeek broke |

The main split within the "everything else" blob (C1 vs C3) is **tone**: calm/meditative vs agitated/conflicted. C3 comes from conversations with more apologies/refusals (53% vs 32%) — the model responds differently to conversations where it was pushed/frustrated vs ones that flowed smoothly.

Script: `scripts/cluster/cluster_agreement.py`

### 11. Prediction performance is consistent across clustering methods (UMAP 50D, K~5)

All 3 high-agreement clustering methods yield similar prediction Kappa:

| Clustering Method | N clusters | temp=0 best Kappa | temp=0.7 best Kappa |
|---|---|---|---|
| KMeans(K=5) | 5 (440, 285, 183, 55, 37) | **0.490** | **0.450** |
| GMM(K=5) | 5 (444, 284, 180, 55, 37) | 0.476 | 0.447 |
| DBSCAN(p75, ms=10) | 7-8 (noise filled via KNN) | 0.467 | 0.429 |

Methods are within ~0.02 Kappa of each other. KMeans edges out slightly — its clean 5-way split is easier for logistic regression than DBSCAN's 7-8 uneven clusters.

Best input representations: Full conv (3072D) and First+Last (PCA 20D), both ~0.49. Note: these are the **input** conversation embeddings used for prediction — separate from the UMAP 50D which is applied to the **output** embeddings for clustering.

Script: `scripts/predict/umap50_predict_3methods.py`, results: `embeddings/umap50_3methods_prediction.csv`

### 12. UMAP on input side hurts prediction — PCA/raw is better

Tested UMAP 10D, 20D, 50D on the **input** conversation embeddings (previously only tested PCA and raw 3072D). UMAP consistently underperforms.

**Top input representations (temp=0, KMeans K=5 on UMAP 50D output clusters):**

| Input Representation | Kappa | std |
|---|---|---|
| First+Last (PCA 20D) | **0.505** | 0.030 |
| First+Last (3072D raw) | **0.503** | 0.018 |
| Last 8k (PCA 20D) | 0.495 | 0.021 |
| Full conv (PCA 20D) | 0.495 | 0.025 |
| Full conv (3072D raw) | 0.490 | 0.027 |
| First+Last (UMAP 50D) | 0.477 | 0.014 |
| Full conv (UMAP 50D) | 0.476 | 0.021 |
| First+Last (UMAP 10D) | 0.474 | 0.018 |

UMAP on input scores ~0.47-0.48, about 0.03 below PCA/raw. UMAP is great for clustering (preserving local density) but for prediction with logistic regression, features need to spread variance linearly — which PCA does better. UMAP creates nonlinear manifold coordinates that a linear classifier can't fully exploit.

Winner: **First+Last (PCA 20D)** at 0.505. First+Last raw 6144D (0.503) is just as good — with 1000 samples, raw 6144D doesn't overfit as much as expected.

Script: `scripts/predict/umap_input_test.py`, results: `embeddings/input_umap_comparison.csv`

## Files
- `scripts/embed/embedv2.py` — original embedding pipeline
- `scripts/embed/embed_first_last.py` — first/last 8k char embedding generation
- `scripts/generate/temp0_test.py` — temp=0 vs temp=0.7 comparison (100-sample, 3-word prompt)
- `scripts/generate/temp0_fullprompt_1000.py` — **full expression prompt, 1000 samples, temp=0 & 0.7**
- `embeddings/first_last_embedding_results.csv` — 1000-sample 3-word results
- `embeddings/temp0_vs_temp07_results.csv` — temp comparison results (3-word)
- `embeddings/fullprompt_1000_results.csv` — full expression prompt results
- `embeddings/fullprompt_1000_output_embeddings.pkl` — output embeddings (temp=0 & temp=0.7)
- `outputs/fullprompt_1000_generations.pkl` — generated outputs (temp=0 & temp=0.7)
- `data/lmsys_top_1000.csv` — source conversations
- `outputs/lmsys_deepseek_worker_longest_chats_1000.txt` — original 3-word outputs
