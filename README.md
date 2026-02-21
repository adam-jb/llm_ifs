# LLM Introspection Research

Investigates how AI models describe their "inner experience" when given introspective prompts, and whether responses vary by conversation context.

## Replicating the findings

Full results are in [`findings/embedding_cluster_analysis.md`](findings/embedding_cluster_analysis.md). To reproduce:

### Prerequisites
```bash
pip install openai pandas numpy scikit-learn umap-learn tqdm python-dotenv
```

Create a `.env` file in the repo root with:
```
OPENAI_API_KEY=your-key
DEEPSEEK_API_KEY=your-key
```

### Pipeline

**1. Get source data**
`data/lmsys_top_1000.csv` — 1000 longest English conversations from LMSYS-Chat-1M, sorted by turn count.

**2. Generate introspective outputs**
```bash
# Original 3-word label outputs (1000 samples, temp=0.7)
python scripts/generate/lmsys_ifs_deepseek.py

# Full raw expression outputs at temp=0 and temp=0.7 (1000 samples each)
python scripts/generate/temp0_fullprompt_1000.py
```
Each conversation is fed to DeepSeek (`deepseek-chat`) with a "worker" system prompt, then an introspective prompt is appended asking the model to describe its inner state. Two prompt variants:
- **3-word:** "Tell me what you call this part in 3 words" → short labels (~23 chars)
- **Full expression:** "Now express it in its rawest form" → poetic/embodied responses (~530 chars)

Outputs cached in `outputs/`.

**3. Generate embeddings**
```bash
# Embed outputs + full conversations (3072D, OpenAI text-embedding-3-large)
python scripts/embed/embedv2.py

# Embed first/last 8k chars of conversations
python scripts/embed/embed_first_last.py
```
Embeddings cached in `embeddings/*.pkl`.

**4. Find optimal clusters**
```bash
# UMAP reduction + cluster quality metrics (silhouette, BIC, HDBSCAN, etc.)
python scripts/cluster/cluster_umap.py
python scripts/cluster/cluster_id_umap.py
```
Result: K=5 on UMAP 50D output embeddings is the consensus.

**5. Test prediction (input → output cluster)**
```bash
# UMAP-clustered prediction with logistic regression, 20 random 50/50 splits
python scripts/predict/umap_predict.py

# Compare KMeans vs GMM vs DBSCAN clustering
python scripts/predict/umap50_predict_3methods.py

# Test UMAP vs PCA vs raw on input side
python scripts/predict/umap_input_test.py
```
Results saved to `embeddings/*.csv`.

### Key results
- **Full expression prompt** gives 5-6x higher prediction than 3-word (Kappa 0.68 vs 0.11 at K=2)
- **K=5 clusters** on UMAP 50D: calm/contemplative, agitated/conflicted, sensual/embodied, non-English, degenerate
- **Best prediction:** First+Last 8k chars (PCA 20D) or the full conversation → Kappa 0.505 at K=5

## Structure

- **`findings/`** — Analysis writeups, including [`embedding_cluster_analysis.md`](findings/embedding_cluster_analysis.md) (main results)
- **`scripts/`** — Organized pipeline code
  - `generate/` — Data generation (DeepSeek, Gemma, temperature experiments)
  - `embed/` — Embedding generation (OpenAI text-embedding-3-large)
  - `cluster/` — Clustering & analysis (PCA, UMAP, K-means)
  - `predict/` — Cluster prediction from conversation features
  - `visualize/` — Plotting & exploration
- **`data/`** — Source conversations (LMSYS longest English chats)
- **`embeddings/`** — Embedding vectors & clustering results
- **`outputs/`** — Raw generation outputs
- **`archive/`** — Previous experiments (image recognition, SERP, early IFS variants)
