# LLM Introspection Research

## Project overview
Investigates how AI models describe their "inner experience" when given introspective prompts, and whether responses vary by conversation context.

## Key findings
See `findings/embedding_cluster_analysis.md` for detailed results (embedding predictions, cluster analysis, temperature effects).

## Repo structure
- `scripts/` — Organized pipeline (generate, embed, cluster, predict, visualize)
- `data/` — Source conversations (LMSYS longest English chats)
- `embeddings/` — Embedding vectors & clustering results
- `outputs/` — Raw generation outputs
- `findings/` — Analysis writeups
- `archive/` — Previous experiments (image recognition, SERP, early IFS variants)

## Important notes
- Two prompt variants exist: 3-word label ("Tell me what you call this part in 3 words") vs full raw expression ("Now express it in its rawest form"). The original 1000 outputs used 3-word.
- Original generation: `temperature=0.7`, `max_tokens=200`, model `deepseek-chat`, "worker" system prompt
- PCA 20D helps with prediction — raw 3072D overfits at these sample sizes

## Key scripts
- `scripts/generate/ifs_deepseek_longest_chats.py` — original DeepSeek generation (has both prompt variants)
- `scripts/generate/lmsys_ifs_deepseek.py` — generated the 1000 3-word outputs
- `scripts/embed/embedv2.py` — core embedding pipeline (OpenAI text-embedding-3-large, 3072D)
- `scripts/embed/embed_first_last.py` — first/last 8k char embeddings + cluster prediction
- `scripts/generate/temp0_test.py` — temperature=0 vs 0.7 comparison (100-sample)
- `scripts/generate/temp0_fullprompt_1000.py` — full expression, 1000 samples

## API keys
All keys are loaded from `.env` (gitignored). Required variables:
- `OPENAI_API_KEY` — for embeddings (text-embedding-3-large)
- `DEEPSEEK_API_KEY` — for generation (deepseek-chat)
- `ANTHROPIC_API_KEY` — for archived Claude experiments
- `GROK_API_KEY` — for archived Grok experiments
