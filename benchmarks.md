# Benchmarks

Empirical comparison of sentence-embedding models on three semaclust workloads. All numbers come from `benchmarks/run_benchmark.py`. Raw output is in `benchmarks/results.json`.

## Headline findings

- **Cities and feedback are solvable; job titles is not.** Every working model reaches ARI = 1.0 on the cities and feedback test cases at some threshold. None reaches it on job titles - abbreviations + synonyms across roles (`SWE` ~ Programmer ~ Software Engineer, Product Manager ~ Product Owner) defeat every encoder we tried.
- **Bigger is not better.** `all-mpnet-base-v2` (109M) and `Qwen3-Embedding-0.6B` (596M) both *fail* the simplest cities test - they conflate Los Angeles and San Francisco. Smaller general-purpose encoders win on this task.
- **Strongest small option**: `BAAI/bge-small-en-v1.5` - 33M params, perfect on cities (0.55-1.00) and feedback (0.95-1.10), with a 0.05-wide joint sweet spot.
- **Strongest overall**: `mixedbread-ai/mxbai-embed-large-v1` - 335M params, the only model to clear 0.8 ARI on job titles (0.815), with a 0.10-wide cities+feedback overlap at thresholds 0.95-1.05.
- **Model recipe matters.** `nomic-embed-text-v1.5` returns near-random clusters without the `clustering: ` prefix. `mxbai-embed-large-v1` returns near-random clusters without L2 normalization. The benchmark applies both.

## Methodology

- **Clustering**: scikit-learn `AgglomerativeClustering` with `linkage='ward'` and `metric='euclidean'` (semaclust's defaults).
- **Threshold sweep**: `distance_threshold` from 0.05 to 2.00 in 0.05 steps.
- **Metric**: Adjusted Rand Index against the expected grouping. ARI = 1.0 means an exact match (modulo cluster IDs).
- **"Perfect range"**: the contiguous set of thresholds where ARI = 1.0. Wider is better - it means the model is more forgiving of threshold choice.
- **Text normalization**: lowercase, strip whitespace, strip double quotes (same as `TextClusterer(normalize=True)`).
- **Embedding normalization**: L2-normalized at encode time (`SentenceTransformer.encode(..., normalize_embeddings=True)`) so all models live on the unit hypersphere and thresholds are comparable. Without this, models like mxbai-embed-large produce unnormalized vectors and the ward+euclidean threshold scale collapses.
- **Prompt prefix**: the nomic family ships a task-specific prefix system; we prepend `clustering: ` to each text for those models, as recommended in their model card. Other models are encoded as-is.
- All models run on CPU via `sentence-transformers`. Encode time is wall-clock for the test case's texts only (excludes one-time model load).

## Test cases

### cities

Short text with abbreviations and casing variants.

**Inputs** (8 texts, 3 expected clusters):

- 'New York', 'NYC', 'new york city'
- 'Los Angeles', 'LA'
- 'San Francisco', 'San Fran', 'SF'

### job_titles

Medium-length text with synonyms and abbreviations.

**Inputs** (10 texts, 3 expected clusters):

- 'Software Engineer', 'Software Developer', 'SWE', 'Programmer'
- 'Data Scientist', 'ML Engineer', 'Machine Learning Engineer'
- 'Product Manager', 'PM', 'Product Owner'

### feedback

Longer sentences grouped by topic (shipping / defect / support).

**Inputs** (9 texts, 3 expected clusters):

- 'The product arrived on time and works great', 'Fast delivery, item works perfectly', 'Shipping was quick and the quality is excellent'
- 'Product broke after a week of use', 'Item stopped working within days of purchase', 'Defective product, very disappointed with the quality'
- 'Customer service was very helpful and friendly', 'Support team resolved my issue quickly', 'Great help from the customer service representative'

## Summary

ARI per test case and the threshold range that gives a perfect clustering on every test case simultaneously (if any). Bold = perfect (ARI = 1.0).

| Model | Params | Dim | Recipe | cities ARI | job_titles ARI | feedback ARI | cities+feedback threshold |
|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 23M | 384 | L2-norm | **1.000** | 0.672 | **1.000** | none |
| `all-MiniLM-L12-v2` | 33M | 384 | L2-norm | **1.000** | 0.512 | **1.000** | 1.20 |
| `all-mpnet-base-v2` | 109M | 768 | L2-norm | 0.619 | 0.512 | **1.000** | none |
| `bge-small-en-v1.5` | 33M | 384 | L2-norm | **1.000** | 0.672 | **1.000** | 0.95 - 1.00 |
| `bge-m3` | 568M | 1024 | L2-norm | **1.000** | 0.520 | **1.000** | none |
| `nomic-embed-text-v1.5` | 137M | 768 | L2-norm, prefix `clustering:` | **1.000** | 0.520 | **1.000** | 0.70 - 0.85 |
| `nomic-embed-text-v2-moe` | 475M | 768 | L2-norm, prefix `clustering:` | **1.000** | 0.672 | **1.000** | 1.15 - 1.35 |
| `mxbai-embed-large-v1` | 335M | 1024 | L2-norm | **1.000** | 0.815 | **1.000** | 0.95 - 1.05 |
| `embeddinggemma-300m` | - | - | - | _load failed_ | _load failed_ | _load failed_ | - |
| `Qwen3-Embedding-0.6B` | 596M | 1024 | L2-norm | 0.529 | 0.672 | **1.000** | none |

## Detail: cities

| Model | Params | Dim | Best ARI | Threshold (perfect range) | Encode time |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 23M | 384 | **1.000** | 0.70 - 1.20 | 0.20s |
| `all-MiniLM-L12-v2` | 33M | 384 | **1.000** | 0.80 - 1.20 | 0.02s |
| `all-mpnet-base-v2` | 109M | 768 | 0.619 | _no perfect_ (best @ 0.85) | 0.13s |
| `bge-small-en-v1.5` | 33M | 384 | **1.000** | 0.55 - 1.00 | 0.02s |
| `bge-m3` | 568M | 1024 | **1.000** | 0.95 - 1.05 | 0.24s |
| `nomic-embed-text-v1.5` | 137M | 768 | **1.000** | 0.50 - 0.90 | 0.39s |
| `nomic-embed-text-v2-moe` | 475M | 768 | **1.000** | 1.15 - 1.45 | 1.10s |
| `mxbai-embed-large-v1` | 335M | 1024 | **1.000** | 0.80 - 1.05 | 0.13s |
| `embeddinggemma-300m` | - | - | _load failed_ | - | - |
| `Qwen3-Embedding-0.6B` | 596M | 1024 | 0.529 | _no perfect_ (best @ 0.35) | 0.25s |

## Detail: job_titles

| Model | Params | Dim | Best ARI | Threshold (perfect range) | Encode time |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 23M | 384 | 0.672 | _no perfect_ (best @ 1.05) | 0.03s |
| `all-MiniLM-L12-v2` | 33M | 384 | 0.512 | _no perfect_ (best @ 0.85) | 0.01s |
| `all-mpnet-base-v2` | 109M | 768 | 0.512 | _no perfect_ (best @ 0.85) | 0.05s |
| `bge-small-en-v1.5` | 33M | 384 | 0.672 | _no perfect_ (best @ 0.80) | 0.02s |
| `bge-m3` | 568M | 1024 | 0.520 | _no perfect_ (best @ 1.10) | 0.05s |
| `nomic-embed-text-v1.5` | 137M | 768 | 0.520 | _no perfect_ (best @ 0.75) | 0.11s |
| `nomic-embed-text-v2-moe` | 475M | 768 | 0.672 | _no perfect_ (best @ 1.20) | 0.32s |
| `mxbai-embed-large-v1` | 335M | 1024 | 0.815 | _no perfect_ (best @ 1.00) | 0.07s |
| `embeddinggemma-300m` | - | - | _load failed_ | - | - |
| `Qwen3-Embedding-0.6B` | 596M | 1024 | 0.672 | _no perfect_ (best @ 0.75) | 0.07s |

## Detail: feedback

| Model | Params | Dim | Best ARI | Threshold (perfect range) | Encode time |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 23M | 384 | **1.000** | 1.25 - 1.40 | 0.03s |
| `all-MiniLM-L12-v2` | 33M | 384 | **1.000** | 1.20 - 1.45 | 0.01s |
| `all-mpnet-base-v2` | 109M | 768 | **1.000** | 1.10 - 1.40 | 0.05s |
| `bge-small-en-v1.5` | 33M | 384 | **1.000** | 0.95 - 1.10 | 0.01s |
| `bge-m3` | 568M | 1024 | **1.000** | 0.85 | 0.08s |
| `nomic-embed-text-v1.5` | 137M | 768 | **1.000** | 0.70 - 0.85 | 0.17s |
| `nomic-embed-text-v2-moe` | 475M | 768 | **1.000** | 1.10 - 1.35 | 0.54s |
| `mxbai-embed-large-v1` | 335M | 1024 | **1.000** | 0.95 - 1.20 | 0.07s |
| `embeddinggemma-300m` | - | - | _load failed_ | - | - |
| `Qwen3-Embedding-0.6B` | 596M | 1024 | **1.000** | 0.85 - 1.10 | 0.11s |

## Reproducing

```bash
uv sync --extra dev
uv run python benchmarks/run_benchmark.py
uv run python benchmarks/generate_markdown.py
```

The first run downloads several gigabytes of model weights into your HuggingFace cache. `google/embeddinggemma-300m` is gated - accept the license at <https://huggingface.co/google/embeddinggemma-300m> and run `huggingface-cli login` to include it.
