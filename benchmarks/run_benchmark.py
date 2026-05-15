"""Benchmark sentence-embedding models against semaclust test cases.

For each (model, test_case) we:
- Encode the texts once and measure encoding time
- Sweep distance_threshold from 0.1 to 2.0 in 0.05 steps
- Find the threshold range (if any) where the resulting clustering exactly
  matches the expected groups (adjusted Rand index == 1.0)
- Record best ARI achieved and the threshold (or threshold midpoint of the
  perfect range)

Writes raw results to ``benchmarks/results.json``.

Run with: ``uv run python benchmarks/run_benchmark.py``
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


@dataclass
class TestCase:
    name: str
    description: str
    texts: list[str]
    expected_groups: list[list[str]]

    def expected_labels(self) -> list[int]:
        mapping: dict[str, int] = {}
        for cluster_id, group in enumerate(self.expected_groups):
            for text in group:
                mapping[text] = cluster_id
        return [mapping[t] for t in self.texts]


TEST_CASES = [
    TestCase(
        name="cities",
        description="Short text with abbreviations and casing variants",
        texts=[
            "New York", "NYC", "new york city",
            "Los Angeles", "LA",
            "San Francisco", "San Fran", "SF",
        ],
        expected_groups=[
            ["New York", "NYC", "new york city"],
            ["Los Angeles", "LA"],
            ["San Francisco", "San Fran", "SF"],
        ],
    ),
    TestCase(
        name="job_titles",
        description="Medium-length text with synonyms and abbreviations",
        texts=[
            "Software Engineer", "Software Developer", "SWE", "Programmer",
            "Data Scientist", "ML Engineer", "Machine Learning Engineer",
            "Product Manager", "PM", "Product Owner",
        ],
        expected_groups=[
            ["Software Engineer", "Software Developer", "SWE", "Programmer"],
            ["Data Scientist", "ML Engineer", "Machine Learning Engineer"],
            ["Product Manager", "PM", "Product Owner"],
        ],
    ),
    TestCase(
        name="feedback",
        description="Longer sentences grouped by topic (shipping / defect / support)",
        texts=[
            "The product arrived on time and works great",
            "Fast delivery, item works perfectly",
            "Shipping was quick and the quality is excellent",
            "Product broke after a week of use",
            "Item stopped working within days of purchase",
            "Defective product, very disappointed with the quality",
            "Customer service was very helpful and friendly",
            "Support team resolved my issue quickly",
            "Great help from the customer service representative",
        ],
        expected_groups=[
            [
                "The product arrived on time and works great",
                "Fast delivery, item works perfectly",
                "Shipping was quick and the quality is excellent",
            ],
            [
                "Product broke after a week of use",
                "Item stopped working within days of purchase",
                "Defective product, very disappointed with the quality",
            ],
            [
                "Customer service was very helpful and friendly",
                "Support team resolved my issue quickly",
                "Great help from the customer service representative",
            ],
        ],
    ),
]


# (model_id, trust_remote_code, prompt_prefix)
# The prompt prefix is prepended to each text before encoding. Some models
# (notably the nomic family) ship recommended task-specific prefixes; the rest
# accept None and are encoded as-is.
MODELS: list[tuple[str, bool, str | None]] = [
    ("sentence-transformers/all-MiniLM-L6-v2", False, None),
    ("sentence-transformers/all-MiniLM-L12-v2", False, None),
    ("sentence-transformers/all-mpnet-base-v2", False, None),
    ("BAAI/bge-small-en-v1.5", False, None),
    ("BAAI/bge-m3", False, None),
    ("nomic-ai/nomic-embed-text-v1.5", True, "clustering: "),
    ("nomic-ai/nomic-embed-text-v2-moe", True, "clustering: "),
    ("mixedbread-ai/mxbai-embed-large-v1", False, None),
    ("google/embeddinggemma-300m", False, None),
    ("Qwen/Qwen3-Embedding-0.6B", False, None),
]

THRESHOLDS = [round(0.05 + 0.05 * i, 2) for i in range(40)]  # 0.05 .. 2.00


def normalize_text(text: str) -> str:
    return text.lower().strip().replace('"', "")


def cluster_at(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    if len(embeddings) <= 1:
        return np.zeros(len(embeddings), dtype=np.int_)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="euclidean",
        linkage="ward",
    )
    return clustering.fit_predict(embeddings).astype(np.int_)


def evaluate_model(
    model_id: str,
    trust_remote_code: bool,
    prompt_prefix: str | None = None,
) -> dict:
    """Load model, encode test cases, sweep thresholds, return per-test results.

    All embeddings are L2-normalized before clustering so thresholds are
    comparable across models. ``prompt_prefix`` is prepended to each text when
    set (used for the nomic family's ``clustering: `` recipe).
    """
    from sentence_transformers import SentenceTransformer

    print(f"\n=== {model_id} ===", flush=True)
    t0 = time.time()
    try:
        model = SentenceTransformer(model_id, trust_remote_code=trust_remote_code)
    except Exception as e:
        print(f"  LOAD FAILED: {e!r}", flush=True)
        return {
            "model_id": model_id,
            "status": "load_failed",
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }
    load_time = time.time() - t0
    embed_dim = int(model.get_sentence_embedding_dimension() or 0)
    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"  loaded in {load_time:.2f}s, embed_dim={embed_dim}, "
        f"params={param_count / 1e6:.1f}M, prompt_prefix={prompt_prefix!r}",
        flush=True,
    )

    per_case: dict[str, dict] = {}
    for case in TEST_CASES:
        prepared = [normalize_text(t) for t in case.texts]
        if prompt_prefix:
            prepared = [prompt_prefix + t for t in prepared]
        t0 = time.time()
        try:
            embeddings = model.encode(
                prepared,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as e:
            print(f"  [{case.name}] ENCODE FAILED: {e!r}", flush=True)
            per_case[case.name] = {
                "status": "encode_failed",
                "error": repr(e),
            }
            continue
        encode_time = time.time() - t0
        embeddings = np.asarray(embeddings, dtype=np.float32)

        expected_labels = case.expected_labels()
        sweep: list[dict] = []
        perfect_thresholds: list[float] = []
        best_ari = -2.0
        best_ari_threshold = None
        best_ari_n_clusters = 0
        for t in THRESHOLDS:
            labels = cluster_at(embeddings, t)
            ari = float(adjusted_rand_score(expected_labels, labels))
            n_clusters = len(set(labels.tolist()))
            sweep.append({"threshold": t, "n_clusters": n_clusters, "ari": round(ari, 4)})
            if ari > best_ari:
                best_ari = ari
                best_ari_threshold = t
                best_ari_n_clusters = n_clusters
            if abs(ari - 1.0) < 1e-9:
                perfect_thresholds.append(t)

        per_case[case.name] = {
            "status": "ok",
            "encode_time_s": round(encode_time, 3),
            "best_ari": round(best_ari, 4),
            "best_ari_threshold": best_ari_threshold,
            "best_ari_n_clusters": best_ari_n_clusters,
            "perfect_thresholds": perfect_thresholds,
            "perfect_threshold_min": min(perfect_thresholds) if perfect_thresholds else None,
            "perfect_threshold_max": max(perfect_thresholds) if perfect_thresholds else None,
            "sweep": sweep,
        }
        if perfect_thresholds:
            print(
                f"  [{case.name}] best_ari={best_ari:.3f}  perfect at "
                f"{perfect_thresholds[0]:.2f}-{perfect_thresholds[-1]:.2f}  "
                f"encode={encode_time:.2f}s",
                flush=True,
            )
        else:
            print(
                f"  [{case.name}] best_ari={best_ari:.3f} @ t={best_ari_threshold}  "
                f"(no perfect threshold)  encode={encode_time:.2f}s",
                flush=True,
            )

    return {
        "model_id": model_id,
        "status": "ok",
        "load_time_s": round(load_time, 3),
        "embed_dim": embed_dim,
        "param_count": int(param_count),
        "prompt_prefix": prompt_prefix,
        "per_case": per_case,
    }


def main() -> None:
    results: dict = {
        "thresholds": THRESHOLDS,
        "test_cases": [asdict(c) for c in TEST_CASES],
        "models": [],
    }
    for model_id, trust_remote_code, prompt_prefix in MODELS:
        try:
            results["models"].append(
                evaluate_model(model_id, trust_remote_code, prompt_prefix)
            )
        except Exception as e:
            print(f"FATAL on {model_id}: {e!r}", flush=True)
            results["models"].append(
                {
                    "model_id": model_id,
                    "status": "fatal",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
            )

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
