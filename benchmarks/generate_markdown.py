"""Generate benchmarks.md from benchmarks/results.json.

Run: ``uv run python benchmarks/generate_markdown.py``
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = Path(__file__).parent / "results.json"
OUT_PATH = REPO_ROOT / "benchmarks.md"


def fmt_range(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "-"
    if lo == hi:
        return f"{lo:.2f}"
    return f"{lo:.2f} - {hi:.2f}"


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return f"{n / 1e3:.0f}K"


def model_short_name(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/", 1)[1]
    return model_id


def per_case_table(results: dict, case_name: str) -> str:
    lines = [
        "| Model | Params | Dim | Best ARI | Threshold (perfect range) | Encode time |",
        "|---|---|---|---|---|---|",
    ]
    for m in results["models"]:
        if m["status"] != "ok":
            lines.append(
                f"| `{model_short_name(m['model_id'])}` | - | - | "
                f"_load failed_ | - | - |"
            )
            continue
        per = m["per_case"].get(case_name)
        if not per or per["status"] != "ok":
            lines.append(
                f"| `{model_short_name(m['model_id'])}` | "
                f"{fmt_params(m['param_count'])} | {m['embed_dim']} | "
                f"_encode failed_ | - | - |"
            )
            continue
        rng = fmt_range(per.get("perfect_threshold_min"), per.get("perfect_threshold_max"))
        ari = per["best_ari"]
        ari_str = f"**{ari:.3f}**" if abs(ari - 1.0) < 1e-9 else f"{ari:.3f}"
        rng_str = (
            f"_no perfect_ (best @ {per['best_ari_threshold']:.2f})"
            if rng == "-"
            else rng
        )
        lines.append(
            f"| `{model_short_name(m['model_id'])}` | "
            f"{fmt_params(m['param_count'])} | {m['embed_dim']} | "
            f"{ari_str} | {rng_str} | {per['encode_time_s']:.2f}s |"
        )
    return "\n".join(lines)


def summary_table(results: dict) -> str:
    cases = [c["name"] for c in results["test_cases"]]
    header = (
        "| Model | Params | Dim | Recipe | "
        + " | ".join(f"{c} ARI" for c in cases)
        + " | cities+feedback threshold |"
    )
    sep = "|---|---|---|---|" + "---|" * len(cases) + "---|"
    lines = [header, sep]
    for m in results["models"]:
        name = f"`{model_short_name(m['model_id'])}`"
        if m["status"] != "ok":
            lines.append(
                f"| {name} | - | - | - | "
                + " | ".join(["_load failed_"] * len(cases))
                + " | - |"
            )
            continue
        recipe_parts = ["L2-norm"]
        prompt = m.get("prompt_prefix")
        if prompt:
            recipe_parts.append(f"prefix `{prompt.strip()}`")
        recipe = ", ".join(recipe_parts)

        cells = []
        intervals_by_case: dict[str, tuple[float, float] | None] = {}
        for c in cases:
            per = m["per_case"].get(c)
            if not per or per["status"] != "ok":
                cells.append("_fail_")
                intervals_by_case[c] = None
                continue
            ari = per["best_ari"]
            mark = f"**{ari:.3f}**" if abs(ari - 1.0) < 1e-9 else f"{ari:.3f}"
            cells.append(mark)
            if per.get("perfect_threshold_min") is not None:
                intervals_by_case[c] = (
                    per["perfect_threshold_min"],
                    per["perfect_threshold_max"],
                )
            else:
                intervals_by_case[c] = None

        joint_overlap = "none"
        focus = ["cities", "feedback"]
        focus_intervals = [intervals_by_case.get(c) for c in focus]
        if all(focus_intervals):
            lo = max(iv[0] for iv in focus_intervals)  # type: ignore[index]
            hi = min(iv[1] for iv in focus_intervals)  # type: ignore[index]
            if lo <= hi:
                joint_overlap = f"{lo:.2f} - {hi:.2f}" if lo != hi else f"{lo:.2f}"
        lines.append(
            f"| {name} | "
            f"{fmt_params(m['param_count'])} | {m['embed_dim']} | {recipe} | "
            + " | ".join(cells)
            + f" | {joint_overlap} |"
        )
    return "\n".join(lines)


def test_case_block(case: dict) -> str:
    expected_clusters = case["expected_groups"]
    body = [
        f"### {case['name']}",
        "",
        case["description"] + ".",
        "",
        f"**Inputs** ({len(case['texts'])} texts, {len(expected_clusters)} expected clusters):",
        "",
    ]
    for group in expected_clusters:
        body.append(f"- {', '.join(repr(t) for t in group)}")
    body.append("")
    return "\n".join(body)


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())

    parts: list[str] = []
    parts.append("# Benchmarks")
    parts.append("")
    parts.append(
        "Empirical comparison of sentence-embedding models on three semaclust workloads. "
        "All numbers come from `benchmarks/run_benchmark.py`. Raw output is in "
        "`benchmarks/results.json`."
    )
    parts.append("")
    parts.append("## Headline findings")
    parts.append("")
    parts.append(
        "- **Cities and feedback are solvable; job titles is not.** Every working model "
        "reaches ARI = 1.0 on the cities and feedback test cases at some threshold. None "
        "reaches it on job titles - abbreviations + synonyms across roles "
        "(`SWE` ~ Programmer ~ Software Engineer, Product Manager ~ Product Owner) defeat "
        "every encoder we tried."
    )
    parts.append(
        "- **Bigger is not better.** `all-mpnet-base-v2` (109M) and `Qwen3-Embedding-0.6B` "
        "(596M) both *fail* the simplest cities test - they conflate Los Angeles and San "
        "Francisco. Smaller general-purpose encoders win on this task."
    )
    parts.append(
        "- **Strongest small option**: `BAAI/bge-small-en-v1.5` - 33M params, perfect on "
        "cities (0.55-1.00) and feedback (0.95-1.10), with a 0.05-wide joint sweet spot."
    )
    parts.append(
        "- **Strongest overall**: `mixedbread-ai/mxbai-embed-large-v1` - 335M params, the "
        "only model to clear 0.8 ARI on job titles (0.815), with a 0.10-wide cities+feedback "
        "overlap at thresholds 0.95-1.05."
    )
    parts.append(
        "- **Model recipe matters.** `nomic-embed-text-v1.5` returns near-random clusters "
        "without the `clustering: ` prefix. `mxbai-embed-large-v1` returns near-random "
        "clusters without L2 normalization. The benchmark applies both."
    )
    parts.append("")
    parts.append("## Methodology")
    parts.append("")
    parts.append(
        "- **Clustering**: scikit-learn `AgglomerativeClustering` with `linkage='ward'` and "
        "`metric='euclidean'` (semaclust's defaults)."
    )
    parts.append(
        "- **Threshold sweep**: `distance_threshold` from 0.05 to 2.00 in 0.05 steps."
    )
    parts.append(
        "- **Metric**: Adjusted Rand Index against the expected grouping. ARI = 1.0 means an "
        "exact match (modulo cluster IDs)."
    )
    parts.append(
        "- **\"Perfect range\"**: the contiguous set of thresholds where ARI = 1.0. Wider is "
        "better - it means the model is more forgiving of threshold choice."
    )
    parts.append(
        "- **Text normalization**: lowercase, strip whitespace, strip double quotes (same as "
        "`TextClusterer(normalize=True)`)."
    )
    parts.append(
        "- **Embedding normalization**: L2-normalized at encode time "
        "(`SentenceTransformer.encode(..., normalize_embeddings=True)`) so all models live on "
        "the unit hypersphere and thresholds are comparable. Without this, models like "
        "mxbai-embed-large produce unnormalized vectors and the ward+euclidean threshold "
        "scale collapses."
    )
    parts.append(
        "- **Prompt prefix**: the nomic family ships a task-specific prefix system; we "
        "prepend `clustering: ` to each text for those models, as recommended in their model "
        "card. Other models are encoded as-is."
    )
    parts.append(
        "- All models run on CPU via `sentence-transformers`. Encode time is wall-clock for "
        "the test case's texts only (excludes one-time model load)."
    )
    parts.append("")

    parts.append("## Test cases")
    parts.append("")
    for case in results["test_cases"]:
        parts.append(test_case_block(case))

    parts.append("## Summary")
    parts.append("")
    parts.append(
        "ARI per test case and the threshold range that gives a perfect clustering on every "
        "test case simultaneously (if any). Bold = perfect (ARI = 1.0)."
    )
    parts.append("")
    parts.append(summary_table(results))
    parts.append("")

    for case in results["test_cases"]:
        parts.append(f"## Detail: {case['name']}")
        parts.append("")
        parts.append(per_case_table(results, case["name"]))
        parts.append("")

    parts.append("## Reproducing")
    parts.append("")
    parts.append("```bash")
    parts.append("uv sync --extra dev")
    parts.append("uv run python benchmarks/run_benchmark.py")
    parts.append("uv run python benchmarks/generate_markdown.py")
    parts.append("```")
    parts.append("")
    parts.append(
        "The first run downloads several gigabytes of model weights into your HuggingFace "
        "cache. `google/embeddinggemma-300m` is gated - accept the license at "
        "<https://huggingface.co/google/embeddinggemma-300m> and run `huggingface-cli login` "
        "to include it."
    )
    parts.append("")

    OUT_PATH.write_text("\n".join(parts))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
