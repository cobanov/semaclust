# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-05-15

### Changed
- `TextClusterer` default `distance_threshold` raised from `0.3` to `1.0`.
  The previous default was incompatible with the README quickstart at the
  documented threshold; `1.0` works for unit-norm sentence-transformer
  embeddings under `ward + euclidean` linkage. Existing users who passed an
  explicit threshold are unaffected.
- `semaclust` CLI default `--threshold` raised from `0.3` to `1.0` to match.
- `SentenceTransformerEncoder` now passes `normalize_embeddings=True` to the
  underlying model so unnormalized encoders (e.g. mxbai-embed-large) cluster
  correctly under the default `ward + euclidean` setup.

### Added
- `benchmarks/` directory with reproducible benchmark across 9 sentence
  encoders (MiniLM, mpnet, BGE small/m3, nomic v1.5/v2-moe, mxbai-embed-large,
  Qwen3-Embedding) and 3 test cases (cities, job titles, customer feedback).
  See [benchmarks.md](benchmarks.md).
- README "Choosing a model and threshold" section with empirical guidance.
- `benchmarks` optional dependency group (adds `einops` for the nomic models).

## [0.3.0] - 2026-05-14

Skips 0.2.0, which was published from an earlier draft on PyPI. This release
is the official modernization of the library.

### Added
- scikit-learn style `fit`, `fit_predict`, `fit_transform`, `transform` API
- `ClusterResult` dataclass exposing `labels`, `clusters`, `representatives`
- `Encoder` protocol so users can plug in any embedding model
- `SentenceTransformerEncoder` with lazy model loading
- `semaclust` CLI (`cluster`, `replace`) built on typer
- `random_state` parameter for reproducible encoding
- `representative_selector` as a constructor argument
- `NotFittedError` raised when accessing fitted state too early
- `py.typed` marker, mypy strict mode in CI
- GitHub Actions CI matrix on Python 3.10 to 3.13
- Pre-commit config with ruff and mypy
- mkdocs-material documentation site

### Changed
- Minimum Python version bumped to 3.10
- Project layout migrated to `src/` with hatchling build backend
- Default representative is now the shortest text (was: first text)
- Empty input no longer raises; returns an empty result

### Removed
- `cluster(texts)`, `get_replacement_map(texts, ...)`, `replace_values(texts, ...)`
  methods. See the [migration guide](docs/migration.md).
- `setuptools` build backend, `requirements.txt`

## [0.1.1] - prior

- Initial public release on PyPI.
