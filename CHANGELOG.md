# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-14

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
