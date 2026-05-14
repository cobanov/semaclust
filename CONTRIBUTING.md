# Contributing

Thanks for considering a contribution. This page covers the development setup
and the small set of conventions the project follows.

## Setup

```bash
git clone https://github.com/cobanov/semaclust.git
cd semaclust
uv sync --extra dev
uv run pre-commit install
```

## Day-to-day

```bash
# Run the test suite
uv run pytest

# Skip integration tests (anything that downloads the real model)
uv run pytest -m "not integration"

# Lint, format, type check
uv run ruff check src tests
uv run ruff format src tests
uv run mypy src/semaclust
```

Pre-commit runs ruff and mypy on every commit. If a hook reports a problem,
fix it and re-stage rather than passing `--no-verify`.

## Pull request checklist

- [ ] Added or updated tests for behavior changes
- [ ] `uv run ruff check src tests` is clean
- [ ] `uv run mypy src/semaclust` is clean
- [ ] `uv run pytest` is green
- [ ] CHANGELOG updated under `[Unreleased]`
- [ ] Public API changes have docstrings and a doc update

## Releasing

Releases are cut by maintainers from `main`:

1. Bump `__version__` in `src/semaclust/_version.py`
2. Move `[Unreleased]` entries under a new `[x.y.z] - YYYY-MM-DD` heading in `CHANGELOG.md`
3. Commit: `chore(release): vX.Y.Z`
4. Tag: `git tag vX.Y.Z && git push --tags`
5. The release workflow builds, publishes to PyPI via trusted publishing, and creates a GitHub release

## Code of conduct

Be kind, be specific, assume good intent.
