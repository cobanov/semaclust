# semaclust

[![CI](https://github.com/cobanov/semaclust/actions/workflows/ci.yml/badge.svg)](https://github.com/cobanov/semaclust/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/semaclust.svg)](https://pypi.org/project/semaclust/)
[![Python versions](https://img.shields.io/pypi/pyversions/semaclust.svg)](https://pypi.org/project/semaclust/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**semaclust** (semantic + clustering) is a small Python library for clustering
similar strings using sentence embeddings and agglomerative clustering. It is
useful for deduplicating free-text fields, normalizing user-entered values, and
collapsing spelling or formatting variants into a canonical form.

## Installation

```bash
pip install semaclust
# or
uv add semaclust
```

Install from source:

```bash
pip install git+https://github.com/cobanov/semaclust.git
```

## Quickstart

```python
from semaclust import TextClusterer

texts = [
    "New York", "NYC", "new york city",
    "Los Angeles", "LA",
    "San Francisco", "San Fran", "SF",
]

clusterer = TextClusterer(distance_threshold=0.5)
clusterer.fit(texts)

print(clusterer.n_clusters_)
# 3

print(clusterer.clusters_)
# {0: ['New York', 'NYC', 'new york city'],
#  1: ['Los Angeles', 'LA'],
#  2: ['San Francisco', 'San Fran', 'SF']}

print(clusterer.transform())
# ['NYC', 'NYC', 'NYC', 'LA', 'LA', 'SF', 'SF', 'SF']
```

`fit_transform` is the one-call shortcut:

```python
TextClusterer(distance_threshold=0.5).fit_transform(texts)
```

## API at a glance

| Method | Returns | Purpose |
|---|---|---|
| `fit(texts)` | `self` | Cluster and store fitted attributes |
| `fit_predict(texts)` | `ndarray[int]` | Cluster labels per input text |
| `fit_transform(texts)` | `list[str]` | Each text replaced with its representative |
| `transform(texts=None)` | `list[str]` | Replace texts seen at fit time |
| `get_replacement_map()` | `dict[str, str]` | Mapping from original to representative |
| `result_` | `ClusterResult` | Frozen dataclass with labels, clusters, reps |

Fitted attributes (sklearn convention): `labels_`, `clusters_`,
`representatives_`, `n_clusters_`, `texts_`.

## Plugging in your own encoder

Any object implementing `encode(texts: list[str]) -> np.ndarray` satisfies the
`Encoder` protocol:

```python
from semaclust import TextClusterer, Encoder
import numpy as np

class MyEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.random.rand(len(texts), 384).astype(np.float32)

clusterer = TextClusterer(encoder=MyEncoder())
```

## CLI

semaclust ships a small `typer`-based CLI:

```bash
# Cluster lines from a file, write JSON
semaclust cluster items.txt --threshold 0.4 --output clusters.json

# Replace each line with its cluster representative
cat items.txt | semaclust replace --threshold 0.4
```

Run `semaclust --help` for the full reference.

## Migration from 0.1.x

The 0.2 release is a breaking change. The single `cluster(texts)` entry point
is gone; the new API mirrors scikit-learn:

| 0.1.x | 0.2.x |
|---|---|
| `clusterer.cluster(texts)` | `clusterer.fit(texts).clusters_` |
| `clusterer.get_replacement_map(texts)` | `clusterer.fit(texts).get_replacement_map()` |
| `clusterer.replace_values(texts)` | `clusterer.fit_transform(texts)` |

The `representative_selector` argument moved from per-call to the constructor.

## Development

```bash
git clone https://github.com/cobanov/semaclust.git
cd semaclust
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run mypy src/semaclust
```

Install the pre-commit hooks once:

```bash
uv run pre-commit install
```

## License

MIT, see [LICENSE](LICENSE).
