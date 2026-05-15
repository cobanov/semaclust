# Quickstart

## Install

```bash
pip install semaclust
```

## Cluster a list of strings

```python
from semaclust import TextClusterer

texts = [
    "New York", "NYC", "new york city",
    "Los Angeles", "LA",
    "San Francisco", "San Fran", "SF",
]

clusterer = TextClusterer(distance_threshold=1.0)
clusterer.fit(texts)

clusterer.n_clusters_       # 3
clusterer.labels_           # ndarray of cluster ids
clusterer.clusters_         # {0: ['New York', 'NYC', ...], ...}
clusterer.representatives_  # {0: 'NYC', 1: 'LA', 2: 'SF'}
```

## Replace values with their cluster representative

```python
clusterer.transform()
# ['NYC', 'NYC', 'NYC', 'LA', 'LA', 'SF', 'SF', 'SF']

clusterer.fit_transform(texts)
# same as fit(texts).transform()
```

## Tune the threshold

`distance_threshold` controls cluster granularity. Smaller values produce
tighter, more numerous clusters; larger values merge more aggressively. The
useful range with the default encoder (`all-MiniLM-L6-v2`) under
`ward + euclidean` linkage is roughly `0.7` to `1.4` for unit-norm embeddings.
See [benchmarks.md](https://github.com/cobanov/semaclust/blob/main/benchmarks.md)
for per-model sweet spots.

## Custom representative

Pick the longest string as the representative instead of the shortest:

```python
clusterer = TextClusterer(
    distance_threshold=1.0,
    representative_selector=lambda texts: max(texts, key=len),
)
```

## Bring your own encoder

```python
import numpy as np
from semaclust import TextClusterer

class HashEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(texts), 64)).astype(np.float32)

TextClusterer(encoder=HashEncoder()).fit_predict(["a", "b"])
```

## CLI

```bash
semaclust cluster items.txt --threshold 1.0 --output clusters.json
cat items.txt | semaclust replace --threshold 1.0 > normalized.txt
```
