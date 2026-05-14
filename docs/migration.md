# Migration from 0.1.x to 0.2.0

The 0.2 release is a deliberate break. The old single-method API is replaced
with a scikit-learn style estimator.

## At a glance

| 0.1.x | 0.2.x |
|---|---|
| `TextClusterer().cluster(texts)` | `TextClusterer().fit(texts).clusters_` |
| `TextClusterer().get_replacement_map(texts)` | `TextClusterer().fit(texts).get_replacement_map()` |
| `TextClusterer().replace_values(texts)` | `TextClusterer().fit_transform(texts)` |
| `replace_values(texts, representative_selector=fn)` | `TextClusterer(representative_selector=fn).fit_transform(texts)` |

## Why the break

- **Predictable state.** Each `fit` call stores its outcome on the instance
  (`labels_`, `clusters_`, etc.), letting downstream code inspect the result
  without re-running the clustering.
- **Composable.** Following the scikit-learn convention means the estimator
  drops cleanly into pipelines.
- **One configuration site.** Per-call options like the representative
  selector are now constructor arguments. Configure once, run many times.

## Step by step

### Before

```python
from semaclust import TextClusterer

clusterer = TextClusterer()
clusters = clusterer.cluster(texts)
mapping = clusterer.get_replacement_map(texts, representative_selector=lambda xs: xs[0])
replaced = clusterer.replace_values(texts)
```

### After

```python
from semaclust import TextClusterer

clusterer = TextClusterer(representative_selector=lambda xs: xs[0])
clusterer.fit(texts)

clusters = clusterer.clusters_
mapping = clusterer.get_replacement_map()
replaced = clusterer.transform()
```

## Other changes

- Minimum Python is now 3.10.
- Built-in CLI: `semaclust cluster` and `semaclust replace`.
- The package ships `py.typed` and is mypy strict.
- Encoders are pluggable via the `Encoder` protocol.
- Empty input no longer raises; it returns an empty result.
