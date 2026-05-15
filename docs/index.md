# semaclust

**semaclust** clusters semantically similar strings using sentence embeddings
and agglomerative clustering. It is a small, focused tool for deduplicating
free-text fields, normalizing user-entered values, and collapsing spelling or
formatting variants into a canonical form.

```python
from semaclust import TextClusterer

clusterer = TextClusterer(distance_threshold=1.0)
clusterer.fit(["New York", "NYC", "Los Angeles", "LA"])
print(clusterer.clusters_)
```

## Why semaclust

- **Drop-in API.** scikit-learn style `fit` / `fit_predict` / `fit_transform`.
- **Pluggable encoders.** Anything with `encode(list[str]) -> np.ndarray` works.
- **CLI included.** `semaclust cluster items.txt` returns JSON.
- **Typed.** Ships with `py.typed`, mypy strict in CI.

## Next steps

- Read the [quickstart](quickstart.md)
- Browse the [API reference](api.md)
- Coming from 0.1.x? See the [migration guide](migration.md)
