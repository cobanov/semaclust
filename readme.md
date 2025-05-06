# semaclust

**semaclust** (semantic + clustering) is a lightweight Python package for semantic text clustering using sentence embeddings and agglomerative clustering.

## Features

- SentenceTransformer-based text encoding
- Agglomerative clustering with configurable thresholds
- Easily map or replace similar text values

## Installation

```bash
pip install git+https://github.com/cobanov/semaclust.git
```

## Usage

```python
# Create clusterer
clusterer = TextClusterer()

texts = ["New York", "Los Angeles", "San Francisco", "new york city", "LA", "San Fran"]
```

```python
# Get clusters
clusters = clusterer.cluster(texts)
print("Clusters:", clusters)

# Clusters: {1: ['New York', 'new york city'], 2: ['Los Angeles', 'LA'], 0: ['San Francisco', 'San Fran']}
```

```python
# Get replacement map
replacement_map = clusterer.get_replacement_map(texts)
print("\nReplacement map:", replacement_map)

# Replacement map: {'New York': 'New York', 'new york city': 'New York', 'Los Angeles': 'Los Angeles', 'LA': 'Los Angeles', 'San Francisco': 'San Francisco', 'San Fran': 'San Francisco'}
```

```python
# Replace values
replaced_texts = clusterer.replace_values(texts)
print("\nReplaced texts:", replaced_texts)

# Replaced texts: ['New York', 'Los Angeles', 'San Francisco', 'New York', 'Los Angeles', 'San Francisco']
```
