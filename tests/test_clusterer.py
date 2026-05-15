"""Unit tests for the TextClusterer API."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from semaclust import ClusterResult, NotFittedError, TextClusterer
from tests.conftest import FakeEncoder


def _build(encoder: FakeEncoder, **kwargs: object) -> TextClusterer:
    return TextClusterer(encoder=encoder, distance_threshold=0.2, **kwargs)


def test_fit_returns_self(cities_encoder: FakeEncoder, city_texts: Sequence[str]) -> None:
    clusterer = _build(cities_encoder)
    assert clusterer.fit(city_texts) is clusterer


def test_fit_populates_attributes(cities_encoder: FakeEncoder, city_texts: Sequence[str]) -> None:
    clusterer = _build(cities_encoder)
    clusterer.fit(city_texts)

    assert clusterer.labels_.shape == (len(city_texts),)
    assert clusterer.labels_.dtype.kind == "i"
    assert clusterer.n_clusters_ == 3
    assert set(clusterer.clusters_.keys()) == set(clusterer.representatives_.keys())
    assert clusterer.texts_ == tuple(city_texts)


def test_groups_synonyms_together(cities_encoder: FakeEncoder, city_texts: Sequence[str]) -> None:
    clusterer = _build(cities_encoder).fit(city_texts)

    def cluster_of(text: str) -> int:
        return clusterer.labels_[city_texts.index(text)]

    assert cluster_of("New York") == cluster_of("NYC") == cluster_of("new york city")
    assert cluster_of("Los Angeles") == cluster_of("LA")
    assert cluster_of("San Francisco") == cluster_of("San Fran") == cluster_of("SF")
    assert cluster_of("New York") != cluster_of("Los Angeles")


def test_fit_predict_returns_labels(cities_encoder: FakeEncoder, city_texts: Sequence[str]) -> None:
    clusterer = _build(cities_encoder)
    labels = clusterer.fit_predict(city_texts)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (len(city_texts),)


def test_fit_transform_replaces_with_representatives(
    cities_encoder: FakeEncoder, city_texts: Sequence[str]
) -> None:
    clusterer = _build(cities_encoder)
    out = clusterer.fit_transform(city_texts)
    assert len(out) == len(city_texts)
    assert len(set(out)) == 3


def test_transform_passes_through_unseen_texts(
    cities_encoder: FakeEncoder, city_texts: Sequence[str]
) -> None:
    clusterer = _build(cities_encoder).fit(city_texts)
    out = clusterer.transform(["NYC", "Totally Unknown"])
    assert out[0] != "NYC" or out[0] in {"NYC", "New York", "new york city"}
    assert out[1] == "Totally Unknown"


def test_transform_before_fit_raises(cities_encoder: FakeEncoder) -> None:
    clusterer = _build(cities_encoder)
    with pytest.raises(NotFittedError):
        clusterer.transform(["foo"])


def test_result_property_before_fit_raises(cities_encoder: FakeEncoder) -> None:
    clusterer = _build(cities_encoder)
    with pytest.raises(NotFittedError):
        _ = clusterer.result_


def test_empty_input_returns_empty(cities_encoder: FakeEncoder) -> None:
    clusterer = _build(cities_encoder).fit([])
    assert clusterer.n_clusters_ == 0
    assert clusterer.labels_.size == 0
    assert clusterer.transform() == []


def test_single_text_is_its_own_cluster(cities_encoder: FakeEncoder) -> None:
    clusterer = _build(cities_encoder).fit(["NYC"])
    assert clusterer.n_clusters_ == 1
    assert clusterer.transform() == ["NYC"]


def test_result_dataclass_shape(cities_encoder: FakeEncoder, city_texts: Sequence[str]) -> None:
    clusterer = _build(cities_encoder).fit(city_texts)
    result = clusterer.result_
    assert isinstance(result, ClusterResult)
    assert result.n_clusters == clusterer.n_clusters_
    assert result.texts == tuple(city_texts)


def test_replacement_map_covers_all_inputs(
    cities_encoder: FakeEncoder, city_texts: Sequence[str]
) -> None:
    clusterer = _build(cities_encoder).fit(city_texts)
    mapping = clusterer.get_replacement_map()
    assert set(mapping) == set(city_texts)
    for text, rep in mapping.items():
        assert rep in clusterer.representatives_.values()
        cluster_id = clusterer.labels_[city_texts.index(text)]
        assert clusterer.representatives_[int(cluster_id)] == rep


def test_custom_representative_selector(
    cities_encoder: FakeEncoder, city_texts: Sequence[str]
) -> None:
    clusterer = TextClusterer(
        encoder=cities_encoder,
        distance_threshold=0.2,
        representative_selector=lambda texts: max(texts, key=len),
    ).fit(city_texts)
    mapping = clusterer.get_replacement_map()
    for rep in clusterer.representatives_.values():
        cluster_members = [m for m in city_texts if mapping.get(m) == rep]
        assert rep == max(cluster_members, key=len)


def test_determinism_with_random_state(
    cities_encoder: FakeEncoder, city_texts: Sequence[str]
) -> None:
    a = _build(cities_encoder, random_state=42).fit_predict(city_texts)
    b = _build(cities_encoder, random_state=42).fit_predict(city_texts)
    np.testing.assert_array_equal(a, b)


def test_threshold_controls_granularity(
    cities_encoder: FakeEncoder, city_texts: Sequence[str]
) -> None:
    tight = TextClusterer(encoder=cities_encoder, distance_threshold=0.001).fit(city_texts)
    loose = TextClusterer(encoder=cities_encoder, distance_threshold=10.0).fit(city_texts)
    assert tight.n_clusters_ >= loose.n_clusters_
    assert loose.n_clusters_ == 1


@pytest.mark.parametrize("normalize", [True, False])
def test_normalize_toggle_runs(
    cities_encoder: FakeEncoder, city_texts: Sequence[str], normalize: bool
) -> None:
    clusterer = TextClusterer(
        encoder=cities_encoder, distance_threshold=0.2, normalize=normalize
    ).fit(city_texts)
    assert clusterer.n_clusters_ >= 1


def test_repr_unfitted_includes_config(cities_encoder: FakeEncoder) -> None:
    clusterer = _build(cities_encoder)
    text = repr(clusterer)
    assert "TextClusterer(" in text
    assert "distance_threshold=0.2" in text
    assert "linkage='ward'" in text
    assert "not fitted" in text


def test_repr_fitted_shows_cluster_count(
    cities_encoder: FakeEncoder, city_texts: Sequence[str]
) -> None:
    clusterer = _build(cities_encoder).fit(city_texts)
    text = repr(clusterer)
    assert f"n_clusters={clusterer.n_clusters_}" in text
    assert f"n_texts={len(city_texts)}" in text
    assert "not fitted" not in text


def test_cluster_result_repr_is_summary(
    cities_encoder: FakeEncoder, city_texts: Sequence[str]
) -> None:
    result = _build(cities_encoder).fit(city_texts).result_
    text = repr(result)
    assert text.startswith("ClusterResult(")
    assert f"n_clusters={result.n_clusters}" in text
    assert f"n_texts={len(city_texts)}" in text
    # The verbose dataclass default would dump the full labels array; ours doesn't.
    assert "labels=" not in text
