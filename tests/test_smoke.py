"""Smoke tests: import surface and version exposure."""

from __future__ import annotations


def test_package_imports() -> None:
    import semaclust

    assert hasattr(semaclust, "TextClusterer")
    assert hasattr(semaclust, "ClusterResult")
    assert hasattr(semaclust, "Encoder")
    assert hasattr(semaclust, "SentenceTransformerEncoder")
    assert hasattr(semaclust, "NotFittedError")


def test_version_exposed() -> None:
    from semaclust import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts[:2])


def test_all_is_complete() -> None:
    import semaclust

    for name in semaclust.__all__:
        assert hasattr(semaclust, name), f"__all__ lists {name!r} but it is not exported"
