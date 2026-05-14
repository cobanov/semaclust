"""Smoke tests for the typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from semaclust import __version__
from semaclust.cli import app
from tests.conftest import FakeEncoder


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def patched_encoder(cities_anchors: dict[str, np.ndarray]) -> FakeEncoder:
    return FakeEncoder(cities_anchors)


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "cluster" in result.stdout
    assert "replace" in result.stdout


def _patch_clusterer(monkeypatch: pytest.MonkeyPatch, fake_encoder: FakeEncoder) -> None:
    from semaclust import cli

    real_cls = cli.TextClusterer

    def factory(encoder: str, *, distance_threshold: float = 0.3) -> object:
        return real_cls(encoder=fake_encoder, distance_threshold=distance_threshold)

    monkeypatch.setattr(cli, "TextClusterer", factory)


def test_cluster_command_writes_json(
    runner: CliRunner,
    tmp_path: Path,
    patched_encoder: FakeEncoder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clusterer(monkeypatch, patched_encoder)

    input_path = tmp_path / "input.txt"
    input_path.write_text("New York\nNYC\nLos Angeles\nLA\n", encoding="utf-8")
    output_path = tmp_path / "out.json"

    result = runner.invoke(
        app,
        ["cluster", str(input_path), "--threshold", "0.2", "--output", str(output_path)],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["n_clusters"] == 2
    assert set(payload["clusters"].keys()) == set(payload["representatives"].keys())


def test_replace_command_to_stdout(
    runner: CliRunner,
    tmp_path: Path,
    patched_encoder: FakeEncoder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clusterer(monkeypatch, patched_encoder)

    input_path = tmp_path / "input.txt"
    input_path.write_text("New York\nNYC\nLA\n", encoding="utf-8")

    result = runner.invoke(app, ["replace", str(input_path), "--threshold", "0.2"])
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line]
    assert len(lines) == 3


def test_empty_input_exits_with_error(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_encoder: FakeEncoder
) -> None:
    _patch_clusterer(monkeypatch, patched_encoder)
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n   \n", encoding="utf-8")

    result = runner.invoke(app, ["cluster", str(empty)])
    assert result.exit_code == 1


def test_stdin_input(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, patched_encoder: FakeEncoder
) -> None:
    _patch_clusterer(monkeypatch, patched_encoder)
    result = runner.invoke(app, ["replace"], input="New York\nNYC\n")
    assert result.exit_code == 0
