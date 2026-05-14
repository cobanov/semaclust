"""Command-line interface for semaclust."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from ._version import __version__
from .clusterer import TextClusterer

app = typer.Typer(
    name="semaclust",
    help="Semantic text clustering using sentence embeddings.",
    no_args_is_help=True,
    add_completion=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Print version and exit.",
        ),
    ] = False,
) -> None:
    """semaclust CLI."""


def _read_texts(path: Path | None) -> list[str]:
    raw = path.read_text(encoding="utf-8") if path is not None else sys.stdin.read()
    return [line.strip() for line in raw.splitlines() if line.strip()]


@app.command()
def cluster(
    input: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to a text file with one item per line. Reads stdin if omitted.",
        ),
    ] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="SentenceTransformer model name.")
    ] = "all-MiniLM-L6-v2",
    threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Agglomerative distance threshold.")
    ] = 0.3,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write JSON output here instead of stdout."),
    ] = None,
) -> None:
    """Cluster lines from a file (or stdin) and emit JSON.

    Output shape: ``{"clusters": {id: [texts]}, "representatives": {id: text}}``.
    """
    texts = _read_texts(input)
    if not texts:
        typer.secho("No input texts provided.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    clusterer = TextClusterer(encoder=model, distance_threshold=threshold)
    clusterer.fit(texts)

    payload = {
        "n_clusters": clusterer.n_clusters_,
        "clusters": {str(k): v for k, v in clusterer.clusters_.items()},
        "representatives": {str(k): v for k, v in clusterer.representatives_.items()},
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if output is not None:
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        typer.echo(rendered)


@app.command()
def replace(
    input: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Input file. Reads stdin if omitted.",
        ),
    ] = None,
    model: Annotated[str, typer.Option("--model", "-m")] = "all-MiniLM-L6-v2",
    threshold: Annotated[float, typer.Option("--threshold", "-t")] = 0.3,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Replace each input line with its cluster representative."""
    texts = _read_texts(input)
    if not texts:
        typer.secho("No input texts provided.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    clusterer = TextClusterer(encoder=model, distance_threshold=threshold)
    replaced = clusterer.fit_transform(texts)
    rendered = "\n".join(replaced) + "\n"
    if output is not None:
        output.write_text(rendered, encoding="utf-8")
    else:
        typer.echo(rendered, nl=False)


if __name__ == "__main__":
    app()
