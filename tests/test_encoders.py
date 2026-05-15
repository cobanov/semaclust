"""Tests for the SentenceTransformerEncoder device selection."""

from __future__ import annotations

import pytest

from semaclust.encoders import SentenceTransformerEncoder, _pick_device


class _FakeMpsBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(self, *, cuda: bool, mps: bool) -> None:
        self.cuda = _FakeCuda(cuda)

        class _FakeBackends:
            pass

        self.backends = _FakeBackends()
        self.backends.mps = _FakeMpsBackend(mps)


def test_pick_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTorch(cuda=True, mps=True)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    assert _pick_device() == "cuda"


def test_pick_device_picks_mps_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTorch(cuda=False, mps=True)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    assert _pick_device() == "mps"


def test_pick_device_returns_none_when_neither_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeTorch(cuda=False, mps=False)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    assert _pick_device() is None


def test_pick_device_handles_old_torch_without_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    class _OldTorch:
        cuda = _FakeCuda(False)

        class _Backends:
            pass

        backends = _Backends()  # no .mps attribute

    monkeypatch.setitem(__import__("sys").modules, "torch", _OldTorch())
    assert _pick_device() is None


def test_encoder_effective_device_for_explicit_value() -> None:
    enc = SentenceTransformerEncoder("dummy", device="cuda")
    assert enc.effective_device == "cuda"
    assert enc.device == "cuda"


def test_encoder_effective_device_for_none() -> None:
    enc = SentenceTransformerEncoder("dummy", device=None)
    assert enc.effective_device is None
    assert enc.device is None


def test_encoder_effective_device_resolves_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTorch(cuda=False, mps=True)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    enc = SentenceTransformerEncoder("dummy", device="auto")
    # Until accessed, device attribute still shows "auto".
    assert enc.device == "auto"
    assert enc.effective_device == "mps"
    # Second access uses the cached resolution.
    assert enc.effective_device == "mps"


def test_encoder_repr_shows_resolved_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTorch(cuda=True, mps=False)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    enc = SentenceTransformerEncoder("dummy", device="auto")
    assert "device='auto'" in repr(enc)
    # Trigger resolution; repr should now annotate the resolved value.
    _ = enc.effective_device
    assert "device='auto' (-> 'cuda')" in repr(enc)


def test_encoder_default_device_is_auto() -> None:
    enc = SentenceTransformerEncoder("dummy")
    assert enc.device == "auto"
