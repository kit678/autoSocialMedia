import types
from types import SimpleNamespace

import subprocess

import pytest


class DummyCompleted(SimpleNamespace):
    """Stand-in for subprocess.CompletedProcess"""

    def __init__(self, stdout: str):
        super().__init__(stdout=stdout, returncode=0)


FAKE_FILTERS_OUTPUT = """
Filters:
  V.. vflip             V->V       Flip the input video vertically.
  V.. hflip             V->V       Flip the input video horizontally.
  A.. volume            A->A       Change input volume.
"""


def fake_run(cmd, *args, **kwargs):
    # Only intercept `ffmpeg -filters` invocation.
    if cmd[:1] == ["ffmpeg"] and "-filters" in cmd:
        return DummyCompleted(stdout=FAKE_FILTERS_OUTPUT)
    raise FileNotFoundError("ffmpeg not found")


def test_ffmpeg_filter_exists(monkeypatch):
    # Clear module-level cache before test
    import importlib
    import agent.video.ffmpeg_utils as ffmpeg_utils
    importlib.reload(ffmpeg_utils)  # Reset module state in case of prior tests
    ffmpeg_utils._available_filters = None

    monkeypatch.setattr(subprocess, "run", fake_run)

    from agent.video.ffmpeg_utils import ffmpeg_filter_exists  # after reload

    assert ffmpeg_filter_exists("vflip") is True
    assert ffmpeg_filter_exists("hflip") is True
    assert ffmpeg_filter_exists("volume") is True
    assert ffmpeg_filter_exists("nonexistent") is False

    # Ensure subsequent calls did not trigger additional `subprocess.run` calls
    # by patching fake_run to raise if called again.
    called = {"count": 0}

    def fake_run_count(cmd, *args, **kwargs):
        called["count"] += 1
        return DummyCompleted(stdout=FAKE_FILTERS_OUTPUT)

    monkeypatch.setattr(subprocess, "run", fake_run_count)

    # cache should make run() unnecessary
    assert ffmpeg_filter_exists("vflip") is True
    assert called["count"] == 0

