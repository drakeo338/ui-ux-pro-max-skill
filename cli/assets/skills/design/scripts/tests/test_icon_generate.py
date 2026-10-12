"""Offline regression tests for icon/generate.py (no network, mocked genai)."""

import importlib.util
import sys
import types as pytypes
from pathlib import Path
from unittest import mock

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "icon" / "generate.py"


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def gen(monkeypatch):
    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    types = pytypes.ModuleType("google.genai.types")
    types.GenerateContentConfig = _Config
    genai.Client = mock.MagicMock()
    genai.types = types
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types)
    spec = importlib.util.spec_from_file_location("icon_generate_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.GEMINI_API_KEY = "test-key"
    return mod


def _response(text, finish_reason=None):
    cand = mock.Mock()
    cand.finish_reason = finish_reason
    cand.content = None
    return mock.Mock(text=text, candidates=[cand])


def test_viewbox_size_only_touches_root_tag(gen):
    src = '<svg viewBox="0 0 24 24" stroke-width="2"><rect width="10" height="8"/></svg>'
    out = gen.apply_viewbox_size(src, 48)
    assert 'stroke-width="2"' in out
    assert '<rect width="10" height="8"/>' in out
    assert 'viewBox="0 0 24 24"' in out
    root = out[: out.index(">") + 1]
    assert 'width="48"' in root and 'height="48"' in root


def test_viewbox_size_handles_width_and_height_independently(gen):
    out = gen.apply_viewbox_size("<svg width='10' viewBox='0 0 24 24'><path d='M0 0'/></svg>", 32)
    root = out[: out.index(">") + 1]
    assert "width=\"32\"" in root and "height=\"32\"" in root
    assert "width='10'" not in root


def test_generate_icon_uses_large_token_budget(gen, tmp_path):
    client = gen.genai.Client.return_value
    client.models.generate_content.return_value = _response(
        '<svg viewBox="0 0 24 24"><path d="M0 0"/></svg>')
    out = tmp_path / "a.svg"
    assert gen.generate_icon("crane", output_path=str(out)) == str(out)
    cfg = client.models.generate_content.call_args.kwargs["config"]
    assert cfg.max_output_tokens >= 16384


def test_generate_icon_reports_max_tokens(gen, capsys):
    client = gen.genai.Client.return_value
    client.models.generate_content.return_value = _response(
        '```svg\n<svg viewBox="0 0 24 24"><path d="M 15 14 H 19', "MAX_TOKENS")
    assert gen.generate_icon("crane") is None
    assert "MAX_TOKENS" in capsys.readouterr().out
    assert client.models.generate_content.call_count == 1


def test_generate_icon_reports_empty_output_separately(gen, capsys):
    client = gen.genai.Client.return_value
    client.models.generate_content.return_value = _response("", "STOP")
    assert gen.generate_icon("crane") is None
    out = capsys.readouterr().out
    assert "empty response" in out and "MAX_TOKENS" not in out
