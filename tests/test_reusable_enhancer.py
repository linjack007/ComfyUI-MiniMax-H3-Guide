import json
import sys
from types import SimpleNamespace

import pytest
import torch

import reusable_enhancer as r


def settings(**changes):
    values = dict(enabled=True, mode=r.MODES[0], reuse_text="", revision=0,
                  clip=SimpleNamespace(minimax_h3_checkpoint=["base", 1024, 100, 1]),
                  manual_prompt="A gaming mouse with a slow camera push-in.", mode_report="I2VA",
                  system_prompt="Describe the pictured product.", max_new_tokens=512,
                  sampling="deterministic", temperature=.7, top_k=64, top_p=.95, min_p=.05,
                  repetition_penalty=1.05, presence_penalty=0, seed=42, thinking=False,
                  offload_after_generation=True)
    values.update(changes)
    return values


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(r.MiniMaxH3ReusablePromptEnhancer, "_cache_directory", lambda self: tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_full_path_or_raise=lambda group, name: str(tmp_path / name)))
    def generate(self, **inputs):
        calls.append(inputs)
        return ("Enhanced product shot.", inputs["system_prompt"], "chat", "Enhancement completed successfully. Structure warning: review alignment.")
    monkeypatch.setattr(r.MiniMaxH3PromptEnhancer, "enhance", generate)
    return calls, tmp_path


@pytest.mark.parametrize("enabled,mode,expected", [(False, r.MODES[0], "draft"), (True, r.MODES[1], "edited")])
def test_bypass_and_manual_reuse_skip_cache_and_generation(monkeypatch, enabled, mode, expected):
    def forbidden(*args, **kwargs):
        pytest.fail("Bypass/manual reuse must not invoke generation or cache")
    monkeypatch.setattr(r, "cache_key", forbidden)
    monkeypatch.setattr(r.MiniMaxH3PromptEnhancer, "enhance", forbidden)
    output = r.MiniMaxH3ReusablePromptEnhancer().enhance(**settings(
        enabled=enabled, mode=mode, manual_prompt="draft", reuse_text="edited", clip=None, image=object()))
    assert output["result"][0] == expected
    assert "未执行 Qwen" in output["result"][3]


def test_cache_survives_node_recreation_and_preserves_warnings(isolated):
    calls, path = isolated
    first = r.MiniMaxH3ReusablePromptEnhancer().enhance(**settings(image=torch.zeros(1, 2, 2, 3)))
    cached = r.MiniMaxH3ReusablePromptEnhancer().enhance(**settings(image=torch.zeros(1, 2, 2, 3)))
    assert len(calls) == 1
    assert cached["result"][:3] == first["result"][:3]
    assert "复用磁盘缓存" in cached["result"][3]
    assert "Structure warning" in cached["result"][3]
    assert cached["ui"]["h3_reusable_prompt"] == [first["result"][0]]
    assert len(list(path.iterdir())) == 1
    saved = next(path.glob("*.json")).read_text()
    assert "shape" not in saved and "sha256" not in saved  # No image pixels or tensors.


@pytest.mark.parametrize("change", [dict(manual_prompt="Another scene"), dict(system_prompt="Another instruction"),
    dict(revision=1), dict(seed=43), dict(max_new_tokens=800), dict(mode_report="Ref2VA"),
    dict(image=torch.ones(1, 2, 2, 3)), dict(thinking=True),
    dict(clip=SimpleNamespace(minimax_h3_checkpoint=["base", 2048, 200, 1]))])
def test_changed_inputs_never_reuse_stale_text(isolated, change):
    calls, path = isolated
    r.MiniMaxH3ReusablePromptEnhancer().enhance(**settings(image=torch.zeros(1, 2, 2, 3)))
    second = settings(image=torch.zeros(1, 2, 2, 3)); second.update(change)
    r.MiniMaxH3ReusablePromptEnhancer().enhance(**second)
    assert len(calls) == 2
    assert len(list(path.glob("*.json"))) == 2


def test_tail_change_invalidates_cache(isolated):
    calls, path = isolated
    tail = path / "tail.safetensors"; tail.write_bytes(b"old")
    values = settings(clip_tail={"tail_name": tail.name})
    r.MiniMaxH3ReusablePromptEnhancer().enhance(**values)
    tail.write_bytes(b"replacement")
    r.MiniMaxH3ReusablePromptEnhancer().enhance(**values)
    assert len(calls) == 2


@pytest.mark.parametrize("report", ["Enhancement fallback: empty", "conditioning only", "Enhancement completed successfully. Token limit reached"])
def test_incomplete_generation_never_caches_or_starts_downstream(monkeypatch, isolated, report):
    calls, path = isolated
    monkeypatch.setattr(r.MiniMaxH3PromptEnhancer, "enhance", lambda *a, **kw: ("text", "system", "chat", report))
    with pytest.raises(RuntimeError, match="未完整完成"):
        r.MiniMaxH3ReusablePromptEnhancer().enhance(**settings())
    assert not list(path.iterdir())


def test_failed_generation_preserves_previous_results(monkeypatch, isolated):
    calls, path = isolated
    r.MiniMaxH3ReusablePromptEnhancer().enhance(**settings())
    saved = next(path.glob("*.json")); before = saved.read_bytes()
    def fail(*a, **kw): raise RuntimeError("generation failed")
    monkeypatch.setattr(r.MiniMaxH3PromptEnhancer, "enhance", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        r.MiniMaxH3ReusablePromptEnhancer().enhance(**settings(revision=2))
    assert saved.read_bytes() == before and len(list(path.iterdir())) == 1


def test_empty_manual_reuse_is_rejected():
    with pytest.raises(ValueError, match="复用文本为空"):
        r.MiniMaxH3ReusablePromptEnhancer().enhance(**settings(mode=r.MODES[1]))


def test_only_automatic_generation_requests_lazy_clip():
    node = r.MiniMaxH3ReusablePromptEnhancer()
    assert node.check_lazy_status(True, r.MODES[0]) == ["clip"]
    assert node.check_lazy_status(True, r.MODES[0], clip=object()) == []
    assert node.check_lazy_status(False, r.MODES[0]) == []
    assert node.check_lazy_status(True, r.MODES[1]) == []


def test_atomic_write_failure_cleans_temporary_and_keeps_old_file(monkeypatch, tmp_path):
    target = tmp_path / "cache.json"; target.write_text("original")
    def fail(*a): raise OSError("disk full")
    monkeypatch.setattr(r.os, "replace", fail)
    with pytest.raises(OSError): r.write_cache(target, ("new", "", "", ""))
    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_reference_tensor_fingerprint_handles_bfloat16():
    value = {"entries": [{"image": torch.zeros(1, 2, 2, 3, dtype=torch.bfloat16), "label": "first"}]}
    hashed = r.fingerprint(value)
    assert hashed["entries"][0]["image"]["dtype"] == "torch.bfloat16"
    json.dumps(hashed)
