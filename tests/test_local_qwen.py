import asyncio
import io
import json
from pathlib import Path

import pytest

import local_qwen as q


def settings(**changes):
    values = dict(enabled=True, mode=q.MODES[0], profile=q.PROFILES[1], draft="Slow camera push-in.",
                  reuse_text="", model="qwen3.8:27b", endpoint="http://127.0.0.1:11434",
                  max_tokens=384, revision=0)
    values.update(changes)
    return values


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(q.LocalQwenPromptEnhancer, "_cache_directory", lambda self: tmp_path)
    monkeypatch.setattr(q.LocalQwenPromptEnhancer, "_interrupted", staticmethod(lambda: None))
    monkeypatch.setattr(q.LocalQwenPromptEnhancer, "_release_comfy_models", staticmethod(lambda: None))
    return tmp_path


@pytest.mark.parametrize("enabled,mode,text", [(False, q.MODES[0], "Slow camera push-in."),
                                                (True, q.MODES[1], "Edited reusable description.")])
def test_off_and_manual_reuse_do_not_touch_model_or_image(monkeypatch, enabled, mode, text):
    def forbidden(*args):
        pytest.fail("Bypass/reuse must not touch cache, image, model or network")
    monkeypatch.setattr(q.LocalQwenPromptEnhancer, "_cache_directory", forbidden)
    monkeypatch.setattr(q.LocalQwenPromptEnhancer, "_release_comfy_models", forbidden)
    monkeypatch.setattr(q, "image_png", forbidden)
    result = asyncio.run(q.LocalQwenPromptEnhancer().enhance(**settings(
        enabled=enabled, mode=mode, reuse_text="Edited reusable description.", endpoint="offline", image=object())))
    assert result["result"][0] == text
    assert "未调用 Qwen" in result["result"][1]


def test_cache_survives_node_recreation_and_inputs_invalidate(monkeypatch, isolated):
    calls = []
    async def generate(endpoint, payload, interrupted):
        calls.append(payload)
        return {"message": {"content": '{"prompt":"A gentle camera push-in."}'}, "eval_count": 12, "done_reason": "stop"}
    monkeypatch.setattr(q, "chat", generate)
    first = asyncio.run(q.LocalQwenPromptEnhancer().enhance(**settings()))
    again = asyncio.run(q.LocalQwenPromptEnhancer().enhance(**settings()))
    assert first["result"][0] == again["result"][0]
    assert "复用缓存" in again["result"][1]
    assert len(calls) == 1
    for change in [dict(draft="Move left."), dict(model="another-local-model"), dict(revision=1), dict(max_tokens=400)]:
        asyncio.run(q.LocalQwenPromptEnhancer().enhance(**settings(**change)))
    assert len(calls) == 5
    assert len(list(isolated.glob('*.json'))) == 5
    assert not list(isolated.glob('tmp*'))


def test_reference_image_and_target_have_distinct_cache_keys():
    payload = q.request_data(q.PROFILES[0], "draft", "qwen", 384, b"image one")
    key = q.cache_key("http://127.0.0.1:11434", payload, 0)
    for profile, image in [(q.PROFILES[1], b"image one"), (q.PROFILES[0], b"image two"), (q.PROFILES[0], None)]:
        other = q.request_data(profile, "draft", "qwen", 384, image)
        assert q.cache_key("http://127.0.0.1:11434", other, 0) != key
    assert payload["keep_alive"] == 0
    assert payload["think"] is False
    assert "format" not in payload  # Ollama's MLX backend rejects structured-output mode.


@pytest.mark.parametrize("result", [
    {"done_reason": "length", "message": {"content": '{"prompt":"cut off"}'}},
    {"message": {"content": "not json"}},
    {"message": {"content": '{"wrong_field":"text"}'}},
    {"message": {"content": '{"prompt":""}'}},
])
def test_incomplete_results_never_become_reusable_cache(monkeypatch, isolated, result):
    async def generate(*args): return result
    monkeypatch.setattr(q, "chat", generate)
    with pytest.raises(ValueError):
        asyncio.run(q.LocalQwenPromptEnhancer().enhance(**settings()))
    assert not list(isolated.iterdir())


def test_backend_failure_preserves_existing_results(monkeypatch, isolated):
    saved = isolated / "other.json"
    saved.write_text('{"prompt":"saved"}')
    async def generate(*args): raise RuntimeError("offline")
    monkeypatch.setattr(q, "chat", generate)
    with pytest.raises(RuntimeError, match="offline"):
        asyncio.run(q.LocalQwenPromptEnhancer().enhance(**settings()))
    assert saved.read_text() == '{"prompt":"saved"}'
    assert len(list(isolated.iterdir())) == 1


def test_h3_output_always_places_shot_inside_required_section():
    text = q.render_prompt(q.PROFILES[0], json.dumps(dict(description="Camera moves closer.", soundscape="N/A", music="N/A")))
    assert "0.00 seconds" in text
    assert "integrated_multimodal_description:\n[Shot 1] Camera moves closer." in text
    assert "overall_soundscape:\nN/A" in text
    assert text.endswith("non_diegetic_music:\nN/A")
    assert q.render_prompt(q.PROFILES[1], '{"prompt":"Camera moves closer."}') == "Camera moves closer."


@pytest.mark.parametrize("url", ["https://example.com", "http://192.168.1.1:11434", "http://127.0.0.1:11434/path", "http://user@127.0.0.1:11434", "http://127.0.0.1:11434?x=1", "http://localhost:11434"])
def test_only_explicit_loopback_endpoints_are_accepted(url):
    with pytest.raises(ValueError): q.local_endpoint(url)


def test_image_preprocessing_preserves_original():
    import torch
    from PIL import Image
    original = torch.rand(1, 100, 1000, 3)
    clone = original.clone()
    data = q.image_png(original)
    assert Image.open(io.BytesIO(data)).size == (512, 51)
    assert torch.equal(original, clone)


def test_request_cancellation_closes_connection():
    from aiohttp import web
    async def run():
        started = asyncio.Event()
        async def handler(request):
            started.set()
            await asyncio.sleep(2)
            return web.json_response({"message": {"content": "ignored"}})
        app = web.Application()
        app.router.add_post('/api/chat', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        class Interrupted(Exception): pass
        def interrupt():
            if started.is_set(): raise Interrupted()
        try:
            with pytest.raises(Interrupted):
                await q.chat(f'http://127.0.0.1:{port}', {}, interrupt)
        finally:
            await runner.cleanup()
    asyncio.run(run())
