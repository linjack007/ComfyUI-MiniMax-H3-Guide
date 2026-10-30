import sys
from types import ModuleType, SimpleNamespace

import pytest

from mps_loader import MiniMaxH3ImageToVideoOffload


@pytest.mark.parametrize("failed", [False, True])
def test_native_conditioning_handoff_also_cleans_up_on_error(monkeypatch, failed):
    calls = []
    result = (object(), object())
    clip = object()

    def execute(**kwargs):
        calls.append(("native", kwargs))
        if failed:
            raise ValueError("native encoding failed")
        return SimpleNamespace(result=result)

    native = ModuleType("comfy_extras.nodes_minimax_h3")
    native.MiniMaxH3ImageToVideo = SimpleNamespace(execute=execute)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr("enhancer._offload_connected_clip", lambda value: calls.append(("offload", value)) or "offloaded")
    node = MiniMaxH3ImageToVideoOffload()
    if failed:
        with pytest.raises(ValueError, match="native encoding failed"):
            node.encode(clip=clip, prompt="draft", width=256)
    else:
        assert node.encode(clip=clip, prompt="draft", width=256) is result
    assert calls == [("native", {"clip": clip, "prompt": "draft", "width": 256}), ("offload", clip)]
