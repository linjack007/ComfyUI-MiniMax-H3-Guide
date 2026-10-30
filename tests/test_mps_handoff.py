import sys
from types import ModuleType, SimpleNamespace

import pytest

from mps_loader import MiniMaxH3ImageToVideoOffload, align_h3_dimensions


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("requested,expected", [((256, 256), (256, 256)), ((336, 600), (352, 608))])
def test_native_conditioning_handoff_also_cleans_up_on_error(monkeypatch, failed, requested, expected):
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
            node.encode(clip=clip, prompt="draft", width=requested[0], height=requested[1])
    else:
        assert node.encode(clip=clip, prompt="draft", width=requested[0], height=requested[1]) is result
    assert calls == [("native", {"clip": clip, "prompt": "draft", "width": expected[0], "height": expected[1]}), ("offload", clip)]


@pytest.mark.parametrize("width,height", [(0, 600), (336, -1)])
def test_nonpositive_canvas_is_rejected(width, height):
    with pytest.raises(ValueError, match="must be positive"):
        align_h3_dimensions(width, height)


def test_alignment_covers_selector_outputs_and_preserves_compatible_sizes():
    import math

    # Cover all selector aspect ratios, including the failing multiple=24 setting.
    for wr, hr in [(1, 1), (2, 3), (3, 2), (3, 4), (4, 3), (9, 16), (16, 9), (21, 9)]:
        for tenths in range(1, 11):
            scale = math.sqrt(tenths / 10 * 1024 * 1024 / (wr * hr))
            for multiple in (8, 16, 24, 32, 64):
                requested = tuple(round(r * scale / multiple) * multiple for r in (wr, hr))
                aligned = align_h3_dimensions(*requested)
                for before, after in zip(requested, aligned):
                    assert after >= 32 and after % 32 == 0
                    assert abs(after - before) <= 16
                    if before % 32 == 0:
                        assert after == before
