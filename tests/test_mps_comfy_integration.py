"""Optional real-Comfy test: COMFYUI_PATH=/path/to/ComfyUI pytest this file."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("COMFYUI_PATH"), reason="Set COMFYUI_PATH for native integration")
@pytest.mark.parametrize("patched", [False, True])
def test_native_cast_offload_and_weight_patch(monkeypatch, patched):
    if not torch.backends.mps.is_available():
        pytest.skip("Apple GPU required")
    monkeypatch.syspath_prepend(os.environ["COMFYUI_PATH"])
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    import comfy.ops
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

    path = Path(__file__).parents[1] / "mps_ops.py"
    spec = importlib.util.spec_from_file_location("_mps_native_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    optimized = module.mixed_precision_ops({}, torch.float16)
    native = comfy.ops.mixed_precision_ops({}, torch.float16, full_precision_mm=True)
    rng = torch.Generator().manual_seed(123)
    q = torch.randint(-127, 128, (64, 64), generator=rng, dtype=torch.int8).to("mps")
    scale = torch.rand(64, 1, generator=rng).to("mps") * 0.01
    params = TensorWiseINT8Layout.Params(
        scale=scale, orig_dtype=torch.float16, orig_shape=q.shape,
        convrot=True, convrot_groupsize=16,
    )
    quant = QuantizedTensor(q, "TensorWiseINT8Layout", params)
    layers = [ops.Linear(64, 64, bias=False, device="mps") for ops in (optimized, native)]
    for layer in layers:
        layer.weight = torch.nn.Parameter(quant, requires_grad=False)
        layer.comfy_force_cast_weights = True
        if patched:
            layer.weight_function = [lambda weight: weight + 0.125]
    x = torch.randn(1, 3, 64, generator=rng).to(device="mps", dtype=torch.bfloat16)
    with torch.inference_mode():
        actual, expected = (layer(x) for layer in layers)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.125)
        layers[0].to("cpu")
        layers[0].to("mps")
        torch.testing.assert_close(layers[0](x), actual, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
