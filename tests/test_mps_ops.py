import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from test_hybrid_tail import _load_tail_module


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("per_row", [False, True])
def test_activation_rotation_matches_dequantized_weight(monkeypatch, device, dtype, rotated, per_row):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple GPU required")
    tail = _load_tail_module(monkeypatch)
    monkeypatch.setitem(sys.modules, "hybrid_tail", tail)
    path = Path(__file__).parents[1] / "mps_ops.py"
    spec = importlib.util.spec_from_file_location("_mps_ops_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = torch.Generator().manual_seed(123)
    q = torch.randint(-127, 128, (37, 64), generator=rng, dtype=torch.int8).to(device)
    scale = (torch.rand((37, 1) if per_row else (), generator=rng) * 0.03).to(device)
    x = torch.randn((2, 3, 64), generator=rng).to(device=device, dtype=dtype)
    bias = torch.randn(37, generator=rng).to(device=device, dtype=dtype)
    weight = SimpleNamespace(_qdata=q, _params=SimpleNamespace(
        scale=scale, convrot=rotated, convrot_groupsize=16,
    ))
    dequantized = q.float() * scale
    if rotated:
        dequantized = tail._rotate_activation(dequantized, 16)
    expected = torch.nn.functional.linear(x.float(), dequantized, bias.float()).to(dtype)
    actual = module.int8_linear(x, weight, bias)
    assert actual.device.type == device
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    # FP32 algebra is equivalent; BF16 permits final rounding at bin edges.
    torch.testing.assert_close(actual, expected, rtol=0.01 if dtype == torch.bfloat16 else 2e-5,
                               atol=0.03 if dtype == torch.bfloat16 else 2e-5)
