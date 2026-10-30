"""Weight-only INT8 execution for the opt-in Apple GPU encoder.

MPS has no torch._int_mm. Comfy's general fallback unrotates every large
ConvRot weight matrix for every token. Rotate the much smaller activation
instead, keeping the published INT8 weights and doing floating-point GEMM.
"""

import logging
import torch


def int8_linear(input, weight, bias=None):
    if __package__:
        from .hybrid_tail import _rotate_activation, _int8_scale_chunk
    else:
        from hybrid_tail import _rotate_activation, _int8_scale_chunk

    params = weight._params
    qdata = weight._qdata
    x = input.float()
    if getattr(params, "convrot", False):
        x = _rotate_activation(x, params.convrot_groupsize)
    # Keep scales and accumulation in FP32, including when the transformer
    # passes BF16 activations. No additional activation quantization is used.
    output = torch.nn.functional.linear(x, qdata.float())
    scale = _int8_scale_chunk(params.scale, 0, qdata.shape[0], qdata.shape[0])
    output = output * scale.reshape(-1)
    if bias is not None:
        output = output + bias.float()
    return output.to(input.dtype)


def mixed_precision_ops(quant_config, compute_dtype):
    import comfy.ops

    native = comfy.ops.mixed_precision_ops(
        quant_config, compute_dtype, full_precision_mm=True
    )

    class MPSOps(native):
        logged_device = False

        class Linear(native.Linear):
            def forward_comfy_cast_weights(self, input, compute_dtype=None,
                                          want_requant=False, weight_only_quant=False):
                # Text encoders force casting; their FP16 storage dtype can
                # differ from BF16 decoding activations. Preserve INT8 until
                # our FP32 GEMM instead of letting that cast unrotate weights.
                layout = getattr(self.weight, "_layout_cls", None)
                layout = layout if isinstance(layout, str) else getattr(layout, "__name__", None)
                if (input.device.type == "mps" and layout == "TensorWiseINT8Layout"
                        and not self.weight_function and not self.bias_function
                        and getattr(self, "weight_lowvram_function", None) is None):
                    with comfy.ops.CastBiasWeightContext(
                        self, input=None, dtype=self.weight.dtype, device=input.device,
                        bias_dtype=input.dtype, offloadable=True,
                        compute_dtype=compute_dtype, want_requant=True,
                    ) as (weight, bias):
                        return self._forward(input, weight, bias)
                return super().forward_comfy_cast_weights(
                    input, compute_dtype, want_requant, weight_only_quant
                )

            def _forward(self, input, weight, bias):
                layout = getattr(weight, "_layout_cls", None)
                layout = layout if isinstance(layout, str) else getattr(layout, "__name__", None)
                if input.device.type == "mps" and layout == "TensorWiseINT8Layout":
                    if not MPSOps.logged_device:
                        logging.info("MiniMax H3 MPS INT8 path: activation=%s, weights=%s",
                                     input.device, weight._qdata.device)
                        MPSOps.logged_device = True
                    return int8_linear(input, weight, bias)
                return super()._forward(input, weight, bias)

    return MPSOps
