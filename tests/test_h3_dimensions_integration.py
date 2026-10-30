"""Native conditioning/patchification regression, without loading model weights."""

import os

import pytest
import torch

from mps_loader import MiniMaxH3ImageToVideoOffload


@pytest.mark.skipif(not os.environ.get("COMFYUI_PATH"), reason="Set COMFYUI_PATH for native integration")
@pytest.mark.parametrize("requested,expected", [
    ((336, 600), (352, 608)),
    ((240, 432), (256, 448)),
    ((600, 336), (608, 352)),
    ((256, 256), (256, 256)),
])
def test_native_first_last_and_output_latents_share_patchable_canvas(monkeypatch, requested, expected):
    monkeypatch.syspath_prepend(os.environ["COMFYUI_PATH"])
    import comfy.model_management
    from comfy.ldm.minimax.model import patchify_video, unpatchify_video

    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("enhancer._offload_connected_clip", lambda clip: "offloaded")
    encoded_sizes = []
    tokenized_sizes = []

    class Clip:
        def tokenize(self, prompt, images):
            tokenized_sizes.extend((im.shape[2], im.shape[1]) for im in images)
            return prompt

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros(1, 1, 1), {}]]

    class Vae:
        def encode(self, image):
            encoded_sizes.append((image.shape[2], image.shape[1]))
            return torch.zeros(1, 24, 1, image.shape[1] // 16, image.shape[2] // 16)

    image = torch.rand(1, 64, 48, 3)
    original = image.clone()
    positive, latent = MiniMaxH3ImageToVideoOffload().encode(
        clip=Clip(), vae=Vae(), prompt="A gaming mouse with softly pulsing lights.",
        width=requested[0], height=requested[1], length=22, first_frame=image, last_frame=image,
    )
    assert encoded_sizes == tokenized_sizes == [expected, expected]
    torch.testing.assert_close(image, original, rtol=0, atol=0)
    video = latent["samples"].unbind()[0]
    references = [item["latent"] for item in positive[0][1]["minimax_keyframes"]]
    for value in [video, *references]:
        assert value.shape[-2:] == (expected[1] // 16, expected[0] // 16)
        rows = patchify_video(value)
        recovered = unpatchify_video(rows, value.shape[2], value.shape[3] // 2, value.shape[4] // 2)
        torch.testing.assert_close(recovered, value)
    if requested == (336, 600):
        # Same 18,648 elements and failing reshape as the reported crash.
        with pytest.raises(RuntimeError, match="invalid for input of size 18648"):
            patchify_video(torch.zeros(1, 24, 1, 600 // 16, 336 // 16))
