"""Opt-in Apple GPU loading without globally disabling ComfyUI offloading."""

import logging


def align_h3_dimensions(width, height):
    """H3's 16x VAE compression and 2x2 patches require 32-pixel axes."""
    if width <= 0 or height <= 0:
        raise ValueError("MiniMax H3 width and height must be positive.")
    return tuple(max(32, ((value + 16) // 32) * 32) for value in (width, height))


class MiniMaxH3MPSCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {"required": {
            "clip_name": (folder_paths.get_filename_list("text_encoders"),),
        }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "MiniMax H3"
    DESCRIPTION = (
        "Load the MiniMax H3 Qwen encoder on the Apple GPU (MPS). "
        "The generation tail follows this device. CPU offloading stays enabled "
        "so video generation can reuse GPU memory. Requires enough unified "
        "memory for both the encoder and generation tail."
    )

    def load_clip(self, clip_name):
        import torch
        import folder_paths
        import comfy.sd
        import comfy.utils

        if __package__:
            from .mps_ops import mixed_precision_ops
        else:
            from mps_ops import mixed_precision_ops

        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MiniMax H3 Apple GPU loader requires an available PyTorch MPS "
                "device. Use the standard CLIPLoader on other systems."
            )
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        state, _ = comfy.utils.convert_old_quants(state, metadata=metadata)
        quant_config = comfy.utils.detect_layer_quantization(state, "")
        del state
        model_options = {
            "load_device": torch.device("mps"),
            "offload_device": torch.device("cpu"),
        }
        if quant_config is not None:
            model_options["custom_operations"] = mixed_precision_ops(quant_config, torch.float16)
        clip = comfy.sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.MINIMAX,
            model_options=model_options,
        )
        logging.info(
            "MiniMax H3 Qwen: execution=%s, offload=%s",
            clip.patcher.load_device, clip.patcher.offload_device,
        )
        return (clip,)


class MiniMaxH3ImageToVideoOffload:
    """Native H3 conditioning with an explicit CLIP-to-video memory handoff."""

    @classmethod
    def INPUT_TYPES(cls):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

        return MiniMaxH3ImageToVideo.INPUT_TYPES()

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "LATENT")
    FUNCTION = "encode"
    CATEGORY = "MiniMax H3"
    DESCRIPTION = (
        "Round width and height to the nearest multiple of 32 (ties round up) "
        "before native MiniMax H3 image-to-video conditioning, then offload the "
        "connected CLIP before video sampling. Useful for large Apple GPU encoders."
    )

    def encode(self, clip, **kwargs):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        if __package__:
            from .enhancer import _offload_connected_clip
        else:
            from enhancer import _offload_connected_clip

        try:
            requested = (kwargs["width"], kwargs["height"])
            aligned = align_h3_dimensions(*requested)
            if aligned != requested:
                logging.info(
                    "MiniMax H3 canvas aligned: %sx%s -> %sx%s (multiple of 32)",
                    *requested, *aligned,
                )
            kwargs["width"], kwargs["height"] = aligned
            return MiniMaxH3ImageToVideo.execute(clip=clip, **kwargs).result
        finally:
            status = _offload_connected_clip(clip)
            if status != "offloaded":
                logging.warning("MiniMax H3 conditioning CLIP offload: %s", status)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MPSCLIPLoader": MiniMaxH3MPSCLIPLoader,
    "MiniMaxH3ImageToVideoOffload": MiniMaxH3ImageToVideoOffload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MPSCLIPLoader": "MiniMax H3 Qwen Loader (Apple GPU)",
    "MiniMaxH3ImageToVideoOffload": "MiniMax H3 Image to Video (offload CLIP)",
}
