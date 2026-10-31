"""Switchable Qwen enhancement with a persistent, input-addressed text cache."""

import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time

if __package__:
    from .enhancer import MiniMaxH3PromptEnhancer, _resolve_tail_name, resolve_system_prompt
else:
    from enhancer import MiniMaxH3PromptEnhancer, _resolve_tail_name, resolve_system_prompt

MODES = ("自动生成并缓存", "复用文本")
CACHE_VERSION = 1


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino]


def fingerprint(value):
    """Hash image/reference pixels without persisting them in the text cache."""
    import torch

    if isinstance(value, torch.Tensor):
        pixels = value.detach().cpu().contiguous()
        raw = pixels.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"shape": list(pixels.shape), "dtype": str(pixels.dtype),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {key: fingerprint(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported enhancement cache input: {type(value).__name__}")


def cache_key(clip, revision, inputs):
    import folder_paths

    if not hasattr(clip, "minimax_h3_checkpoint"):
        raise ValueError("自动缓存需要连接 MiniMax H3 Qwen Loader (Apple GPU)，以确认实际模型版本。")
    tail = _resolve_tail_name(inputs.get("clip_tail"))
    tail_identity = file_identity(folder_paths.get_full_path_or_raise("text_encoders", tail)) if tail else None
    material = dict(version=CACHE_VERSION, checkpoint=clip.minimax_h3_checkpoint,
                    tail=tail_identity, revision=revision, inputs=fingerprint(inputs))
    return hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_cache(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump({"version": CACHE_VERSION, "result": result}, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class MiniMaxH3ReusablePromptEnhancer(MiniMaxH3PromptEnhancer):
    @classmethod
    def INPUT_TYPES(cls):
        schema = copy.deepcopy(super().INPUT_TYPES())
        schema["required"]["clip"][1]["lazy"] = True
        schema["required"] = {
            "enabled": ("BOOLEAN", {"default": True, "tooltip": "关闭后直接使用 manual_prompt，跳过增强推理。H3 视频仍需要文本编码。"}),
            "mode": (list(MODES), {"tooltip": "自动模式按图片、原文、模型和增强参数复用磁盘缓存；复用文本使用下方保存的内容。"}),
            "reuse_text": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "",
                                      "tooltip": "点击复用本次结果后可编辑；保存工作流会一并保存此文本。"}),
            "revision": ("INT", {"default": 0, "min": 0, "max": 2147483647,
                                  "tooltip": "增加编号会在下次运行时重新增强，旧缓存仍保留。"}),
            **schema["required"],
        }
        return schema

    OUTPUT_NODE = True
    DESCRIPTION = "保留 Qwen 32B/MPS 增强；关闭、手动复用或缓存命中时跳过增强推理。成功结果原子保存到本机 user/minimax_h3/prompt_cache。"

    @classmethod
    def IS_CHANGED(cls, clip_tail=None, **kwargs):
        # Keep Comfy's normal in-memory cache, but notice replaced tail weights.
        import folder_paths
        tail = _resolve_tail_name(clip_tail)
        return file_identity(folder_paths.get_full_path_or_raise("text_encoders", tail)) if tail else None

    def _cache_directory(self):
        import folder_paths
        return Path(folder_paths.get_user_directory()) / "minimax_h3" / "prompt_cache"

    def check_lazy_status(self, enabled, mode, clip=None, **kwargs):
        # Standalone bypass/reuse must not load the 32B encoder just to pass text.
        return ["clip"] if enabled and mode == MODES[0] and clip is None else []

    @staticmethod
    def _output(result, status):
        values = tuple(result[:3]) + (status + "\n" + result[3],)
        return {"ui": {"h3_reusable_prompt": [values[0]], "h3_reusable_report": [values[3]]},
                "result": values}

    def enhance(self, enabled, mode, reuse_text, revision, clip=None, **inputs):
        started = time.perf_counter()
        if not enabled:
            return self._output((inputs["manual_prompt"], inputs["system_prompt"], "", ""),
                                "增强已关闭：直接使用原提示词；未执行 Qwen 增强推理。")
        if mode == MODES[1]:
            if not reuse_text.strip():
                raise ValueError("复用文本为空：请粘贴已保存的提示词，或切换到自动生成并缓存。")
            return self._output((reuse_text, inputs["system_prompt"], "", ""),
                                "复用已编辑文本；未执行 Qwen 增强推理。请确认文本适合当前图片和剧情。")
        if mode != MODES[0]:
            raise ValueError(f"Unknown enhancement mode: {mode}")

        material = dict(inputs)
        material["system_prompt"] = resolve_system_prompt(inputs["system_prompt"])[0]
        key = cache_key(clip, revision, material)
        path = self._cache_directory() / (key + ".json")
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            result = record.get("result")
            if (record.get("version") != CACHE_VERSION or not isinstance(result, list)
                    or len(result) != 4 or any(not isinstance(x, str) for x in result) or not result[0].strip()):
                raise ValueError("提示词缓存不完整；增加 revision 后重新增强，或使用复用文本。")
            return self._output(result, f"复用磁盘缓存 {key[:12]}，{time.perf_counter() - started:.3f}s；未执行 Qwen 增强推理。以下为原始增强报告：")

        result = super().enhance(clip=clip, **inputs)
        if (not result[0].strip() or "Enhancement completed successfully" not in result[3]
                or "Token limit reached" in result[3]):
            raise RuntimeError("增强未完整完成，未保存缓存、未开始视频生成。请检查增强参数或使用复用文本。\n" + result[3])
        try:
            write_cache(path, result)
            status = f"最近一次增强已保存缓存 {key[:12]}；耗时 {time.perf_counter() - started:.3f}s。若 ComfyUI 直接复用内存结果，此处仍显示上次报告。结构警告（如有）保留在下方，请审阅。"
        except OSError as exc:
            logging.warning("MiniMax H3 prompt cache could not be saved: %s", exc)
            status = "增强完成，但缓存写入失败。请点击复用本次结果并保存工作流，或导出文本。"
        return self._output(result, status)


NODE_CLASS_MAPPINGS = {"MiniMaxH3ReusablePromptEnhancer": MiniMaxH3ReusablePromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3ReusablePromptEnhancer": "Qwen 32B · 增强开关 / 保存复用"}
