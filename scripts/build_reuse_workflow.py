"""Build an additive upgrade from the existing safe Apple GPU example."""

import copy
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
NAME = "MiniMax-H3-I2V-Enhancer-Apple-GPU-Reuse-Turbo8"
TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def build():
    graph = json.loads((ROOT / "example_workflows/MiniMax-H3-I2V-Enhancer-Apple-GPU.json").read_text())
    graph["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, NAME))
    sub = graph["definitions"]["subgraphs"][0]
    sub["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, NAME + "/video"))
    sub["name"] = "MiniMax H3 · Turbo 8 步首帧视频"
    top = {n["id"]: n for n in graph["nodes"] if n["id"] in (92, 105, 114, 115)}
    inner = {n["id"]: n for n in sub["nodes"]}
    pulled = {i: inner[i] for i in (13, 121, 122, 123, 124, 125)}
    outer = top[105]
    original_values = outer["widgets_values"][:]
    outer["type"] = sub["id"]
    outer["title"] = "③ MiniMax H3 · Turbo 8 步（首帧视频）"
    outer["inputs"][2]["link"] = 241
    outer["inputs"][8] = dict(name="clip", type="CLIP", link=251)
    outer["widgets_values"].pop(6)
    outer["widgets_values"][0] = ""
    outer["size"] = [540, 520]
    sub["inputs"][8].update(name="clip", type="CLIP", linkIds=[189])
    sub["inputs"][0]["linkIds"] = [195]
    sub["nodes"] = [n for n in sub["nodes"] if n["id"] not in pulled]
    sub["links"] = [l for l in sub["links"] if l["origin_id"] not in pulled and l["target_id"] not in pulled]
    sub["links"] += [dict(id=197, origin_id=-10, origin_slot=2, target_id=104, target_slot=4, type="STRING"),
                     dict(id=189, origin_id=-10, origin_slot=8, target_id=104, target_slot=0, type="CLIP")]
    inner[104]["inputs"][4]["link"] = 197
    inner[104]["widgets_values"][0] = ""

    # LoRA and both flow shifts feed the guider and scheduler together.
    for link in sub["links"]:
        if link["origin_id"] == 6: link["origin_id"] = 131
    inner[6]["outputs"][0]["links"] = [250]
    def model_node(id, kind, title, values, incoming, outgoing, pos):
        return dict(id=id, type=kind, title=title, pos=pos, size=[500, 140], flags={}, order=9, mode=0,
                    inputs=[dict(name="model", type="MODEL", link=incoming)],
                    outputs=[dict(name="MODEL", type="MODEL", links=outgoing)],
                    properties={"Node name for S&R": kind}, widgets_values=values)
    sub["nodes"] += [model_node(130, "LoraLoaderModelOnly", "Turbo 8-step LoRA · 强度 1.0", [TURBO, 1.0], 250, [252], [-1700, 4300]),
                     model_node(131, "MiniMaxH3SigmaShift", "Turbo v1.0 · 视频 shift 12 / 音频 shift 3", [12.0, 3.0], 252, [5, 193], [-1100, 4300])]
    sub["links"] += [dict(id=250, origin_id=6, origin_slot=0, target_id=130, target_slot=0, type="MODEL"),
                     dict(id=252, origin_id=130, origin_slot=0, target_id=131, target_slot=0, type="MODEL")]
    inner[9]["widgets_values"] = ["simple", 8, 1.0]
    inner[17]["widgets_values"] = ["euler"]
    sub["state"].update(lastNodeId=131, lastLinkId=252)

    enhancer = pulled[122]
    enhancer["type"] = "MiniMaxH3ReusablePromptEnhancer"
    enhancer["properties"] = {"Node name for S&R": enhancer["type"]}
    enhancer["title"] = "② Qwen 32B · 增强开关 / 自动缓存 / 保存复用"
    enhancer["inputs"][4]["link"] = None
    enhancer["widgets_values"][0] = original_values[0]
    # A time-independent rewrite can be reused when only video duration changes.
    enhancer["widgets_values"][1] = "Recommended mode: I2VA. Single exact first-frame image; no last-frame, reference video or reference audio. Preserve any explicit timing in the draft."
    enhancer["widgets_values"] = [True, "自动生成并缓存", "", 0] + enhancer["widgets_values"]
    enhancer["outputs"][0]["links"] = [241, 234]
    enhancer["size"] = [620, 1100]
    pulled[13]["inputs"] = []
    pulled[13]["widgets_values"] = [original_values[6]]
    pulled[13]["outputs"][0]["links"] = [230, 251]
    pulled[13]["title"] = "Qwen 32B · Apple GPU 编码器（保留原模型）"
    pulled[13]["size"] = [620, 90]
    pulled[121]["size"] = [620, 90]
    pulled[121]["title"] = "Qwen generation tail · 仅增强生成时加载"
    pulled[125]["size"] = [450, 150]
    top[114]["outputs"][0]["links"] = [218, 232]
    top[114]["title"] = "① 首帧图片（原图传给 H3）"
    top[114]["size"] = [450, 560]
    top[92]["widgets_values"][0] = "video/MiniMax_H3_Reuse_Turbo8"
    top[92]["size"] = [540, 580]
    pulled[123].update(title="增强结果 · 可复制 / 回填编辑", size=[620, 320])
    pulled[124].update(title="增强状态 · 缓存 / 耗时 / 格式提醒", size=[620, 240])
    positions = {114:[0,0],115:[0,600],125:[0,820],13:[510,1460],121:[510,1600],122:[510,0],
                 123:[1170,720],124:[1170,1110],105:[1170,0],92:[1760,0]}
    top.update(pulled)
    for id, node in top.items(): node["pos"] = positions[id]
    graph["nodes"] = list(top.values())
    graph["links"] = [l for l in graph["links"] if l[1] in top and l[3] in top]
    graph["links"] += [[232,114,0,125,0,"IMAGE"],[236,125,0,122,3,"IMAGE"],
                       [230,13,0,122,0,"CLIP"],[229,121,0,122,1,"MINIMAX_H3_GENERATION_TAIL"],
                       [251,13,0,105,8,"CLIP"],[241,122,0,105,2,"STRING"],
                       [234,122,0,123,0,"STRING"],[235,122,3,124,0,"STRING"]]
    note = ("## Qwen 32B + Turbo 8 步\n"
            "enabled=false：跳过增强，直接使用 manual_prompt。\n"
            "自动生成并缓存：相同图片、原文、模型和增强设置自动复用；只改视频种子、分辨率、时长不会重写提示词。\n"
            "生成后点击「复用本次结果」，可编辑 reuse_text，再保存工作流；也可导出提示词.txt。\n"
            "换图片或剧情后用自动模式；需要新版本时点「下次运行重新增强」。手动复用不会自动核对新图片。\n"
            "Turbo：8 步、LoRA 1.0、Euler / Simple、视频 shift 12 / 音频 shift 3。不要仅改步数而移除 LoRA。\n"
            "原图用于首帧，增强仅看缩小副本。保留尺寸自动对齐到 32。Qwen 的 H3 条件编码仍需要运行。\n"
            "默认保留 5 秒、0.4 MP；先测试可用 0.7 秒、0.2 MP。画面与音频质量仍需审阅。")
    graph["nodes"].append(dict(id=140,type="MarkdownNote",pos=[0,-350],size=[1700,290],flags={},order=0,mode=0,
                               inputs=[],outputs=[],properties={},widgets_values=[note]))
    graph["groups"] = []
    graph["last_node_id"] = 140; graph["last_link_id"] = 251
    graph["extra"]["ds"] = {"scale": .5, "offset": [100, 400]}
    for node in graph["nodes"] + sub["nodes"]: node.pop("widgets_values_named", None)
    return graph


if __name__ == "__main__":
    target = ROOT / "example_workflows" / (NAME + ".json")
    target.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
    print(target)
