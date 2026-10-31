import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_saved_upgrade_has_consistent_connections_and_turbo_contract():
    graph = json.loads((ROOT / "example_workflows/MiniMax-H3-I2V-Enhancer-Apple-GPU-Reuse-Turbo8.json").read_text())
    top = {n["id"]: n for n in graph["nodes"]}
    sub = graph["definitions"]["subgraphs"][0]
    inner = {n["id"]: n for n in sub["nodes"]}
    for id, origin, output, target, slot, kind in graph["links"]:
        assert id in top[origin]["outputs"][output]["links"]
        assert top[target]["inputs"][slot]["link"] == id
    for link in sub["links"]:
        if link["origin_id"] >= 0:
            assert link["id"] in inner[link["origin_id"]]["outputs"][link["origin_slot"]]["links"]
        else:
            assert link["id"] in sub["inputs"][link["origin_slot"]]["linkIds"]
        if link["target_id"] >= 0:
            assert inner[link["target_id"]]["inputs"][link["target_slot"]]["link"] == link["id"]
    assert inner[9]["widgets_values"] == ["simple", 8, 1.0]
    assert inner[17]["widgets_values"] == ["euler"]
    assert inner[130]["widgets_values"][1] == 1.0
    assert "8step_v1.0" in inner[130]["widgets_values"][0]
    assert inner[131]["type"] == "MiniMaxH3SigmaShift"
    assert inner[131]["widgets_values"] == [12.0, 3.0]
    assert {l["target_id"] for l in sub["links"] if l["origin_id"] == 131} == {9, 16}
    assert top[122]["widgets_values"][:4] == [True, "自动生成并缓存", "", 0]
    assert top[105]["widgets_values"][3] == 5  # Retain original duration.
    assert top[115]["widgets_values"] == ["1:1 (Square)", 0.4, 32]
    assert top[114]["widgets_values"][0] == "transparent_rgb_gaming_mouse.png"
