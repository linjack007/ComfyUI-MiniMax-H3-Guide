import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "LocalAI.QwenPromptReuse",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LocalQwenPromptEnhancer") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const widget = (name) => this.widgets?.find((item) => item.name === name);
            const root = document.createElement("div");
            root.style.cssText = "display:flex;gap:8px;padding:4px;box-sizing:border-box;width:100%";
            const button = (label) => {
                const element = document.createElement("button");
                element.textContent = label;
                element.style.cssText = "flex:1;min-height:32px;border:1px solid #666;border-radius:5px;background:var(--comfy-input-bg,#333);color:var(--input-text,#ddd);cursor:pointer";
                root.append(element);
                return element;
            };
            this.localQwenReuseButton = button("复用本次结果");
            this.localQwenReuseButton.disabled = true;
            this.localQwenReuseButton.title = "先运行一次增强或读取缓存，然后可回填并编辑文本。";
            this.localQwenReuseButton.onclick = () => {
                if (!this.localQwenResult) return;
                widget("reuse_text").value = this.localQwenResult;
                widget("mode").value = "复用文本";
                widget("enabled").value = true;
                this.graph?.change();
                this.setDirtyCanvas(true, true);
            };
            button("下次运行重新增强").onclick = () => {
                widget("mode").value = "自动生成并缓存";
                widget("enabled").value = true;
                widget("revision").value += 1;
                this.graph?.change();
                this.setDirtyCanvas(true, true);
            };
            this.addDOMWidget("local_qwen_actions", "div", root, {
                serialize: false,
                getMinHeight: () => 42,
                getMaxHeight: () => 42,
            });
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            if (message.local_qwen_prompt?.length) {
                this.localQwenResult = message.local_qwen_prompt[0];
                if (this.localQwenReuseButton) this.localQwenReuseButton.disabled = false;
            }
        };
    },
});
