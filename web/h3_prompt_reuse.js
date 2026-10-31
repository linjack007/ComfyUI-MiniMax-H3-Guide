import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "MiniMaxH3.PromptReuse",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxH3ReusablePromptEnhancer") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const widget = (name) => this.widgets?.find((item) => item.name === name);
            const changed = () => {
                this.graph?.change();
                this.setDirtyCanvas(true, true);
            };
            const root = document.createElement("div");
            root.style.cssText = "display:flex;gap:6px;padding:4px;width:100%;box-sizing:border-box";
            const button = (label, action) => {
                const element = document.createElement("button");
                element.textContent = label;
                element.onclick = action;
                element.style.cssText = "flex:1;min-height:32px;border:1px solid #666;border-radius:5px;background:var(--comfy-input-bg,#333);color:var(--input-text,#ddd);cursor:pointer";
                root.append(element);
                return element;
            };
            this.h3ReuseButton = button("复用本次结果", () => {
                if (!this.h3LatestPrompt) return;
                widget("reuse_text").value = this.h3LatestPrompt;
                widget("mode").value = "复用文本";
                widget("enabled").value = true;
                changed();
            });
            this.h3ReuseButton.disabled = true;
            this.h3ReuseButton.title = "先运行一次增强。回填后可编辑 reuse_text，再保存工作流。";
            button("导出提示词.txt", () => {
                const value = !widget("enabled").value ? widget("manual_prompt").value
                    : widget("mode").value === "复用文本" ? widget("reuse_text").value : this.h3LatestPrompt;
                if (!value?.trim()) { alert("请先运行增强，或填写复用文本。"); return; }
                const url = URL.createObjectURL(new Blob([value], { type: "text/plain;charset=utf-8" }));
                const link = document.createElement("a");
                link.href = url;
                link.download = "MiniMax-H3-prompt.txt";
                link.click();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
            });
            button("下次运行重新增强", () => {
                widget("mode").value = "自动生成并缓存";
                widget("enabled").value = true;
                widget("revision").value += 1;
                changed();
            });
            this.addDOMWidget("h3_reuse_actions", "div", root, {
                serialize: false, getMinHeight: () => 42, getMaxHeight: () => 42,
            });
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            if (message.h3_reusable_prompt?.length) {
                this.h3LatestPrompt = message.h3_reusable_prompt[0];
                if (this.h3ReuseButton) this.h3ReuseButton.disabled = false;
            }
        };
    },
});
