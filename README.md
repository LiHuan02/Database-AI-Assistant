# Database AI Assistant

基于自建数据库的 AI 问答助手（LangChain + Chroma）。

功能要点：
- 多独立“库”（collection），每个库可单独管理文档与对话
- 支持文档格式：md、PDF、docx、txt
- 对话必须指定库（每次对话绑定一个库），支持多会话管理与轮内历史上下文
- UI 支持快速切换 API Key 与模型

目录结构（骨架）:

- `src/` - 应用源码
	- `app.py` - Streamlit 入口（占位）
	- `core/` - 核心逻辑：库管理、文档 ingest、会话管理、memory
	- `db/` - Chroma 封装与元数据持久化
	- `loaders/` - 文档加载器（md/pdf/docx/txt）
	- `ui/` - Streamlit UI 组件
	- `utils/` - 工具函数
- `.env.example` - 环境变量示例
- `requirements.txt` - 依赖列表

快速开始（仅 Scaffold，未运行）：

1. 建议使用 Python 3.10+
2. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. 复制并填写 `.env`：

```bash
copy .env.example .env
# 编辑 .env 设置 OPENAI_API_KEY 等
```

4. 运行（桌面应用示例）:

```bash
python -m pip install -r requirements.txt
python src/gui/desktop_app.py
```

说明：
- 本仓库提供一个桌面 GUI 骨架（`src/gui/desktop_app.py`），使用 `PySimpleGUI`，替代原有的 web UI。GUI 支持库/对话的创建、对话历史持久化、以及临时模拟的助手回复。向量检索（Chroma）在 `src/db/chroma_client.py` 中有封装，实际调用需要在运行环境中安装并配置 `chromadb`。

数据存放:
- 用户上传的原始文档建议放在 `data/libraries/<library>/docs/`。
- 向量数据库持久化目录建议为 `data/chroma/<library>/`。
- 会话与历史保存在 `data/libraries/<library>/chats/<chat_id>.json`。

后续：查看 `src/` 中的模块占位实现，按需实现 ingest、retrieval、chat 逻辑和完善 UI。 

## 新的流水线与 UI 说明

本项目采用 LCEL 风格对话流水线：
- Summarize：先用 LLM 对当前会话历史生成简短摘要，帮助检索意图；
- Retrieve：根据用户问题与摘要生成 query embedding，检索 Chroma 向量库；
- Answer：将检索到的上下文与历史一起发送给 LLM 得到最终回答。

在桌面应用（`src/gui/desktop_app.py`）和 Streamlit（`src/ui/streamlit_app.py`）中均已接入该流水线。GUI 新增设置：可以分别选择 LLM 提供商与模型（`LLM Provider` / `LLM Model`）以及 Embedding 提供商与模型（`Embed Provider` / `Embed Model`）。

注意：本地 embedding 使用 `sentence-transformers`，首次加载模型会下载权重。

## 向量库与代码结构调整

- 向量库构建与管理已移入 `src/db/vector_store.py`，该模块负责根据 `data/libraries/<lib>/docs/` 构建或删除持久化的 Chroma 向量存储；当文档被删除时，桌面 GUI 会触发重建以保持同步。核心对话逻辑位于 `src/core/`（`pipeline.py`、`llm.py`、`retrieval.py` 等），只负责对话相关行为与检索调用。
- `src/core/ingest.py` 仍负责读取文件、切分与计算 embeddings，但不直接管理向量库生命周期。

## 关于 Runnable 流水线

项目提供了一个轻量的 `SimpleRunnable` 实现（见 `src/core/pipeline.py`），支持使用 `|` 操作符串联步骤：`summarize | retrieve | answer`。如果你已安装并希望使用 LangChain 的原生 `Runnable` API，可将这些步骤替换为 LangChain 的 `Runnable` 实现以获得更多功能与可观察性。

## 打包与发布（Windows 可执行）

推荐使用 `pyinstaller` 将桌面应用打包为单文件 exe：

1. 创建并激活虚拟环境，安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. 使用 `pyinstaller` 打包主程序：

```bash
pyinstaller --onefile --noconsole --add-data "data;data" src/gui/desktop_app.py
```

参数说明：
- `--onefile` 生成单个 exe 文件；
- `--noconsole` 隐藏控制台窗口（GUI 应用）；
- `--add-data "data;data"` 将 `data` 目录一并打包（注意 Windows 路径分隔符）；

3. 打包后，查看 `dist` 目录中的可执行文件，按需把 `data/` 下的大型模型或额外数据与 exe 一起分发。

注意事项：
- 建议在运行时通过界面输入或切换 `API Key`，不要把密钥写入可执行文件中；
- 若使用本地大型模型（transformers/PyTorch），通常不打包入 exe，应在目标机器上单独安装并在运行时加载；
- 如遇到丢失动态库（如 PyTorch），可参考 PyInstaller 文档，使用 `--paths` 或 `--add-binary` 附加库路径。
