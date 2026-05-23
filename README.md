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

4. 运行（示例）:

```bash
streamlit run src/app.py
```

后续：查看 `src/` 中的模块占位实现，按需实现 ingest、retrieval、chat 逻辑和 UI。 
