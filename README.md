# Database AI Assistant

一个中文知识库问答应用，使用 Streamlit、LangChain Runnable、Chroma 和 OpenAI 兼容模型接口。

## 功能

- 支持多个知识库，每个知识库独立管理文档、对话和 Chroma collection。
- 支持 txt、md、pdf、docx 文档。
- 对话流程固定为：当前问题 + 历史对话生成检索摘要，摘要进入 Chroma 检索，再由检索上下文 + 历史对话 + 当前问题生成最终回答。
- LLM 和向量模型配置相互独立，均支持 provider、model、api_key、base_url。
- Streamlit 主界面支持流式输出、模型下拉选择、手动模型填写和接口设置弹窗。

## 运行

Streamlit 主界面：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run src/app.py
```

桌面端：

```bash
python src/gui/desktop_app.py
```

## 配置

`.env` 中主要配置项：

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=
LLM_BASE_URL=

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_BATCH_SIZE=25

CHROMA_PERSIST_DIRECTORY=./persist/chroma
```

DashScope 等兼容接口通常需要设置独立的 `BASE_URL`，并且 embedding 批量大小不要超过 25。

## 结构

- `src/app.py`：Streamlit 入口。
- `src/ui/streamlit_app.py`：主界面。
- `src/gui/desktop_app.py`：桌面端入口。
- `src/core/pipeline.py`：Runnable RAG 管道和流式输出入口。
- `src/core/llm.py`：LLM 调用封装。
- `src/core/embeddings.py`：向量模型调用封装。
- `src/core/ingest.py`：文档读取、切分和写入。
- `src/db/vector_store.py`：Chroma 同步和检索。
- `src/db/chroma_client.py`：Chroma 客户端封装。
- `src/loaders/`：文档加载器。
- `src/utils/`：配置和日志工具。

## 打包

构建独立可分发的可执行文件（无需安装 Python）：

```bash
pip install pyinstaller
pyinstaller DatabaseAIAssistant.spec --clean --noconfirm
# 输出在 dist/DatabaseAIAssistant/
```

## 发布版本使用

1. 从 [GitHub Releases](https://github.com/LiHuan02/Database-AI-Assistant/releases) 下载对应平台的压缩包
2. 解压到任意目录
3. 复制 `.env.example` 为 `.env` 并填入 API 密钥
4. 运行 `DatabaseAIAssistant.exe` (Windows) 或 `./DatabaseAIAssistant` (Linux/Mac)
5. 浏览器打开 http://localhost:8501

### 开机自启

- **Windows**: 右键 `scripts/install_startup_windows.ps1` → "使用 PowerShell 运行"，或在终端执行：
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/install_startup_windows.ps1
  ```
  禁用：删除开始菜单启动文件夹中的 `DatabaseAIAssistant.lnk` 快捷方式

- **Linux**:
  ```bash
  bash scripts/install_startup_linux.sh
  ```
  禁用：`rm ~/.config/autostart/database-ai-assistant.desktop`

- **macOS**:
  ```bash
  bash scripts/install_startup_macos.sh
  ```
  禁用：`launchctl unload ~/Library/LaunchAgents/com.database-ai-assistant.plist && rm ~/Library/LaunchAgents/com.database-ai-assistant.plist`

## 测试

```bash
.\.venv\Scripts\python.exe -m pytest -q
```
