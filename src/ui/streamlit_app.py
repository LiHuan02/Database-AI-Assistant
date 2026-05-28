import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core.chat_manager import ChatManager
from core.library_manager import LibraryManager
from core.pipeline import stream_lcel_pipeline
from db.chroma_client import clear_chroma_system_cache
from db.vector_store import delete_vector_store, sync_vector_store
from utils.config import (
    get_embedding_config,
    get_library_config,
    get_llm_config,
    LibraryConfig,
    save_library_config,
)

DATA_ROOT = "data"

LLM_PROVIDERS = [
    "dashscope", "openai", "deepseek", "siliconflow",
    "zhipu", "moonshot", "openrouter", "compatible", "local", "手动填写",
]
EMBEDDING_PROVIDERS = [
    "dashscope", "openai", "siliconflow", "zhipu",
    "compatible", "local", "hash", "手动填写",
]
LLM_MODELS = {
    "dashscope": ["qwen-plus", "qwen-turbo", "qwen-max", "glm-5", "手动填写"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "手动填写"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner", "手动填写"],
    "siliconflow": ["Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3", "手动填写"],
    "zhipu": ["glm-4-plus", "glm-4-flash", "glm-5", "手动填写"],
    "moonshot": ["moonshot-v1-8k", "moonshot-v1-32k", "手动填写"],
    "openrouter": ["openai/gpt-4o-mini", "deepseek/deepseek-chat", "手动填写"],
}
EMBEDDING_MODELS = {
    "dashscope": ["text-embedding-v2", "text-embedding-v3", "手动填写"],
    "openai": ["text-embedding-3-small", "text-embedding-3-large", "手动填写"],
    "siliconflow": ["BAAI/bge-m3", "netease-youdao/bce-embedding-base_v1", "手动填写"],
    "zhipu": ["embedding-3", "手动填写"],
    "local": ["all-MiniLM-L6-v2", "BAAI/bge-small-zh-v1.5", "手动填写"],
    "hash": ["hash", "手动填写"],
}


def _handle_delete_request(library_manager: LibraryManager):
    """Perform library deletion BEFORE any widget renders.

    This function is called at the top of main().  It only touches
    session-state keys that are NOT bound to any widget, so there is
    no risk of the "cannot be modified after instantiation" error.
    """
    target = st.session_state.pop("_delete_target", None)
    if not target:
        return

    import gc, time
    try:
        delete_vector_store(target, data_root=DATA_ROOT)
    except Exception:
        pass
    clear_chroma_system_cache()
    gc.collect()
    time.sleep(0.3)
    for _ in range(4):
        if not library_manager.delete_library(target):
            gc.collect()
            time.sleep(0.8)
        else:
            break


def main():
    st.set_page_config(
        page_title="Database AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    library_manager = LibraryManager(data_root=DATA_ROOT)
    library_manager.create_library("base")
    chat_manager = ChatManager(data_root=DATA_ROOT)

    llm_config = get_llm_config()
    embedding_config = get_embedding_config()
    _init_state(llm_config, embedding_config)

    # ── Handle pending library deletion BEFORE any widget renders ──
    _handle_delete_request(library_manager)

    with st.sidebar:
        _render_sidebar_header()
        selected_library = _library_panel(library_manager)
        embedding_settings = _embedding_panel(selected_library)
        _library_config_panel(selected_library)
        _document_panel(selected_library, embedding_settings)

    llm_settings = _chat_header()
    chat_id = _chat_selector(chat_manager, selected_library)
    _render_history(chat_manager, selected_library, chat_id)

    if prompt := st.chat_input("输入问题，Enter 发送"):
        _handle_chat_submit(
            chat_manager, selected_library, chat_id, prompt,
            llm_settings, embedding_settings,
        )


# ── session state ──────────────────────────────────────────────

def _init_state(llm_config, embedding_config):
    defaults = {
        "llm_provider": llm_config.provider,
        "llm_model": llm_config.model,
        "llm_api_key": llm_config.api_key or "",
        "llm_base_url": llm_config.base_url or "",
        "embedding_provider": embedding_config.provider,
        "embedding_model": embedding_config.model,
        "embedding_api_key": embedding_config.api_key or "",
        "embedding_base_url": embedding_config.base_url or "",
        "_processed_upload": "",
        "_chat_search_results": None,
        "_chat_export_data": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ── sidebar header ─────────────────────────────────────────────

def _render_sidebar_header():
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-logo">🤖</div>
          <div>
            <div class="sidebar-title">Database AI Assistant</div>
            <div class="sidebar-subtitle">智能知识库问答</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── library panel ──────────────────────────────────────────────

def _library_panel(library_manager: LibraryManager) -> str:
    libraries = [
        item.get("name") for item in library_manager.list_libraries()
    ]
    if not libraries:
        library_manager.create_library("base")
        libraries = ["base"]

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-section-header">📚 知识库</div>',
        unsafe_allow_html=True,
    )
    selected = st.selectbox(
        "当前知识库", libraries,
        label_visibility="collapsed", key="library_select",
    )

    with st.expander("➕ 新建知识库"):
        name = st.text_input(
            "名称", key="new_library_name",
            placeholder="输入知识库名称…",
        )
        if st.button("创建知识库", use_container_width=True):
            if not name.strip():
                st.error("知识库名称不能为空")
            else:
                library_manager.create_library(name.strip())
                st.rerun()

    with st.popover("🗑 删除知识库", use_container_width=True):
        st.warning(f"确定要删除「{selected}」吗？所有文档、向量和对话将永久删除。")
        _, col_confirm = st.columns(2)
        with col_confirm:
            if st.button("确认删除", type="primary", use_container_width=True):
                st.session_state["_delete_target"] = selected
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    return selected


# ── embedding panel ────────────────────────────────────────────

def _embedding_panel(library_name: str) -> dict:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-section-header">🧬 向量模型</div>',
        unsafe_allow_html=True,
    )
    provider = _provider_select("embedding_provider", EMBEDDING_PROVIDERS)
    model_options = EMBEDDING_MODELS.get(provider, ["手动填写"])
    model = _model_select("embedding_model", model_options)

    with st.popover("⚙ 接口设置", use_container_width=True):
        api_key = st.text_input(
            "API Key",
            value=st.session_state.embedding_api_key,
            type="password",
            key="embedding_api_key_input",
        )
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.embedding_base_url,
            key="embedding_base_url_input",
        )
        if st.button("保存向量设置", use_container_width=True):
            st.session_state.embedding_api_key = api_key
            st.session_state.embedding_base_url = base_url
            st.success("设置已保存到当前会话")

    if st.button("🔄 同步向量库", use_container_width=True):
        with st.spinner("正在同步向量库…"):
            try:
                result = sync_vector_store(
                    library_name,
                    data_root=DATA_ROOT,
                    api_key=st.session_state.embedding_api_key or None,
                    embedding_provider=provider,
                    embedding_model=model,
                    embedding_base_url=st.session_state.embedding_base_url or None,
                )
                if result.get("synced"):
                    inserted = result.get("inserted") or 0
                    deleted = result.get("deleted") or 0
                    parts = []
                    if inserted:
                        parts.append(f"写入 {inserted} 个文本块")
                    if deleted:
                        parts.append(f"移除 {deleted} 篇文档")
                    st.success("同步完成，" + "，".join(parts))
                else:
                    st.info("向量库已是最新，无需同步")
            except Exception as exc:
                st.error(f"同步失败：{exc}")
    st.markdown("</div>", unsafe_allow_html=True)
    return {
        "provider": provider,
        "model": model,
        "api_key": st.session_state.embedding_api_key or None,
        "base_url": st.session_state.embedding_base_url or None,
    }


# ── library config panel ───────────────────────────────────────

def _library_config_panel(library_name: str):
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-section-header">⚙ 知识库参数</div>',
        unsafe_allow_html=True,
    )
    lib_cfg = get_library_config(library_name, data_root=DATA_ROOT)

    with st.expander("分块 & 检索参数"):
        chunk_size = st.number_input(
            "chunk_size",
            min_value=200, max_value=4000, value=lib_cfg.chunk_size, step=100,
            help="文本分块大小（字符数）",
        )
        chunk_overlap = st.number_input(
            "chunk_overlap",
            min_value=0, max_value=1000, value=lib_cfg.chunk_overlap, step=50,
            help="相邻分块重叠字符数",
        )
        top_k = st.number_input(
            "top_k",
            min_value=1, max_value=20, value=lib_cfg.top_k, step=1,
            help="检索返回的最大结果数",
        )
        if st.button("保存参数", use_container_width=True):
            new_cfg = LibraryConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
            )
            save_library_config(library_name, new_cfg, data_root=DATA_ROOT)
            st.success("参数已保存（需重新同步才生效）")
    st.markdown("</div>", unsafe_allow_html=True)


# ── document panel ─────────────────────────────────────────────

def _document_panel(library_name: str, embedding_settings: dict):
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-section-header">📄 文档</div>',
        unsafe_allow_html=True,
    )
    docs_dir = Path(DATA_ROOT) / "libraries" / library_name / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    uploaded = st.file_uploader(
        "上传文档",
        type=["txt", "md", "pdf", "docx"],
        label_visibility="collapsed",
        key=f"doc_upload_{library_name}",
    )
    if uploaded is not None:
        upload_id = f"{uploaded.name}:{uploaded.size}"
        already_processed = st.session_state.get("_processed_upload") == upload_id
        if not already_processed:
            try:
                dst = docs_dir / uploaded.name
                dst.write_bytes(uploaded.read())
                with st.spinner("正在同步向量库…"):
                    sync_vector_store(
                        library_name,
                        data_root=DATA_ROOT,
                        api_key=embedding_settings["api_key"],
                        embedding_provider=embedding_settings["provider"],
                        embedding_model=embedding_settings["model"],
                        embedding_base_url=embedding_settings["base_url"],
                    )
                st.session_state["_processed_upload"] = upload_id
                st.success(f"已上传并同步 `{uploaded.name}`")
                st.rerun()
            except Exception as exc:
                st.error(f"上传失败：{exc}")

    docs = sorted(
        [item for item in docs_dir.iterdir() if item.is_file()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not docs:
        st.caption("暂无文档，上传后自动同步")
    else:
        st.caption(f"共 {len(docs)} 个文档")
    for doc in docs:
        _render_doc_item(doc, library_name, embedding_settings)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_doc_item(doc: Path, library_name: str, embedding_settings: dict):
    suffix = doc.suffix.lower()
    icon_map = {".pdf": "📕", ".docx": "📘", ".md": "📝", ".txt": "📄"}
    icon = icon_map.get(suffix, "📎")
    col_name, col_btn = st.columns([0.82, 0.18], vertical_alignment="center")
    col_name.markdown(
        f'<span class="doc-item">{icon} {doc.name}</span>',
        unsafe_allow_html=True,
    )
    if col_btn.button("🗑", key=f"del-{library_name}-{doc.name}", help=f"删除 {doc.name}"):
        try:
            doc.unlink()
            with st.spinner("正在更新向量库…"):
                sync_vector_store(
                    library_name,
                    data_root=DATA_ROOT,
                    api_key=embedding_settings["api_key"],
                    embedding_provider=embedding_settings["provider"],
                    embedding_model=embedding_settings["model"],
                    embedding_base_url=embedding_settings["base_url"],
                )
            st.session_state["_processed_upload"] = ""
            st.rerun()
        except Exception as exc:
            st.error(f"删除失败：{exc}")


# ── chat header ────────────────────────────────────────────────

def _chat_header() -> dict:
    col_title, col_model = st.columns([0.52, 0.48], vertical_alignment="center")
    with col_title:
        st.markdown(
            '<h1 class="main-title">🤖 Database AI Assistant</h1>',
            unsafe_allow_html=True,
        )
    with col_model:
        col_prov, col_m, col_set = st.columns([0.34, 0.38, 0.28], vertical_alignment="center")
        with col_prov:
            provider = _provider_select(
                "llm_provider", LLM_PROVIDERS, label="厂商",
            )
        with col_m:
            model = _model_select(
                "llm_model",
                LLM_MODELS.get(provider, ["手动填写"]),
                label="模型",
            )
        with col_set:
            with st.popover("⚙ 设置", use_container_width=True):
                api_key = st.text_input(
                    "API Key",
                    value=st.session_state.llm_api_key,
                    type="password",
                    key="llm_api_key_input",
                )
                base_url = st.text_input(
                    "Base URL",
                    value=st.session_state.llm_base_url,
                    key="llm_base_url_input",
                )
                if st.button("保存 LLM 设置", use_container_width=True):
                    st.session_state.llm_api_key = api_key
                    st.session_state.llm_base_url = base_url
                    st.success("LLM 设置已保存到当前会话")
    return {
        "provider": provider,
        "model": model,
        "api_key": st.session_state.llm_api_key or None,
        "base_url": st.session_state.llm_base_url or None,
    }


# ── chat selector ─────────────────────────────────────────────

def _chat_selector(
    chat_manager: ChatManager, library_name: str,
) -> str:
    chats = chat_manager.list_chats(library_name)
    if not chats:
        return chat_manager.create_chat(library_name, "新对话")

    chat_map = {
        f"{item.get('title') or item.get('id')}": item.get("id")
        for item in chats
    }
    titles = list(chat_map.keys())
    active_id = st.session_state.get("active_chat_id")
    current_index = 0
    if active_id and active_id in chat_map.values():
        for i, cid in enumerate(chat_map.values()):
            if cid == active_id:
                current_index = i
                break

    col_select, col_new, col_del = st.columns([0.52, 0.24, 0.24], vertical_alignment="center")
    with col_select:
        selected_title = st.selectbox(
            "对话", titles,
            index=current_index,
            label_visibility="collapsed",
            key="chat_select",
        )
    with col_new:
        if st.button("➕ 新", use_container_width=True, help="新建对话"):
            new_id = chat_manager.create_chat(library_name, "新对话")
            st.session_state.active_chat_id = new_id
            st.rerun()
    with col_del:
        if st.button("🗑 删", use_container_width=True, help="删除当前对话"):
            chat_manager.delete_chat(library_name, chat_map[selected_title])
            st.session_state.pop("active_chat_id", None)
            st.rerun()

    chat_id = st.session_state.get("active_chat_id") or chat_map[selected_title]

    # ── chat actions row ─────────────────────────
    col_rename, col_search, col_export = st.columns([0.37, 0.33, 0.30])
    with col_rename:
        with st.popover("✏ 重命名", use_container_width=True):
            cur_title = next(
                (t for t, cid in chat_map.items() if cid == chat_id),
                "新对话",
            )
            new_title = st.text_input(
                "对话标题", value=cur_title,
                key=f"rename_{chat_id}",
            )
            if st.button("确认重命名", use_container_width=True):
                chat_manager.rename_chat(library_name, chat_id, new_title)
                st.rerun()

    with col_search:
        with st.popover("🔍 搜索", use_container_width=True):
            q = st.text_input(
                "搜索对话内容",
                key=f"chat_search_q_{library_name}",
                placeholder="输入关键词…",
            )
            if q and st.button("搜索", use_container_width=True):
                results = chat_manager.search_chats(library_name, q)
                if results:
                    for r in results:
                        st.markdown(
                            f"**{r['title']}** · {r['role']}\n>{r['snippet']}",
                        )
                else:
                    st.caption("未找到匹配内容")

    with col_export:
        if st.button("📥 导出", use_container_width=True, help="导出对话为 Markdown"):
            md = chat_manager.export_chat_markdown(library_name, chat_id)
            st.session_state["_chat_export_data"] = md
            st.session_state["_chat_export_filename"] = (
                f"{library_name}_{chat_id[:8]}.md"
            )
            st.rerun()

    # Download outside of button context
    if st.session_state.get("_chat_export_data"):
        st.download_button(
            label="⬇ 下载 Markdown",
            data=st.session_state["_chat_export_data"],
            file_name=st.session_state.get("_chat_export_filename", "chat.md"),
            mime="text/markdown",
            use_container_width=True,
            key="_export_dl",
        )
        if st.button("关闭", use_container_width=True, key="_close_export"):
            st.session_state["_chat_export_data"] = None
            st.rerun()

    return chat_id


# ── history renderer ──────────────────────────────────────────

def _render_history(
    chat_manager: ChatManager, library_name: str, chat_id: str,
):
    messages = chat_manager.get_messages(library_name, chat_id)
    if not messages:
        st.markdown(
            """
            <div class="empty-chat">
              <div class="empty-chat-icon">💬</div>
              <div class="empty-chat-text">
                开始提问，AI 将基于知识库内容回答<br>
                <span style="font-size:.78rem;color:#9ca3af;">
                  复杂问题将自动拆分为多个检索查询
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    for message in messages:
        _render_message(message.get("role", "assistant"), message.get("text", ""))


# ── chat submit handler ────────────────────────────────────────

def _handle_chat_submit(
    chat_manager, library_name, chat_id, user_input, llm_settings, embedding_settings,
):
    _render_message("user", user_input)
    placeholder = st.empty()
    chunks: list[str] = []

    def token_stream():
        try:
            for chunk in stream_lcel_pipeline(
                library_name,
                chat_id,
                user_input,
                chat_manager,
                data_root=DATA_ROOT,
                api_key=llm_settings["api_key"],
                base_url=llm_settings["base_url"],
                embedding_api_key=embedding_settings["api_key"],
                embedding_base_url=embedding_settings["base_url"],
                embedding_provider=embedding_settings["provider"],
                embedding_model=embedding_settings["model"],
                llm_provider=llm_settings["provider"],
                llm_model=llm_settings["model"],
            ):
                chunks.append(chunk)
                yield chunk
        except Exception as exc:
            message = f"\n\n处理失败：{exc}"
            chunks.append(message)
            yield message

    with placeholder.container():
        with st.chat_message("assistant"):
            st.write_stream(token_stream())

    answer = "".join(chunks).strip()
    if answer:
        messages = chat_manager.get_messages(library_name, chat_id)
        if not messages:
            auto_title = user_input.strip()[:30]
            chat_manager.rename_chat(library_name, chat_id, auto_title)
        chat_manager.append_user_message(library_name, chat_id, user_input)
        chat_manager.append_assistant_message(library_name, chat_id, answer)
    st.rerun()


# ── provider / model select ───────────────────────────────────

def _provider_select(
    state_key: str, options: list[str], label: str = "厂商",
) -> str:
    current = st.session_state.get(state_key) or options[0]
    index = options.index(current) if current in options else len(options) - 1
    selected = st.selectbox(
        label, options, index=index, key=f"{state_key}_select",
    )
    if selected == "手动填写":
        manual = st.text_input(
            "手动填写厂商",
            value=current if current not in options else "",
            key=f"{state_key}_manual",
        )
        selected = manual or selected
    st.session_state[state_key] = selected
    return selected


def _model_select(
    state_key: str, options: list[str], label: str = "模型",
) -> str:
    current = st.session_state.get(state_key) or options[0]
    index = options.index(current) if current in options else len(options) - 1
    selected = st.selectbox(
        label, options, index=index, key=f"{state_key}_select",
    )
    if selected == "手动填写":
        manual = st.text_input(
            "手动填写模型",
            value=current if current not in options else "",
            key=f"{state_key}_manual",
        )
        selected = manual or selected
    st.session_state[state_key] = selected
    return selected


# ── message bubble ─────────────────────────────────────────────


def _render_message(role: str, text: str):
    if role == "user":
        display = html.escape(text).replace("\n", "<br>")
    else:
        import markdown as md
        display = md.markdown(text, extensions=["fenced_code", "tables"])
    klass = "msg-user" if role == "user" else "msg-assistant"
    avatar = "👤" if role == "user" else "🤖"
    name = "你" if role == "user" else "AI 助手"
    st.markdown(
        f"""
        <div class="msg-row {klass}">
          <div class="msg-avatar">{avatar}</div>
          <div class="msg-body">
            <div class="msg-name">{name}</div>
            <div class="msg-text">{display}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── CSS ────────────────────────────────────────────────────────

def _inject_css():
    st.markdown(
        """
        <style>
        /* ── global ─────────────────────────────── */
        :root {
          --bg: #f8f9fb;
          --surface: #ffffff;
          --border: #e5e7ec;
          --text: #1f2937;
          --text-secondary: #6b7280;
          --accent: #4f46e5;
          --accent-light: #eef2ff;
          --radius: 12px;
          --shadow-sm: 0 1px 2px rgba(0,0,0,.04);
          --shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
        }

        .stApp {
          background: var(--bg);
        }

        .block-container {
          max-width: 1200px;
          padding-top: 1.2rem;
          padding-bottom: 8rem;
        }

        /* ── sidebar ───────────────────────────── */
        section[data-testid="stSidebar"] {
          border-right: 1px solid var(--border);
          background: var(--surface);
        }
        section[data-testid="stSidebar"] .block-container {
          padding: .8rem 1rem;
        }

        .sidebar-brand {
          display: flex;
          align-items: center;
          gap: .7rem;
          padding: .6rem .4rem 1rem;
        }
        .sidebar-logo {
          font-size: 2rem;
          line-height: 1;
        }
        .sidebar-title {
          font-size: 1.05rem;
          font-weight: 700;
          color: var(--text);
          line-height: 1.25;
        }
        .sidebar-subtitle {
          font-size: .78rem;
          color: var(--text-secondary);
        }

        .sidebar-section {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: .7rem .85rem;
          margin-bottom: .8rem;
          box-shadow: var(--shadow-sm);
        }
        .sidebar-section-header {
          font-size: .82rem;
          font-weight: 650;
          color: var(--text);
          margin-bottom: .5rem;
        }

        .doc-item {
          font-size: .78rem;
          color: var(--text-secondary);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          display: block;
        }

        /* ── main header ──────────────────────── */
        .main-title {
          font-size: 1.4rem !important;
          font-weight: 700 !important;
          color: var(--text) !important;
          margin: 0 !important;
          padding: 0 !important;
        }

        /* ── chat frame ───────────────────────── */
        .empty-chat {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 4rem 1rem;
          text-align: center;
        }
        .empty-chat-icon {
          font-size: 3rem;
          margin-bottom: .8rem;
        }
        .empty-chat-text {
          font-size: .92rem;
          color: var(--text-secondary);
        }

        /* ── message bubbles ──────────────────── */
        .msg-row {
          display: flex;
          align-items: flex-start;
          gap: .65rem;
          margin: .85rem 0;
        }
        .msg-row.msg-assistant {
          flex-direction: row;
        }
        .msg-row.msg-user {
          flex-direction: row-reverse;
        }
        .msg-avatar {
          width: 34px;
          height: 34px;
          min-width: 34px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1.1rem;
          line-height: 1;
          background: var(--surface);
          border: 1px solid var(--border);
          box-shadow: var(--shadow-sm);
        }

        .msg-body {
          max-width: min(760px, 76%);
        }
        .msg-name {
          font-size: .72rem;
          color: var(--text-secondary);
          margin-bottom: .22rem;
          padding: 0 .2rem;
        }
        .msg-row.msg-user .msg-name {
          text-align: right;
        }
        .msg-text {
          padding: .72rem .9rem;
          border-radius: 16px;
          line-height: 1.68;
          font-size: .94rem;
          box-shadow: var(--shadow);
          word-break: break-word;
          overflow-wrap: break-word;
        }
        .msg-row.msg-assistant .msg-text {
          background: var(--surface);
          color: var(--text);
          border: 1px solid var(--border);
          border-top-left-radius: 4px;
        }
        .msg-row.msg-user .msg-text {
          background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
          color: #fff;
          border-top-right-radius: 4px;
        }

        /* ── chat input ───────────────────────── */
        div[data-testid="stChatInput"] {
          max-width: 820px;
          margin: 0 auto;
          padding-bottom: 1rem;
        }
        div[data-testid="stChatInput"] textarea {
          border-radius: 14px !important;
          border: 1px solid var(--border) !important;
          box-shadow: var(--shadow) !important;
          padding: .65rem .85rem !important;
        }
        div[data-testid="stChatInput"] textarea:focus {
          border-color: var(--accent) !important;
          box-shadow: 0 0 0 3px rgba(79,70,229,.12) !important;
        }

        /* ── buttons ──────────────────────────── */
        .stButton > button {
          border-radius: 8px !important;
          font-weight: 550 !important;
          font-size: .8rem !important;
          transition: all .15s ease;
          border: 1px solid var(--border) !important;
          background: var(--surface) !important;
          color: var(--text) !important;
          box-shadow: var(--shadow-sm);
        }
        .stButton > button:hover {
          background: var(--accent-light) !important;
          border-color: var(--accent) !important;
          color: var(--accent) !important;
        }

        /* ── selectbox ────────────────────────── */
        .stSelectbox > div > div {
          border-radius: 8px !important;
          border-color: var(--border) !important;
        }

        /* ── text input ───────────────────────── */
        .stTextInput > div > div > input {
          border-radius: 8px !important;
          border-color: var(--border) !important;
        }
        .stTextInput > div > div > input:focus {
          border-color: var(--accent) !important;
          box-shadow: 0 0 0 3px rgba(79,70,229,.1) !important;
        }

        /* ── expander ─────────────────────────── */
        .streamlit-expanderHeader {
          border-radius: 8px !important;
          font-size: .8rem !important;
        }

        /* ── file uploader ────────────────────── */
        section[data-testid="stFileUploadDropzone"] {
          border-radius: 10px !important;
          border: 1.5px dashed var(--border) !important;
          background: #fafbfd;
        }

        /* ── popover ──────────────────────────── */
        div[data-testid="stPopover"] {
          border-radius: var(--radius) !important;
          box-shadow: 0 4px 24px rgba(0,0,0,.1) !important;
        }

        /* ── success / info toasts ────────────── */
        div[data-testid="stNotification"] {
          border-radius: 10px !important;
        }

        /* ── caption ──────────────────────────── */
        .stCaption {
          font-size: .75rem;
          color: var(--text-secondary);
        }

        /* ── number input ─────────────────────── */
        .stNumberInput > div > div > input {
          border-radius: 8px !important;
          border-color: var(--border) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
