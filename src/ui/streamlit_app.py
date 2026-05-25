import html
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core.chat_manager import ChatManager
from core.library_manager import LibraryManager
from core.pipeline import stream_lcel_pipeline
from db.vector_store import sync_vector_store
from utils.config import get_embedding_config, get_llm_config


DATA_ROOT = "data"

LLM_PROVIDERS = ["dashscope", "openai", "deepseek", "siliconflow", "zhipu", "moonshot", "openrouter", "compatible", "local", "手动填写"]
EMBEDDING_PROVIDERS = ["dashscope", "openai", "siliconflow", "zhipu", "compatible", "local", "hash", "手动填写"]
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


def main():
    st.set_page_config(page_title="Database AI Assistant", layout="wide")
    _inject_css()

    library_manager = LibraryManager(data_root=DATA_ROOT)
    library_manager.create_library("base")
    chat_manager = ChatManager(data_root=DATA_ROOT)

    llm_config = get_llm_config()
    embedding_config = get_embedding_config()
    _init_state(llm_config, embedding_config)

    with st.sidebar:
        selected_library = _library_panel(library_manager)
        embedding_settings = _embedding_panel(selected_library)
        _document_panel(selected_library, embedding_settings)

    llm_settings = _chat_header()
    chat_id = _chat_selector(chat_manager, selected_library)
    _render_history(chat_manager, selected_library, chat_id)

    user_input = st.chat_input("输入问题，Enter 发送")
    if user_input:
        _handle_chat_submit(
            chat_manager,
            selected_library,
            chat_id,
            user_input,
            llm_settings,
            embedding_settings,
        )


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
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _library_panel(library_manager: LibraryManager) -> str:
    st.markdown("### 知识库")
    libraries = [item.get("name") for item in library_manager.list_libraries()] or ["base"]
    selected = st.selectbox("当前知识库", libraries, label_visibility="collapsed")
    with st.expander("新建知识库"):
        name = st.text_input("名称", key="new_library_name")
        if st.button("创建", use_container_width=True):
            if not name.strip():
                st.error("知识库名称不能为空")
            else:
                library_manager.create_library(name.strip())
                st.rerun()
    return selected


def _embedding_panel(library_name: str) -> dict:
    st.markdown("### 向量模型")
    provider = _provider_select("embedding_provider", EMBEDDING_PROVIDERS)
    model = _model_select("embedding_model", EMBEDDING_MODELS.get(provider, ["手动填写"]))
    with st.popover("向量接口设置", use_container_width=True):
        api_key = st.text_input("API Key", value=st.session_state.embedding_api_key, type="password", key="embedding_api_key_input")
        base_url = st.text_input("Base URL", value=st.session_state.embedding_base_url, key="embedding_base_url_input")
        if st.button("保存向量设置", use_container_width=True):
            st.session_state.embedding_api_key = api_key
            st.session_state.embedding_base_url = base_url
            st.success("已保存到当前会话")
    if st.button("同步向量库", use_container_width=True):
        try:
            result = sync_vector_store(
                library_name,
                data_root=DATA_ROOT,
                api_key=st.session_state.embedding_api_key or None,
                embedding_provider=provider,
                embedding_model=model,
                embedding_base_url=st.session_state.embedding_base_url or None,
            )
            st.success(f"同步完成，写入 {result.get('inserted') or 0} 个文本块")
        except Exception as exc:
            st.error(f"同步失败：{exc}")
    return {
        "provider": provider,
        "model": model,
        "api_key": st.session_state.embedding_api_key or None,
        "base_url": st.session_state.embedding_base_url or None,
    }


def _document_panel(library_name: str, embedding_settings: dict):
    st.markdown("### 文档")
    docs_dir = Path(DATA_ROOT) / "libraries" / library_name / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    uploaded = st.file_uploader("上传文档", type=["txt", "md", "pdf", "docx"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            dst = docs_dir / uploaded.name
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            shutil.move(tmp_path, dst)
            sync_vector_store(
                library_name,
                data_root=DATA_ROOT,
                api_key=embedding_settings["api_key"],
                embedding_provider=embedding_settings["provider"],
                embedding_model=embedding_settings["model"],
                embedding_base_url=embedding_settings["base_url"],
            )
            st.success("文档已上传并同步")
            st.rerun()
        except Exception as exc:
            st.error(f"上传或同步失败：{exc}")

    docs = [item.name for item in sorted(docs_dir.iterdir()) if item.is_file()]
    if not docs:
        st.caption("暂无文档")
    for doc in docs:
        col_name, col_action = st.columns([0.78, 0.22], vertical_alignment="center")
        col_name.caption(doc)
        if col_action.button("删除", key=f"delete-doc-{library_name}-{doc}"):
            try:
                (docs_dir / doc).unlink()
                sync_vector_store(
                    library_name,
                    data_root=DATA_ROOT,
                    api_key=embedding_settings["api_key"],
                    embedding_provider=embedding_settings["provider"],
                    embedding_model=embedding_settings["model"],
                    embedding_base_url=embedding_settings["base_url"],
                )
                st.rerun()
            except Exception as exc:
                st.error(f"删除或同步失败：{exc}")


def _chat_header() -> dict:
    left, model_col, settings_col = st.columns([0.48, 0.34, 0.18], vertical_alignment="center")
    left.markdown("<h1>Database AI Assistant</h1>", unsafe_allow_html=True)
    with model_col:
        provider = _provider_select("llm_provider", LLM_PROVIDERS, label="语言模型厂商")
        model = _model_select("llm_model", LLM_MODELS.get(provider, ["手动填写"]), label="语言模型")
    with settings_col:
        with st.popover("接口设置", use_container_width=True):
            api_key = st.text_input("LLM API Key", value=st.session_state.llm_api_key, type="password", key="llm_api_key_input")
            base_url = st.text_input("LLM Base URL", value=st.session_state.llm_base_url, key="llm_base_url_input")
            if st.button("保存语言模型设置", use_container_width=True):
                st.session_state.llm_api_key = api_key
                st.session_state.llm_base_url = base_url
                st.success("已保存到当前会话")
    return {
        "provider": provider,
        "model": model,
        "api_key": st.session_state.llm_api_key or None,
        "base_url": st.session_state.llm_base_url or None,
    }


def _chat_selector(chat_manager: ChatManager, library_name: str) -> str:
    chats = chat_manager.list_chats(library_name)
    if not chats:
        return chat_manager.create_chat(library_name, "新对话")
    chat_map = {f"{item.get('title') or item.get('id')}": item.get("id") for item in chats}
    selected_title = st.selectbox("对话", list(chat_map.keys()), label_visibility="collapsed")
    col_new, col_delete = st.columns([0.5, 0.5])
    if col_new.button("新对话", use_container_width=True):
        chat_id = chat_manager.create_chat(library_name, "新对话")
        st.session_state.active_chat_id = chat_id
        st.rerun()
    chat_id = st.session_state.get("active_chat_id") or chat_map[selected_title]
    if chat_id not in chat_map.values():
        chat_id = chat_map[selected_title]
    if col_delete.button("删除对话", use_container_width=True):
        chat_manager.delete_chat(library_name, chat_id)
        st.session_state.pop("active_chat_id", None)
        st.rerun()
    return chat_id


def _render_history(chat_manager: ChatManager, library_name: str, chat_id: str):
    st.markdown('<div class="chat-frame">', unsafe_allow_html=True)
    for message in chat_manager.get_messages(library_name, chat_id):
        _render_message(message.get("role", "assistant"), message.get("text", ""))
    st.markdown("</div>", unsafe_allow_html=True)


def _handle_chat_submit(
    chat_manager: ChatManager,
    library_name: str,
    chat_id: str,
    user_input: str,
    llm_settings: dict,
    embedding_settings: dict,
):
    _render_message("user", user_input)
    placeholder = st.empty()
    chunks = []

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
        chat_manager.append_user_message(library_name, chat_id, user_input)
        chat_manager.append_assistant_message(library_name, chat_id, answer)
    st.rerun()


def _provider_select(state_key: str, options: list[str], label: str = "厂商") -> str:
    current = st.session_state.get(state_key) or options[0]
    index = options.index(current) if current in options else len(options) - 1
    selected = st.selectbox(label, options, index=index, key=f"{state_key}_select")
    if selected == "手动填写":
        selected = st.text_input("手动填写厂商", value=current if current not in options else "", key=f"{state_key}_manual")
    st.session_state[state_key] = selected
    return selected


def _model_select(state_key: str, options: list[str], label: str = "模型") -> str:
    current = st.session_state.get(state_key) or options[0]
    index = options.index(current) if current in options else len(options) - 1
    selected = st.selectbox(label, options, index=index, key=f"{state_key}_select")
    if selected == "手动填写":
        selected = st.text_input("手动填写模型", value=current if current not in options else "", key=f"{state_key}_manual")
    st.session_state[state_key] = selected
    return selected


def _render_message(role: str, text: str):
    safe_text = html.escape(text).replace("\n", "<br>")
    klass = "user" if role == "user" else "assistant"
    name = "你" if role == "user" else "助手"
    st.markdown(
        f"""
        <div class="msg-row {klass}">
          <div class="msg-bubble">
            <div class="msg-name">{name}</div>
            <div>{safe_text}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_css():
    st.markdown(
        """
        <style>
        .block-container { max-width: 1160px; padding-top: 1.4rem; }
        h1 { font-size: 1.55rem !important; margin: 0 0 .2rem 0 !important; }
        section[data-testid="stSidebar"] { border-right: 1px solid #ececf1; }
        .chat-frame { padding: .5rem 0 1.5rem; }
        .msg-row { display: flex; margin: .75rem 0; }
        .msg-row.assistant { justify-content: flex-start; }
        .msg-row.user { justify-content: flex-end; }
        .msg-bubble {
            max-width: min(760px, 78%);
            padding: .78rem .95rem;
            border-radius: 14px;
            line-height: 1.65;
            font-size: .96rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
            word-break: break-word;
        }
        .msg-row.assistant .msg-bubble {
            background: #f7f7f8;
            color: #202123;
            border-bottom-left-radius: 5px;
        }
        .msg-row.user .msg-bubble {
            background: #2563eb;
            color: white;
            border-bottom-right-radius: 5px;
        }
        .msg-name {
            font-size: .75rem;
            opacity: .72;
            margin-bottom: .25rem;
        }
        div[data-testid="stChatInput"] { max-width: 900px; margin: 0 auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
