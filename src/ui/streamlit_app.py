import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core.chat_manager import ChatManager
from core.library_manager import LibraryManager
from core.pipeline import lcel_pipeline
from db.vector_store import sync_vector_store
from utils.config import get_embedding_config, get_llm_config


DATA_ROOT = "data"


def main():
    st.set_page_config(page_title="Database AI Assistant")

    lm = LibraryManager(data_root=DATA_ROOT)
    lm.create_library("base")
    cm = ChatManager(data_root=DATA_ROOT)

    libs = [item.get("name") for item in lm.list_libraries()] or ["base"]
    lib = st.sidebar.selectbox("知识库", libs)

    st.sidebar.caption("模型配置从 .env 读取，可在下方临时覆盖 provider/model/base_url。")
    llm_config = get_llm_config()
    embedding_config = get_embedding_config()
    llm_provider = st.sidebar.text_input("LLM Provider", llm_config.provider)
    llm_model = st.sidebar.text_input("LLM Model", llm_config.model)
    llm_base_url = st.sidebar.text_input("LLM Base URL", llm_config.base_url or "")
    embedding_provider = st.sidebar.text_input("Embedding Provider", embedding_config.provider)
    embedding_model = st.sidebar.text_input("Embedding Model", embedding_config.model)
    embedding_base_url = st.sidebar.text_input("Embedding Base URL", embedding_config.base_url or "")

    st.title(f"知识库：{lib}")
    _docs_panel(lib, embedding_provider, embedding_model, embedding_base_url)

    chats = cm.list_chats(lib)
    chat_map = {f"{c.get('title') or c.get('id')} ({c.get('id')})": c.get("id") for c in chats}
    chat_options = list(chat_map.keys()) + ["新建对话"]
    chat_sel = st.selectbox("对话", chat_options)
    if chat_sel == "新建对话":
        title = st.text_input("标题", "New Chat")
        if st.button("创建对话"):
            cm.create_chat(lib, title)
            st.rerun()
        return

    chat_id = chat_map.get(chat_sel)
    for message in cm.get_messages(lib, chat_id):
        with st.chat_message("user" if message.get("role") == "user" else "assistant"):
            st.markdown(message.get("text", ""))

    user_input = st.chat_input("输入问题")
    if user_input:
        cm.append_user_message(lib, chat_id, user_input)
        with st.spinner("同步文档、检索并生成回答..."):
            answer, contexts = lcel_pipeline(
                lib,
                chat_id,
                user_input,
                cm,
                data_root=DATA_ROOT,
                base_url=llm_base_url or None,
                embedding_base_url=embedding_base_url or None,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            if contexts:
                answer += "\n\n来源：\n" + "\n".join(
                    f"- {item.get('meta', {}).get('source', '')}" for item in contexts
                )
            cm.append_assistant_message(lib, chat_id, answer)
        st.rerun()


def _docs_panel(lib: str, embedding_provider: str, embedding_model: str, embedding_base_url: str):
    docs_dir = Path(DATA_ROOT) / "libraries" / lib / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    with st.expander("文档", expanded=True):
        uploaded = st.file_uploader("添加文档", type=["txt", "md", "pdf", "docx"])
        if uploaded is not None:
            dst = docs_dir / uploaded.name
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            shutil.move(tmp_path, dst)
            sync_vector_store(
                lib,
                data_root=DATA_ROOT,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url or None,
            )
            st.success("文档已导入并同步")
            st.rerun()

        docs = [p.name for p in sorted(docs_dir.iterdir()) if p.is_file()]
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(doc)
            if col2.button("删除", key=f"delete-{doc}"):
                (docs_dir / doc).unlink()
                sync_vector_store(
                    lib,
                    data_root=DATA_ROOT,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                    embedding_base_url=embedding_base_url or None,
                )
                st.rerun()


if __name__ == "__main__":
    main()
