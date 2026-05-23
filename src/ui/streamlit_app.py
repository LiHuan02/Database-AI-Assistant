import os
import streamlit as st

from core.pipeline import lcel_pipeline
from core.chat_manager import ChatManager

DATA_ROOT = "data"


def main():
    st.set_page_config(page_title="Database AI Assistant (Streamlit)")
    st.sidebar.title("Libraries")
    libs_dir = os.path.join(DATA_ROOT, "libraries")
    libs = [d for d in os.listdir(libs_dir)] if os.path.exists(libs_dir) else ["base"]
    lib = st.sidebar.selectbox("Library", libs)

    cm = ChatManager(data_root=DATA_ROOT)
    chats = cm.list_chats(lib)
    chat_map = {f"{c.get('title') or c.get('id')} ({c.get('id')})": c.get('id') for c in chats}
    chat_sel = st.sidebar.selectbox("Chat", list(chat_map.keys()) + ["New Chat"])
    if chat_sel == "New Chat":
        title = st.sidebar.text_input("Title", "New Chat")
        if st.sidebar.button("Create Chat"):
            chat_id = cm.create_chat(lib, title)
            st.experimental_rerun()
    else:
        chat_id = chat_map.get(chat_sel)

    st.title(f"Library: {lib}")

    if chat_id:
        history = cm.get_messages(lib, chat_id)
    else:
        history = []

    for m in history:
        if m.get('role') == 'user':
            st.markdown(f"**User:** {m.get('text')}")
        else:
            st.markdown(f"**Assistant:** {m.get('text')}")

    user_input = st.text_input("Your question")
    provider = st.selectbox("Provider", ['openai','azure_openai','openrouter','anthropic','local'])
    model = st.text_input("Model", value='gpt-3.5-turbo')
    embed_provider = st.sidebar.selectbox('Embed provider', ['openai','local','hash'])

    if st.button("Send") and user_input.strip():
        with st.spinner("Processing..."):
            assistant_text, contexts = lcel_pipeline(
                lib,
                chat_id,
                user_input.strip(),
                cm,
                data_root=DATA_ROOT,
                api_key=None,
                embedding_provider='openai' if st.sidebar.selectbox('Embed provider', ['openai','local','hash']) == 'openai' else 'local',
                embedding_model=None,
                llm_provider=provider,
                llm_model=model,
            )
            if assistant_text:
                cm.append_assistant_message(lib, chat_id, assistant_text)
            cm.append_user_message(lib, chat_id, user_input.strip())
            st.experimental_rerun()


if __name__ == '__main__':
    main()
