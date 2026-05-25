"""Desktop GUI for Database AI Assistant."""

import os
import shutil
import sys
import threading
import traceback

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import PySimpleGUI as sg
except ImportError as exc:
    print("PySimpleGUI is not installed. Run: pip install PySimpleGUI")
    print(exc)
    sys.exit(1)

try:
    from core.chat_manager import ChatManager
    from core.library_manager import LibraryManager
    from core.pipeline import lcel_pipeline
    from db.vector_store import sync_vector_store
    from utils.config import get_embedding_config, get_llm_config
    from utils.logger import get_logger
except ImportError as exc:
    print(f"Error importing modules: {exc}")
    traceback.print_exc()
    sys.exit(1)


DATA_ROOT = "data"


def main():
    log = get_logger()
    cm = ChatManager(data_root=DATA_ROOT)
    lm = LibraryManager(data_root=DATA_ROOT)
    lm.create_library("base")

    llm_config = get_llm_config()
    embedding_config = get_embedding_config()

    sg.theme("LightBlue")
    libraries = _list_libraries()

    lib_column = [
        [sg.Text("知识库", font=(None, 12, "bold"))],
        [sg.Combo(libraries, default_value=libraries[0] if libraries else "base", key="-LIB-", size=(24, 1), enable_events=True, readonly=True)],
        [sg.Button("刷新", key="-REFRESH_LIBS-"), sg.Input(key="-NEW_LIB-", size=(12, 1)), sg.Button("新建", key="-CREATE_LIB-")],
        [sg.HorizontalSeparator()],
        [sg.Text("对话", font=(None, 12, "bold"))],
        [sg.Listbox(values=[], size=(28, 19), key="-CHAT_LIST-", enable_events=True)],
        [sg.Input(key="-NEW_CHAT_TITLE-", size=(18, 1)), sg.Button("新建对话", key="-NEW_CHAT-")],
        [sg.Button("删除对话", key="-DEL_CHAT-")],
    ]

    chat_column = [
        [sg.Text("问答", font=(None, 14, "bold"))],
        [sg.Multiline("", size=(82, 30), key="-CHAT_VIEW-", disabled=True, autoscroll=True)],
        [sg.Input(key="-USER_INPUT-", size=(66, 1)), sg.Button("发送", key="-SEND-")],
        [sg.Checkbox("回答附带来源", key="-ATTACH_SOURCES-", default=True), sg.Button("显示上次来源", key="-SHOW_SOURCES-")],
        [sg.Multiline("", size=(82, 8), key="-SOURCE_VIEW-", disabled=True)],
    ]

    docs_column = [
        [sg.Text("文档", font=(None, 12, "bold"))],
        [sg.Listbox(values=[], size=(38, 19), key="-DOC_LIST-")],
        [sg.Input(key="-DOC_PATH-", visible=False), sg.FileBrowse("添加文档", target="-DOC_PATH-", key="-BROWSE_DOC-"), sg.Button("导入", key="-ADD_DOC-")],
        [sg.Button("删除文档", key="-DEL_DOC-")],
        [sg.Text("向量库会在文档变化或提问前自动同步。", key="-SYNC_STATUS-", size=(38, 2))],
    ]

    settings = [
        [sg.Text("LLM Provider"), sg.Input(llm_config.provider, key="-LLM_PROVIDER-", size=(14, 1)), sg.Text("Model"), sg.Input(llm_config.model, key="-LLM_MODEL-", size=(24, 1))],
        [sg.Text("LLM Base URL"), sg.Input(llm_config.base_url or "", key="-LLM_BASE_URL-", size=(48, 1))],
        [sg.Text("Embedding Provider"), sg.Input(embedding_config.provider, key="-EMBED_PROVIDER-", size=(14, 1)), sg.Text("Model"), sg.Input(embedding_config.model, key="-EMBED_MODEL-", size=(24, 1))],
        [sg.Text("Embedding Base URL"), sg.Input(embedding_config.base_url or "", key="-EMBED_BASE_URL-", size=(48, 1))],
    ]

    layout = [
        [sg.Column(lib_column), sg.VSeperator(), sg.Column(chat_column), sg.VSeperator(), sg.Column(docs_column)],
        [sg.HorizontalSeparator()],
        [sg.Column(settings)],
    ]

    window = sg.Window("Database AI Assistant", layout, finalize=True)

    current_lib = libraries[0] if libraries else "base"
    current_chat_id = None
    last_contexts = []

    def refresh_libs():
        nonlocal libraries, current_lib
        libraries = _list_libraries()
        if current_lib not in libraries:
            current_lib = libraries[0] if libraries else "base"
        window["-LIB-"].update(values=libraries, value=current_lib)

    def refresh_chats():
        chats = cm.list_chats(current_lib)
        names = [f"{item.get('title') or item.get('id')} ({item.get('id')})" for item in chats]
        window["-CHAT_LIST-"].update(names)

    def refresh_docs():
        docs_dir = _docs_dir(current_lib)
        docs_dir.mkdir(parents=True, exist_ok=True)
        docs = [p.name for p in sorted(docs_dir.iterdir()) if p.is_file()]
        window["-DOC_LIST-"].update(docs)

    def show_messages(chat_id):
        messages = cm.get_messages(current_lib, chat_id)
        window["-CHAT_VIEW-"].update("\n".join(_format_message(item) for item in messages))

    def run_in_background(fn, *args, event_key):
        def worker():
            try:
                window.write_event_value(event_key, {"success": True, "result": fn(*args)})
            except Exception as exc:
                log.exception("background task failed")
                window.write_event_value(event_key, {"success": False, "error": str(exc)})

        threading.Thread(target=worker, daemon=True).start()

    refresh_libs()
    refresh_chats()
    refresh_docs()

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break

        if event == "-REFRESH_LIBS-":
            refresh_libs()
            refresh_chats()
            refresh_docs()

        if event == "-CREATE_LIB-":
            name = (values.get("-NEW_LIB-") or "").strip()
            if name:
                lm.create_library(name)
                current_lib = name
                refresh_libs()
                refresh_chats()
                refresh_docs()

        if event == "-LIB-":
            current_lib = values.get("-LIB-") or "base"
            current_chat_id = None
            window["-CHAT_VIEW-"].update("")
            window["-SOURCE_VIEW-"].update("")
            refresh_chats()
            refresh_docs()

        if event == "-NEW_CHAT-":
            title = values.get("-NEW_CHAT_TITLE-") or "New Chat"
            current_chat_id = cm.create_chat(current_lib, title)
            refresh_chats()
            show_messages(current_chat_id)

        if event == "-CHAT_LIST-":
            selection = values.get("-CHAT_LIST-")
            if selection:
                current_chat_id = _chat_id_from_label(selection[0])
                show_messages(current_chat_id)

        if event == "-DEL_CHAT-":
            selection = values.get("-CHAT_LIST-")
            if selection:
                cm.delete_chat(current_lib, _chat_id_from_label(selection[0]))
                current_chat_id = None
                window["-CHAT_VIEW-"].update("")
                refresh_chats()

        if event == "-ADD_DOC-":
            src = values.get("-DOC_PATH-")
            if src and os.path.isfile(src):
                dst = _docs_dir(current_lib) / os.path.basename(src)
                shutil.copy2(src, dst)
                refresh_docs()
                window["-SYNC_STATUS-"].update("文档已导入，正在同步向量库...")
                run_in_background(_sync_from_values, current_lib, values, event_key="-BG_SYNC-")

        if event == "-DEL_DOC-":
            selection = values.get("-DOC_LIST-")
            if selection:
                path = _docs_dir(current_lib) / selection[0]
                try:
                    path.unlink()
                    refresh_docs()
                    window["-SYNC_STATUS-"].update("文档已删除，正在同步向量库...")
                    run_in_background(_sync_from_values, current_lib, values, event_key="-BG_SYNC-")
                except Exception as exc:
                    log.exception("delete doc failed")
                    sg.popup_error(f"删除失败：{exc}")

        if event == "-BG_SYNC-":
            payload = values.get(event) or {}
            if payload.get("success"):
                result = payload.get("result") or {}
                inserted = result.get("inserted")
                window["-SYNC_STATUS-"].update(f"向量库已同步：{inserted if inserted is not None else 0} chunks")
            else:
                window["-SYNC_STATUS-"].update(f"同步失败：{payload.get('error', 'unknown error')}")

        if event == "-SHOW_SOURCES-":
            window["-SOURCE_VIEW-"].update(_format_sources(last_contexts) if last_contexts else "暂无来源")

        if event == "-SEND-":
            text = (values.get("-USER_INPUT-") or "").strip()
            if not current_chat_id:
                sg.popup("请先选择或新建一个对话")
                continue
            if not text:
                continue
            cm.append_user_message(current_lib, current_chat_id, text)
            window["-SYNC_STATUS-"].update("正在同步并检索文档...")
            try:
                assistant_text, contexts = lcel_pipeline(
                    current_lib,
                    current_chat_id,
                    text,
                    cm,
                    data_root=DATA_ROOT,
                    api_key=None,
                    base_url=values.get("-LLM_BASE_URL-") or None,
                    embedding_api_key=None,
                    embedding_base_url=values.get("-EMBED_BASE_URL-") or None,
                    embedding_provider=values.get("-EMBED_PROVIDER-") or None,
                    embedding_model=values.get("-EMBED_MODEL-") or None,
                    llm_provider=values.get("-LLM_PROVIDER-") or None,
                    llm_model=values.get("-LLM_MODEL-") or None,
                    k=5,
                )
                last_contexts = contexts or []
                store_text = assistant_text
                if values.get("-ATTACH_SOURCES-") and last_contexts:
                    store_text += "\n\n来源：\n" + _format_sources(last_contexts, compact=True)
                cm.append_assistant_message(current_lib, current_chat_id, store_text)
                show_messages(current_chat_id)
                window["-SOURCE_VIEW-"].update(_format_sources(last_contexts))
                window["-SYNC_STATUS-"].update("完成")
                window["-USER_INPUT-"].update("")
            except Exception as exc:
                log.exception("send failed")
                sg.popup_error(f"处理失败：{exc}")

    window.close()


def _list_libraries():
    root = os.path.join(DATA_ROOT, "libraries")
    if not os.path.exists(root):
        return ["base"]
    libs = [item.name for item in os.scandir(root) if item.is_dir()]
    return libs or ["base"]


def _docs_dir(library_name: str):
    from pathlib import Path

    return Path(DATA_ROOT) / "libraries" / library_name / "docs"


def _format_message(message: dict) -> str:
    role = message.get("role", "").upper()
    text = message.get("text", "")
    return f"{role}:\n{text}\n"


def _chat_id_from_label(label: str) -> str:
    return label.split("(")[-1].strip(")") if "(" in label and label.endswith(")") else label


def _format_sources(contexts, compact: bool = False) -> str:
    parts = []
    limit = 260 if compact else 1200
    for i, context in enumerate(contexts):
        source = context.get("meta", {}).get("source", "")
        score = context.get("score")
        text = (context.get("text") or "")[:limit].replace("\n", " ")
        parts.append(f"[{i + 1}] {source} score={score}\n{text}")
    return "\n\n".join(parts)


def _sync_from_values(library_name: str, values: dict):
    return sync_vector_store(
        library_name,
        data_root=DATA_ROOT,
        api_key=None,
        embedding_provider=values.get("-EMBED_PROVIDER-") or None,
        embedding_model=values.get("-EMBED_MODEL-") or None,
        embedding_base_url=values.get("-EMBED_BASE_URL-") or None,
    )


if __name__ == "__main__":
    main()
