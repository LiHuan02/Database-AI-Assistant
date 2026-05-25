"""Database AI Assistant 桌面端入口。"""

from pathlib import Path
import os
import shutil
import sys
import threading
import traceback

src_path = Path(__file__).resolve().parents[1]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    import PySimpleGUI as sg
except ImportError as exc:
    print("缺少 PySimpleGUI，请先运行：pip install PySimpleGUI")
    print(exc)
    sys.exit(1)

try:
    from core.chat_manager import ChatManager
    from core.library_manager import LibraryManager
    from core.pipeline import lcel_pipeline
    from db.vector_store import sync_vector_store
    from utils.config import get_embedding_config, get_llm_config
    from utils.logger import get_logger
except Exception as exc:
    print(f"导入模块失败：{exc}")
    traceback.print_exc()
    sys.exit(1)


DATA_ROOT = "data"
MANUAL = "手动填写"
LLM_PROVIDERS = ["dashscope", "openai", "deepseek", "siliconflow", "zhipu", "moonshot", "openrouter", "compatible", "local", MANUAL]
EMBEDDING_PROVIDERS = ["dashscope", "openai", "siliconflow", "zhipu", "compatible", "local", "hash", MANUAL]
LLM_MODELS = ["qwen-plus", "qwen-turbo", "qwen-max", "glm-5", "gpt-4o-mini", "deepseek-chat", MANUAL]
EMBEDDING_MODELS = ["text-embedding-v2", "text-embedding-v3", "text-embedding-3-small", "text-embedding-3-large", "BAAI/bge-m3", "hash", MANUAL]


def main():
    logger = get_logger()
    chat_manager = ChatManager(data_root=DATA_ROOT)
    library_manager = LibraryManager(data_root=DATA_ROOT)
    library_manager.create_library("base")

    llm_config = get_llm_config()
    embedding_config = get_embedding_config()

    sg.theme("SystemDefault")
    libraries = _list_libraries()
    current_library = libraries[0] if libraries else "base"
    current_chat_id = None

    layout = [
        [
            sg.Column(_library_column(libraries), expand_y=True),
            sg.VSeparator(),
            sg.Column(_chat_column(llm_config), expand_x=True, expand_y=True),
            sg.VSeparator(),
            sg.Column(_document_column(embedding_config), expand_y=True),
        ],
        [sg.StatusBar("就绪", key="-STATUS-", size=(120, 1))],
    ]

    window = sg.Window("Database AI Assistant", layout, finalize=True, resizable=True)

    def refresh_libraries():
        nonlocal libraries, current_library
        libraries = _list_libraries()
        if current_library not in libraries:
            current_library = libraries[0] if libraries else "base"
        window["-LIB-"].update(values=libraries, value=current_library)

    def refresh_chats():
        chats = chat_manager.list_chats(current_library)
        labels = [f"{item.get('title') or item.get('id')} ({item.get('id')})" for item in chats]
        window["-CHAT_LIST-"].update(labels)

    def refresh_docs():
        docs = [item.name for item in sorted(_docs_dir(current_library).iterdir()) if item.is_file()]
        window["-DOC_LIST-"].update(docs)

    def show_messages():
        if not current_chat_id:
            window["-CHAT_VIEW-"].update("")
            return
        messages = chat_manager.get_messages(current_library, current_chat_id)
        window["-CHAT_VIEW-"].update("\n\n".join(_format_message(item) for item in messages))

    refresh_libraries()
    refresh_chats()
    refresh_docs()

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break

        try:
            if event == "-REFRESH_LIBS-":
                refresh_libraries()
                refresh_chats()
                refresh_docs()

            elif event == "-CREATE_LIB-":
                name = (values.get("-NEW_LIB-") or "").strip()
                if not name:
                    sg.popup_error("知识库名称不能为空")
                    continue
                library_manager.create_library(name)
                current_library = name
                refresh_libraries()
                refresh_chats()
                refresh_docs()

            elif event == "-LIB-":
                current_library = values.get("-LIB-") or "base"
                current_chat_id = None
                refresh_chats()
                refresh_docs()
                show_messages()

            elif event == "-NEW_CHAT-":
                title = values.get("-NEW_CHAT_TITLE-") or "新对话"
                current_chat_id = chat_manager.create_chat(current_library, title)
                refresh_chats()
                show_messages()

            elif event == "-CHAT_LIST-":
                selected = values.get("-CHAT_LIST-")
                if selected:
                    current_chat_id = _chat_id_from_label(selected[0])
                    show_messages()

            elif event == "-DELETE_CHAT-":
                if current_chat_id:
                    chat_manager.delete_chat(current_library, current_chat_id)
                    current_chat_id = None
                    refresh_chats()
                    show_messages()

            elif event == "-ADD_DOC-":
                source = values.get("-DOC_PATH-")
                if source and os.path.isfile(source):
                    _docs_dir(current_library).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, _docs_dir(current_library) / os.path.basename(source))
                    refresh_docs()
                    _run_background(window, "-SYNC_DONE-", _sync_docs, current_library, values)
                    window["-STATUS-"].update("文档已导入，正在同步向量库...")

            elif event == "-DELETE_DOC-":
                selected = values.get("-DOC_LIST-")
                if selected:
                    (_docs_dir(current_library) / selected[0]).unlink()
                    refresh_docs()
                    _run_background(window, "-SYNC_DONE-", _sync_docs, current_library, values)
                    window["-STATUS-"].update("文档已删除，正在同步向量库...")

            elif event == "-SYNC_DOCS-":
                _run_background(window, "-SYNC_DONE-", _sync_docs, current_library, values)
                window["-STATUS-"].update("正在同步向量库...")

            elif event == "-SYNC_DONE-":
                payload = values.get("-SYNC_DONE-") or {}
                if payload.get("ok"):
                    result = payload.get("result") or {}
                    window["-STATUS-"].update(f"向量库同步完成，写入 {result.get('inserted') or 0} 个文本块")
                else:
                    window["-STATUS-"].update(f"向量库同步失败：{payload.get('error')}")

            elif event == "-LLM_SETTINGS-":
                _settings_popup(window, "语言模型接口", "-LLM_API_KEY-", "-LLM_BASE_URL-")

            elif event == "-EMBED_SETTINGS-":
                _settings_popup(window, "向量模型接口", "-EMBED_API_KEY-", "-EMBED_BASE_URL-")

            elif event == "-SEND-":
                question = (values.get("-USER_INPUT-") or "").strip()
                if not question:
                    continue
                if not current_chat_id:
                    current_chat_id = chat_manager.create_chat(current_library, "新对话")
                    refresh_chats()
                chat_manager.append_user_message(current_library, current_chat_id, question)
                show_messages()
                window["-USER_INPUT-"].update("")
                window["-STATUS-"].update("正在同步、检索并生成回答...")
                _run_background(window, "-ANSWER_DONE-", _answer, current_library, current_chat_id, question, chat_manager, values)

            elif event == "-ANSWER_DONE-":
                payload = values.get("-ANSWER_DONE-") or {}
                if payload.get("ok"):
                    chat_manager.append_assistant_message(current_library, current_chat_id, payload.get("answer", ""))
                    show_messages()
                    window["-STATUS-"].update("完成")
                else:
                    error = payload.get("error") or "未知错误"
                    chat_manager.append_assistant_message(current_library, current_chat_id, f"处理失败：{error}")
                    show_messages()
                    window["-STATUS-"].update(f"处理失败：{error}")

        except Exception as exc:
            logger.exception("桌面端事件处理失败")
            sg.popup_error(f"操作失败：{exc}")

    window.close()


def _library_column(libraries):
    return [
        [sg.Text("知识库", font=(None, 12, "bold"))],
        [sg.Combo(libraries, key="-LIB-", size=(24, 1), enable_events=True, readonly=True)],
        [sg.Button("刷新", key="-REFRESH_LIBS-"), sg.Input(key="-NEW_LIB-", size=(14, 1)), sg.Button("新建", key="-CREATE_LIB-")],
        [sg.HorizontalSeparator()],
        [sg.Text("对话", font=(None, 12, "bold"))],
        [sg.Listbox(values=[], size=(32, 22), key="-CHAT_LIST-", enable_events=True)],
        [sg.Input("新对话", key="-NEW_CHAT_TITLE-", size=(20, 1)), sg.Button("新对话", key="-NEW_CHAT-")],
        [sg.Button("删除对话", key="-DELETE_CHAT-")],
    ]


def _chat_column(llm_config):
    provider = llm_config.provider if llm_config.provider in LLM_PROVIDERS else MANUAL
    model = llm_config.model if llm_config.model in LLM_MODELS else MANUAL
    return [
        [sg.Text("问答", font=(None, 14, "bold"))],
        [
            sg.Text("语言模型"),
            sg.Combo(LLM_PROVIDERS, default_value=provider, key="-LLM_PROVIDER-", size=(16, 1)),
            sg.Combo(LLM_MODELS, default_value=model, key="-LLM_MODEL_SELECT-", size=(24, 1)),
            sg.Input(llm_config.model, key="-LLM_MODEL_MANUAL-", size=(24, 1)),
            sg.Button("接口设置", key="-LLM_SETTINGS-"),
        ],
        [sg.Multiline("", key="-CHAT_VIEW-", size=(86, 30), disabled=True, autoscroll=True, expand_x=True, expand_y=True)],
        [sg.Input(key="-USER_INPUT-", size=(76, 1), expand_x=True), sg.Button("发送", key="-SEND-", bind_return_key=True)],
        [sg.Input(llm_config.api_key or "", key="-LLM_API_KEY-", visible=False), sg.Input(llm_config.base_url or "", key="-LLM_BASE_URL-", visible=False)],
    ]


def _document_column(embedding_config):
    provider = embedding_config.provider if embedding_config.provider in EMBEDDING_PROVIDERS else MANUAL
    model = embedding_config.model if embedding_config.model in EMBEDDING_MODELS else MANUAL
    return [
        [sg.Text("向量库", font=(None, 12, "bold"))],
        [
            sg.Text("向量模型"),
            sg.Combo(EMBEDDING_PROVIDERS, default_value=provider, key="-EMBED_PROVIDER-", size=(16, 1)),
        ],
        [
            sg.Combo(EMBEDDING_MODELS, default_value=model, key="-EMBED_MODEL_SELECT-", size=(24, 1)),
            sg.Input(embedding_config.model, key="-EMBED_MODEL_MANUAL-", size=(24, 1)),
        ],
        [sg.Button("接口设置", key="-EMBED_SETTINGS-"), sg.Button("同步向量库", key="-SYNC_DOCS-")],
        [sg.HorizontalSeparator()],
        [sg.Text("文档", font=(None, 12, "bold"))],
        [sg.Listbox(values=[], size=(44, 22), key="-DOC_LIST-")],
        [sg.Input(key="-DOC_PATH-", visible=False), sg.FileBrowse("选择文档", target="-DOC_PATH-", file_types=(("Documents", "*.txt;*.md;*.pdf;*.docx"),)), sg.Button("导入", key="-ADD_DOC-")],
        [sg.Button("删除文档", key="-DELETE_DOC-")],
        [sg.Input(embedding_config.api_key or "", key="-EMBED_API_KEY-", visible=False), sg.Input(embedding_config.base_url or "", key="-EMBED_BASE_URL-", visible=False)],
    ]


def _settings_popup(window, title: str, api_key_key: str, base_url_key: str):
    layout = [
        [sg.Text("API Key"), sg.Input(window[api_key_key].get(), key="-API-", password_char="*", size=(54, 1))],
        [sg.Text("Base URL"), sg.Input(window[base_url_key].get(), key="-BASE-", size=(54, 1))],
        [sg.Button("保存"), sg.Button("取消")],
    ]
    popup = sg.Window(title, layout, modal=True, finalize=True)
    event, values = popup.read()
    if event == "保存":
        window[api_key_key].update(values.get("-API-", ""))
        window[base_url_key].update(values.get("-BASE-", ""))
    popup.close()


def _run_background(window, event_key, fn, *args):
    def worker():
        try:
            window.write_event_value(event_key, {"ok": True, "result": fn(*args)})
        except Exception as exc:
            window.write_event_value(event_key, {"ok": False, "error": str(exc)})

    threading.Thread(target=worker, daemon=True).start()


def _answer(library_name, chat_id, question, chat_manager, values):
    answer, _ = lcel_pipeline(
        library_name,
        chat_id,
        question,
        chat_manager,
        data_root=DATA_ROOT,
        api_key=values.get("-LLM_API_KEY-") or None,
        base_url=values.get("-LLM_BASE_URL-") or None,
        embedding_api_key=values.get("-EMBED_API_KEY-") or None,
        embedding_base_url=values.get("-EMBED_BASE_URL-") or None,
        embedding_provider=_selected_provider(values, "-EMBED_PROVIDER-"),
        embedding_model=_selected_model(values, "-EMBED_MODEL_SELECT-", "-EMBED_MODEL_MANUAL-"),
        llm_provider=_selected_provider(values, "-LLM_PROVIDER-"),
        llm_model=_selected_model(values, "-LLM_MODEL_SELECT-", "-LLM_MODEL_MANUAL-"),
    )
    return answer


def _sync_docs(library_name, values):
    return sync_vector_store(
        library_name,
        data_root=DATA_ROOT,
        api_key=values.get("-EMBED_API_KEY-") or None,
        embedding_provider=_selected_provider(values, "-EMBED_PROVIDER-"),
        embedding_model=_selected_model(values, "-EMBED_MODEL_SELECT-", "-EMBED_MODEL_MANUAL-"),
        embedding_base_url=values.get("-EMBED_BASE_URL-") or None,
    )


def _selected_provider(values, key):
    return values.get(key) or None


def _selected_model(values, select_key, manual_key):
    selected = values.get(select_key)
    if selected == MANUAL:
        return values.get(manual_key) or None
    return selected or values.get(manual_key) or None


def _list_libraries():
    root = Path(DATA_ROOT) / "libraries"
    root.mkdir(parents=True, exist_ok=True)
    libraries = [item.name for item in sorted(root.iterdir()) if item.is_dir()]
    return libraries or ["base"]


def _docs_dir(library_name: str) -> Path:
    path = Path(DATA_ROOT) / "libraries" / library_name / "docs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chat_id_from_label(label: str) -> str:
    return label.rsplit("(", 1)[-1].strip(")") if label.endswith(")") and "(" in label else label


def _format_message(message: dict) -> str:
    role = "你" if message.get("role") == "user" else "助手"
    return f"{role}：\n{message.get('text', '')}"


if __name__ == "__main__":
    main()
