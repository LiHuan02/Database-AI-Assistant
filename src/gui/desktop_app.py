"""
Minimal desktop GUI for Database AI Assistant using PySimpleGUI.
Features:
- Library selector + create
- Chat list per library
- Chat view showing history
- API key and model selector inputs
- Send message (appends to chat history and simulates assistant response)

This is a local GUI skeleton; actual LLM calls / retrieval not executed here.
"""

import os
import sys
import shutil
import traceback
import threading

# ensure src is on path so imports of core/ work when running this script directly
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import PySimpleGUI first - check if available
try:
    import PySimpleGUI as sg
except ImportError as e:
    print(f"Error: PySimpleGUI not installed. Run: pip install PySimpleGUI")
    print(f"Details: {e}")
    sys.exit(1)

# Import other modules with error handling
try:
    from core.chat_manager import ChatManager
    from db.chroma_client import ChromaClient
    from core import ingest as ingest_module
    from core import retrieval as retrieval_module
    from core.library_manager import LibraryManager
    from db import vector_store as vector_store_module
    from core.llm import generate_reply
    from core.pipeline import lcel_pipeline
    from utils.logger import get_logger
    import re
except ImportError as e:
    print(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

DATA_ROOT = "data"


def main():
    try:
        cm = ChatManager(data_root=DATA_ROOT)
        cc = ChromaClient(data_root=DATA_ROOT)  # kept for future use
        lm = LibraryManager(data_root=DATA_ROOT)
        log = get_logger()
    except Exception as e:
        get_logger().exception("Error initializing components")
        traceback.print_exc()
        sg.popup_error(f"初始化失败: {e}")
        return

    # Ensure base library exists
    base_lib_dir = os.path.join(DATA_ROOT, "libraries", "base")
    os.makedirs(base_lib_dir, exist_ok=True)

    libraries = [p.name for p in os.scandir(os.path.join(DATA_ROOT, "libraries")) if p.is_dir()] if os.path.exists(os.path.join(DATA_ROOT, "libraries")) else ["base"]

    sg.theme("LightBlue")

    # Left column: libraries and chats
    lib_column = [
        [sg.Text("Libraries", font=(None, 12, 'bold'))],
        [sg.Combo(libraries, key='-LIB-', size=(24, 1), enable_events=True, readonly=True)],
        [sg.Button('Refresh', key='-REFRESH_LIBS-'), sg.Input(key='-NEW_LIB-', size=(12,1)), sg.Button('Create', key='-CREATE_LIB-')],
        [sg.HorizontalSeparator()],
        [sg.Text('Chats', font=(None, 12, 'bold'))],
        [sg.Listbox(values=[], size=(24, 20), key='-CHAT_LIST-', enable_events=True)],
        [sg.Input(key='-NEW_CHAT_TITLE-', size=(18,1)), sg.Button('New Chat', key='-NEW_CHAT-')],
        [sg.Button('Delete Chat', key='-DEL_CHAT-')]
    ]

    # Center column: main chat view
    chat_column = [
        [sg.Text('Chat', font=(None, 14, 'bold'))],
        [sg.Multiline('', size=(80, 30), key='-CHAT_VIEW-', disabled=True, autoscroll=True)],
        [sg.Input(key='-USER_INPUT-', size=(60,1)), sg.Button('Send', key='-SEND-')],
        [sg.Checkbox('附带来源到答案', key='-ATTACH_SOURCES-'), sg.Button('显示来源', key='-SHOW_SOURCES-')]
    ]

    # Right column: knowledge base docs and retrieval
    kb_column = [
        [sg.Text('Knowledge Base', font=(None, 12, 'bold'))],
        [sg.Listbox(values=[], size=(40, 10), key='-DOC_LIST-', enable_events=True)],
        [sg.Button('Open', key='-OPEN_DOC-'), sg.Button('Delete', key='-DEL_DOC-')],
        [sg.Button('Build Vector Store', key='-BUILD_VS-')],
        [sg.Text('', key='-INGEST_STATUS-')],
        [sg.HorizontalSeparator()],
        [sg.Text('Retrieval Results', font=(None, 12, 'bold'))],
        [sg.Listbox(values=[], size=(40, 8), key='-RESULT_LIST-', enable_events=True)],
        [sg.Button('Insert to Input', key='-INSERT_RESULT-'), sg.Button('Insert as System', key='-INSERT_SYS-'), sg.Button('Insert as Assistant', key='-INSERT_AS_ASSIST-')],
        [sg.Multiline('', size=(40, 10), key='-RESULT_VIEW-', disabled=True)]
    ]

    # Bottom settings row
    settings = [
        [sg.Text('API Key:'), sg.Input('', key='-API_KEY-', password_char='*', size=(30,1))],
        [sg.Text('LLM Provider:'), sg.Combo(['openai','azure_openai','openrouter','anthropic','local'], default_value='openai', key='-PROVIDER-'), sg.Text('LLM Model:'), sg.Input('gpt-3.5-turbo', key='-MODEL-', size=(20,1))],
        [sg.Text('Embed Provider:'), sg.Combo(['openai','local','hash'], default_value='openai', key='-EMBED_PROVIDER-'), sg.Text('Embed Model:'), sg.Input('text-embedding-3-small', key='-EMBED_MODEL-', size=(20,1))]
    ]

    layout = [
        [sg.Column(lib_column), sg.VSeperator(), sg.Column(chat_column), sg.VSeperator(), sg.Column(kb_column)],
        [sg.HorizontalSeparator()],
        [sg.Column(settings)]
    ]

    window = sg.Window('Database AI Assistant (Desktop)', layout, finalize=True)

    current_lib = 'base'
    current_chat_id = None
    results_cache = []
    system_contexts = []
    last_query = ''
    last_contexts = []

    def refresh_libs():
        nonlocal libraries, current_lib
        p = os.path.join(DATA_ROOT, 'libraries')
        if not os.path.exists(p):
            libraries = ['base']
        else:
            libraries = [d.name for d in os.scandir(p) if d.is_dir()]
        window['-LIB-'].update(values=libraries)
        # set selection to first library and update current_lib
        if libraries:
            current_lib = libraries[0]
            window['-LIB-'].update(value=current_lib)

    def refresh_chats():
        if current_lib:
            chats = cm.list_chats(current_lib)
            names = [f"{c.get('title') or c.get('id')} ({c.get('id')})" for c in chats]
            window['-CHAT_LIST-'].update(names)
        refresh_docs()


    def refresh_docs():
        ddir = os.path.join(DATA_ROOT, 'libraries', current_lib, 'docs')
        vals = []
        if os.path.exists(ddir):
            vals = [f for f in os.listdir(ddir) if os.path.isfile(os.path.join(ddir, f))]
        window['-DOC_LIST-'].update(vals)

    def _highlight_text(text: str, query: str) -> str:
        """Return text with query terms wrapped for visual emphasis."""
        if not query or not text:
            return text
        try:
            terms = [re.escape(t) for t in query.split() if t.strip()]
            if not terms:
                return text
            pattern = re.compile("(" + "|".join(terms) + ")", re.IGNORECASE)
            # wrap matches with >>> <<< markers
            return pattern.sub(lambda m: f"»{m.group(0)}«", text)
        except Exception:
            return text

    def run_in_background(fn, *args, event_key=None):
        """Run fn(*args) in a background thread and post result to window via event_key."""
        def _worker():
            try:
                res = fn(*args)
                window.write_event_value(event_key, {'success': True, 'result': res})
            except Exception as e:
                log.exception('background task failed')
                window.write_event_value(event_key, {'success': False, 'error': str(e)})

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _format_message(m: dict) -> str:
        role = m.get('role', '')
        text = m.get('text', '')
        ts = m.get('ts') or ''
        if ts:
            try:
                # show only time portion
                t = ts.replace('T', ' ').split('.')[0]
            except Exception:
                t = ts
            head = f"[{t}] {role.upper()}:"
        else:
            head = f"{role.upper()}:"
        return f"{head}\n{text}\n"

    def show_messages(chat_id):
        msgs = cm.get_messages(current_lib, chat_id)
        lines = [ _format_message(m) for m in msgs ]
        window['-CHAT_VIEW-'].update('\n'.join(lines))

    refresh_libs()
    refresh_chats()
    refresh_docs()

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == '-REFRESH_LIBS-':
            refresh_libs()
        if event == '-CREATE_LIB-':
            name = values.get('-NEW_LIB-')
            if name:
                lm.create_library(name)
                refresh_libs()
        # Ingest via file browse removed; use Build Vector Store which scans docs directory
        if event == '-BUILD_VS-':
            # build vector store from all docs in current library (use selected embedding provider/model)
            window['-INGEST_STATUS-'].update('开始构建向量库...')
            embed_provider = values.get('-EMBED_PROVIDER-') or 'openai'
            embed_model = values.get('-EMBED_MODEL-') or None
            run_in_background(vector_store_module.build_vector_store, current_lib, DATA_ROOT, values.get('-API_KEY-'), embed_provider, embed_model, event_key='-BG_BUILD-')
        
        if event == '-BG_BUILD-':
            payload = values.get(event)
            if payload and payload.get('success'):
                res = payload.get('result') or {}
                if res.get('error'):
                    window['-INGEST_STATUS-'].update(f"构建失败: {res.get('error')}")
                else:
                    window['-INGEST_STATUS-'].update(f"构建完成: {res.get('inserted',0)} chunks")
            else:
                err = payload.get('error') if payload else '未知错误'
                log.exception('background build error: %s', err)
                window['-INGEST_STATUS-'].update(f"构建异常: {err}")
                sg.popup_error(f"构建异常: {err}")
        if event == '-LIB-':
            current_lib = values.get('-LIB-') or 'base'
            refresh_chats()
            refresh_docs()
        if event == '-NEW_CHAT-':
            title = values.get('-NEW_CHAT_TITLE-') or 'New Chat'
            chat_id = cm.create_chat(current_lib, title)
            refresh_chats()
        if event == '-CHAT_LIST-':
            selection = values.get('-CHAT_LIST-')
            if selection:
                # extract chat id from display
                s = selection[0]
                if '(' in s and s.endswith(')'):
                    chat_id = s.split('(')[-1].strip(')')
                    current_chat_id = chat_id
                    show_messages(chat_id)
        if event == '-DEL_CHAT-':
            sel = values.get('-CHAT_LIST-')
            if sel:
                s = sel[0]
                if '(' in s and s.endswith(')'):
                    chat_id = s.split('(')[-1].strip(')')
                    cm.delete_chat(current_lib, chat_id)
                    current_chat_id = None
                    window['-CHAT_VIEW-'].update('')
                    refresh_chats()
                    refresh_docs()
        if event == '-SEARCH-':
            query = values.get('-USER_INPUT-') or ''
            if not query.strip():
                sg.popup('请输入检索查询')
            else:
                try:
                    results = retrieval_module.retrieve_relevant(current_lib, query.strip(), k=10, data_root=DATA_ROOT, api_key=values.get('-API_KEY-'))
                except Exception as e:
                    results = []
                    log.exception('retrieval error')
                results_cache = results
                display = []
                for i, r in enumerate(results):
                    score = r.get('score')
                    src = r.get('meta', {}).get('source', '')
                    snippet = (r.get('text') or '')[:200].replace('\n', ' ')
                    display.append(f"[{i}] score:{score:.3f} src:{src} snippet:{snippet}")
                window['-RESULT_LIST-'].update(display)
                window['-RESULT_VIEW-'].update('')
                # store last query for highlighting
                last_query = query
        if event == '-RESULT_LIST-':
            sel = values.get('-RESULT_LIST-')
            if sel:
                s = sel[0]
                # parse index from display
                try:
                    idx = int(s.split(']')[0].lstrip('['))
                    r = results_cache[idx]
                    text = r.get('text') or ''
                    # gather meta
                    meta = r.get('meta') or {}
                    src = meta.get('source') or meta.get('file') or ''
                    score = r.get('score')
                    # highlight based on last_query
                    try:
                        h = _highlight_text(text, last_query)
                    except Exception:
                        h = text
                    # formatted view: source, score, snippet, and full text
                    parts = []
                    parts.append(f"来源: {src}")
                    if score is not None:
                        try:
                            parts.append(f"相似度: {float(score):.4f}")
                        except Exception:
                            parts.append(f"相似度: {score}")
                    parts.append('\n')
                    parts.append(h[:20000])
                    if len(text) > 20000:
                        parts.append('\n\n(内容已截断)')
                    window['-RESULT_VIEW-'].update('\n'.join(parts))
                except Exception:
                    window['-RESULT_VIEW-'].update('')
        if event == '-INSERT_RESULT-':
            sel = values.get('-RESULT_LIST-')
            if sel:
                try:
                    idx = int(sel[0].split(']')[0].lstrip('['))
                    r = results_cache[idx]
                    cur = values.get('-USER_INPUT-') or ''
                    add = (r.get('text') or '')[:1000]
                    window['-USER_INPUT-'].update(cur + '\n' + add)
                except Exception:
                    pass
        if event == '-INSERT_SYS-':
            sel = values.get('-RESULT_LIST-')
            if sel:
                try:
                    idx = int(sel[0].split(']')[0].lstrip('['))
                    r = results_cache[idx]
                    system_contexts.append(r.get('text') or '')
                    sg.popup('已将选中文档片段加入系统上下文')
                except Exception:
                    pass
        if event == '-INSERT_AS_ASSIST-':
            sel = values.get('-RESULT_LIST-')
            if sel and current_chat_id:
                try:
                    idx = int(sel[0].split(']')[0].lstrip('['))
                    r = results_cache[idx]
                    text_to_insert = r.get('text') or ''
                    cm.append_assistant_message(current_lib, current_chat_id, text_to_insert)
                    show_messages(current_chat_id)
                    sg.popup('已将检索片段插入为助手消息')
                except Exception:
                    pass
        if event == '-OPEN_DOC-':
            sel = values.get('-DOC_LIST-')
            if sel:
                fn = sel[0]
                path = os.path.join(DATA_ROOT, 'libraries', current_lib, 'docs', fn)
                try:
                    text = open(path, 'r', encoding='utf-8', errors='ignore').read()
                except Exception:
                    log.exception('open doc failed: %s', path)
                    text = '无法读取文件内容'
                window['-DOC_VIEW-'].update(text[:20000])
        if event == '-DEL_DOC-':
            sel = values.get('-DOC_LIST-')
            if sel:
                fn = sel[0]
                path = os.path.join(DATA_ROOT, 'libraries', current_lib, 'docs', fn)
                try:
                    os.remove(path)
                    refresh_docs()
                    window['-DOC_VIEW-'].update('')
                    # after deleting a document, rebuild vector store in background
                    embed_provider = values.get('-EMBED_PROVIDER-') or 'openai'
                    embed_model = values.get('-EMBED_MODEL-') or None
                    run_in_background(vector_store_module.build_vector_store, current_lib, DATA_ROOT, values.get('-API_KEY-'), embed_provider, embed_model, event_key='-BG_BUILD-')
                except Exception as e:
                    log.exception('delete doc failed: %s', path)
                    sg.popup_error(f'删除失败: {e}')
        if event == '-SHOW_SOURCES-':
            # display last retrieval contexts in the result view
            if last_contexts:
                parts = []
                for i, c in enumerate(last_contexts):
                    src = c.get('meta', {}).get('source', '')
                    score = c.get('score')
                    txt = (c.get('text') or '')[:2000].replace('\n', ' ')
                    parts.append(f"[{i}] 来源: {src}  相似度: {score}\n{txt}\n")
                window['-RESULT_VIEW-'].update('\n'.join(parts))
            else:
                sg.popup('暂无检索来源')

        if event == '-SEND-':
            text = values.get('-USER_INPUT-')
            if not current_chat_id:
                sg.popup('请选择或新建一个对话')
            elif text and text.strip():
                # append user message
                cm.append_user_message(current_lib, current_chat_id, text.strip())
                # use LCEL pipeline: summarize -> retrieve -> answer
                provider = values.get('-PROVIDER-') or 'openai'
                try:
                    assistant_text, contexts = lcel_pipeline(
                        current_lib,
                        current_chat_id,
                        text.strip(),
                        cm,
                        data_root=DATA_ROOT,
                        api_key=values.get('-API_KEY-'),
                        embedding_provider=values.get('-EMBED_PROVIDER-') or 'openai',
                        embedding_model=values.get('-EMBED_MODEL-') or None,
                        llm_provider=provider,
                        llm_model=values.get('-MODEL-') or 'gpt-3.5-turbo',
                        k=5,
                    )
                except Exception:
                    log.exception('LCEL pipeline failed')
                    contexts = []
                    assistant_text = '（回退）系统暂时无法生成回答。'

                # save last contexts for display
                last_contexts = contexts or []

                # if attach option enabled, append sources to assistant message
                attach = values.get('-ATTACH_SOURCES-')
                store_text = assistant_text
                if attach and last_contexts:
                    s_parts = ["\n\n来源："]
                    for c in last_contexts:
                        src = c.get('meta', {}).get('source', '')
                        score = c.get('score')
                        snippet = (c.get('text') or '')[:500].replace('\n', ' ')
                        s_parts.append(f"来源:{src} 相似度:{score}\n{snippet}\n---\n")
                    store_text = assistant_text + '\n'.join(s_parts)

                cm.append_assistant_message(current_lib, current_chat_id, store_text)
                show_messages(current_chat_id)
                window['-USER_INPUT-'].update('')

    window.close()


if __name__ == '__main__':
    main()
