"""
PyInstaller entry point for Database AI Assistant.
Resolves app paths in both development and frozen (bundled) modes,
then launches the Streamlit server.
"""
import os
import sys
import threading
import webbrowser


def find_app_path() -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "src", "app.py")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "app.py")


def setup_environment() -> None:
    if not getattr(sys, "frozen", False):
        return
    exe_dir = os.path.dirname(sys.executable)
    os.chdir(exe_dir)
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"


def _open_browser() -> None:
    import time
    time.sleep(2)
    webbrowser.open("http://localhost:8501")


def main() -> None:
    setup_environment()
    app_path = find_app_path()

    if not os.path.exists(".env"):
        print("=" * 60)
        print("  WARNING: .env file not found!")
        print("  Copy .env.example to .env and fill in your API keys.")
        print("=" * 60)

    sys.argv = [
        "streamlit", "run", app_path,
        "--global.developmentMode", "false",
        "--server.headless", "true",
        "--browser.serverAddress", "0.0.0.0",
        "--server.port", "8501",
    ]

    threading.Thread(target=_open_browser, daemon=True).start()

    from streamlit.web import cli as stcli
    try:
        stcli.main()
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
