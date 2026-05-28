# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Database AI Assistant.
Build: pyinstaller DatabaseAIAssistant.spec --clean --noconfirm
"""
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

block_cipher = None

# --- Streamlit static assets (frontend HTML/JS/CSS) and metadata ---
streamlit_datas = []
try:
    import streamlit
    streamlit_root = Path(streamlit.__file__).parent
    # Frontend static assets
    static_dir = streamlit_root / "static"
    if static_dir.exists():
        streamlit_datas.append((str(static_dir), "streamlit/static"))
    # Package metadata (dist-info) — needed for importlib.metadata.version()
    site_packages = streamlit_root.parent
    for dist_info in site_packages.glob("streamlit-*.dist-info"):
        dest = dist_info.name
        streamlit_datas.append((str(dist_info), dest))
    # Also include altair, chromadb, and key package metadata
    for pkg in ["altair", "chromadb", "openai", "langchain", "langchain_text_splitters",
                 "pydantic", "pdfplumber", "python-docx", "markdown"]:
        for dist_info in site_packages.glob(f"{pkg}-*.dist-info"):
            streamlit_datas.append((str(dist_info), dist_info.name))
except Exception:
    pass

# --- chromadb data files ---
chromadb_datas = collect_data_files("chromadb")

# --- onnxruntime native binaries ---
onnx_datas = collect_data_files("onnxruntime")
onnx_binaries = collect_dynamic_libs("onnxruntime")

# --- Application data files bundled into the executable ---
add_datas = [
    ("src", "src"),
    (".streamlit", ".streamlit"),
    (".env.example", "."),
]
add_datas.extend(streamlit_datas)
add_datas.extend(chromadb_datas)
add_datas.extend(onnx_datas)

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=onnx_binaries,
    datas=add_datas,
    hiddenimports=[
        # Streamlit
        "streamlit",
        "streamlit.web.cli",
        "streamlit.web.bootstrap",
        "streamlit.runtime",
        "streamlit.runtime.scriptrunner",
        "streamlit.commands",
        "streamlit.elements",
        "streamlit.proto",
        "streamlit.watcher",
        "streamlit.watcher.local_sources_watcher",
        # ChromaDB
        "chromadb",
        "chromadb.api",
        "chromadb.config",
        "chromadb.db",
        "chromadb.db.impl.sqlite",
        "chromadb.segment",
        "chromadb.segment.impl.vector.brute_force_index",
        "chromadb.types",
        # onnxruntime
        "onnxruntime",
        "onnxruntime.capi",
        # OpenAI client
        "openai",
        # Document processing
        "pdfplumber",
        "docx",
        "markdown",
        # LangChain
        "langchain_text_splitters",
        # Config
        "dotenv",
        # Networking / serialization
        "httpx",
        "httpcore",
        "urllib3",
        # Streamlit UI dependencies
        "tornado",
        "watchdog",
        "altair",
        "pyarrow",
        "rich",
        "pydantic",
        "jsonschema",
        "bcrypt",
        "tokenizers",
        "uvicorn",
        "starlette",
        "typer",
        "yaml",
        "overrides",
        "tenacity",
        # Misc
        "packaging",
        "tqdm",
        "certifi",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "_tkinter",
        "matplotlib",
        "pandas",
        "scipy",
        "PIL",
        "numpy.core._dotblas",
        "numpy.random._examples",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="DatabaseAIAssistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
