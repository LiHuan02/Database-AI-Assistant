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

# --- Streamlit 静态资源 ---
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
        streamlit_datas.append((str(dist_info), dist_info.name))
    for pkg in ["altair", "chromadb", "openai", "langchain", "langchain_text_splitters",
                 "pydantic", "pdfplumber", "python-docx", "markdown"]:
        for dist_info in site_packages.glob(f"{pkg}-*.dist-info"):
            streamlit_datas.append((str(dist_info), dist_info.name))
except Exception:
    pass

# --- 自动收集子模块（解决动态导入问题） ---
streamlit_hidden  = collect_submodules("streamlit")
chromadb_hidden   = collect_submodules("chromadb")
onnxruntime_hidden = collect_submodules("onnxruntime")

# --- chromadb / onnxruntime 数据文件 ---
chromadb_datas = collect_data_files("chromadb")
onnx_datas = collect_data_files("onnxruntime")
onnx_binaries = collect_dynamic_libs("onnxruntime")

# --- 应用数据文件 ---
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
        *streamlit_hidden,        # ← 自动包含所有 streamlit 子模块
        *chromadb_hidden,         # ← 自动包含所有 chromadb 子模块
        *onnxruntime_hidden,      # ← 自动包含所有 onnxruntime 子模块
        "openai",
        "pdfplumber",
        "docx",
        "markdown",
        "langchain_text_splitters",
        "dotenv",
        "httpx",
        "httpcore",
        "urllib3",
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
