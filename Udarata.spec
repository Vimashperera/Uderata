# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Udarata desktop coaching app (Windows onedir).

Build:
  powershell -ExecutionPolicy Bypass -File build_exe.ps1

Output folder:
  dist\\Udarata\\Udarata.exe   (+ assets/ data/ models/ copied by the script)
"""

import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None
ROOT = os.path.abspath(SPECPATH)

# Heavy native stacks — collect everything they need
mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")
ctk_datas, ctk_binaries, ctk_hidden = collect_all("customtkinter")
cv2_datas, cv2_binaries, cv2_hidden = collect_all("cv2")

try:
    iff_datas, iff_binaries, iff_hidden = collect_all("imageio_ffmpeg")
except Exception:
    iff_datas, iff_binaries, iff_hidden = [], [], []

datas = []
datas += mp_datas
datas += ctk_datas
datas += cv2_datas
datas += iff_datas
datas += collect_data_files("mediapipe")

binaries = []
binaries += mp_binaries
binaries += ctk_binaries
binaries += cv2_binaries
binaries += iff_binaries

hiddenimports = sorted(
    set(
        list(mp_hidden)
        + list(ctk_hidden)
        + list(cv2_hidden)
        + list(iff_hidden)
        + [
            "PIL._tkinter_finder",
            "pygame",
            "reportlab",
            "matplotlib",
            "numpy",
            "customtkinter",
            "mediapipe",
            "cv2",
            "imageio_ffmpeg",
        ]
    )
)

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Udarata",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # show console for tester diagnostics
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Udarata",
)
