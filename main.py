"""
main.py — Udarata Pa Saramba 01 desktop learning app entry point.
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REQUIRED = {
    "cv2":           "opencv-python>=4.8.0",
    "mediapipe":     "mediapipe>=0.10.0",
    "customtkinter": "customtkinter>=5.2.0",
    "numpy":         "numpy>=1.24.0",
    "PIL":           "Pillow>=10.0.0",
    "matplotlib":    "matplotlib>=3.7.0",
}

missing = []
for module, pkg in REQUIRED.items():
    try:
        __import__(module)
    except ImportError:
        missing.append(pkg)

if missing:
    print("=" * 60)
    print("  Missing required packages. Run:")
    print("    pip install -r requirements.txt")
    print("  Missing:")
    for m in missing:
        print(f"    • {m}")
    print("=" * 60)
    sys.exit(1)

OPTIONAL = {"pygame": "pygame>=2.5.0"}
for module, pkg in OPTIONAL.items():
    try:
        __import__(module)
    except ImportError:
        print(f"[WARN] Optional: {pkg}")

if __name__ == "__main__":
    import config
    from ui.app import App

    print("=" * 60)
    print("  Udarata Dance — Pa Saramba 01")
    print("  Starting…")
    print("=" * 60)

    assets = config.ensure_runtime_assets()
    if assets.get("message") and assets.get("placeholder"):
        print(f"[INFO] {assets['message'].splitlines()[0]}")
    elif assets.get("video_source") == "copied":
        print(f"[INFO] {assets['message']}")

    app = App()
    app.mainloop()
