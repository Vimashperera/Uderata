"""
Paths and labels for the Udarata Pa Saramba 01 desktop learner.

Override folder: set env UDARATA_EXPERT_VIDEOS to the full path of your expert clips.
"""
import json
import os
import shutil

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.normpath(os.path.join(APP_ROOT, "..", ".."))

_DEFAULT_EXPERT_REL = "Pa saramba 01 - experts videos"

ASSETS_DIR = os.path.join(APP_ROOT, "assets")
MODELS_DIR = os.path.join(APP_ROOT, "models")
JSON_PATH = os.path.join(APP_ROOT, "data", "pa_saramba_01.json")

# Preferred display clip names (first existing file wins).
_DISPLAY_VIDEO_CANDIDATES = (
    "Uderata pasaramba expert.mp4",
    "expert_display.mp4",
)

STEP_ID = "pa_saramba_01"
STEP_TITLE = "Pa Saramba 01"
DANCE_STYLE = "Udarata (Kandyan)"

# Shared pose settings (live practice + preprocess must match)
POSE_MODEL_COMPLEXITY = 1   # 0=lite, 1=full, 2=heavy — full is the Phase-1 standard
MIRROR_WEBCAM = True        # selfie-style flip so learner left ↔ expert left when facing camera


def resolve_display_video_path() -> str:
    """Return the best available expert display video under assets/."""
    for name in _DISPLAY_VIDEO_CANDIDATES:
        path = os.path.join(ASSETS_DIR, name)
        if os.path.isfile(path):
            return path
    # Default target used when copying / generating a clip
    return os.path.join(ASSETS_DIR, _DISPLAY_VIDEO_CANDIDATES[0])


VIDEO_PATH = resolve_display_video_path()


def _norm_folder_name(name: str) -> str:
    n = name.lower().strip()
    for ch in ("\u2013", "\u2014", "\u2212"):  # en-dash, em-dash, minus sign → hyphen
        n = n.replace(ch, "-")
    return " ".join(n.split())  # collapse weird spaces


def resolve_expert_source_dir() -> str:
    """
    Folder that contains expert .mp4 files.

    Looks next to any ancestor of the app (so it works whether you open the
    whole repo or only the inner folder), then env UDARATA_EXPERT_VIDEOS, then
    the legacy two-levels-up workspace path.
    """
    env = os.environ.get("UDARATA_EXPERT_VIDEOS", "").strip()
    if env and os.path.isdir(env):
        return os.path.abspath(env)

    rel = _DEFAULT_EXPERT_REL
    want = _norm_folder_name(rel)
    p = APP_ROOT
    seen = set()
    for _ in range(10):
        ap = os.path.abspath(p)
        if ap in seen:
            break
        seen.add(ap)
        parent = os.path.dirname(p)
        if parent == p:
            break

        # Sibling of current folder: ../Pa saramba 01 - experts videos
        cand = os.path.join(parent, rel)
        if os.path.isdir(cand):
            return os.path.abspath(cand)

        if os.path.isdir(parent):
            for entry in os.listdir(parent):
                full = os.path.join(parent, entry)
                if not os.path.isdir(full):
                    continue
                if _norm_folder_name(entry) == want:
                    return os.path.abspath(full)
                el = entry.lower()
                if "saramba" in el and ("expert" in el or "experts" in el):
                    return os.path.abspath(full)

        p = parent

    return os.path.abspath(os.path.join(WORKSPACE_ROOT, rel))


def _discover_first_expert_video(source_dir: str):
    if not os.path.isdir(source_dir):
        return None
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    found = []
    for dp, _, files in os.walk(source_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                found.append(os.path.join(dp, f))
    found.sort()
    return found[0] if found else None


def _write_placeholder_video(path: str, fps: float, duration_sec: float) -> bool:
    """Silent dark placeholder so preview/practice timing still works without expert clips."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    fps = max(float(fps or 30.0), 1.0)
    n_frames = max(int(round(duration_sec * fps)), 30)
    w, h = 640, 360
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        return False

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (30, 26, 26)
    cv2.putText(
        frame, "Expert video missing", (70, 150),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (201, 168, 76), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "Run: python preprocess_multi_expert.py", (40, 210),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (154, 154, 176), 1, cv2.LINE_AA,
    )
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    return os.path.isfile(path) and os.path.getsize(path) > 0


def ensure_runtime_assets() -> dict:
    """
    Create assets/models dirs and ensure a display video exists when possible.

    Priority:
      1. Known local assets (e.g. "Uderata pasaramba expert.mp4")
      2. Any other .mp4 already in assets/
      3. Copy first video from the expert source folder
      4. Build a silent placeholder from JSON timing (practice still scores)
    """
    global VIDEO_PATH

    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)

    VIDEO_PATH = resolve_display_video_path()
    result = {
        "video_ok": os.path.isfile(VIDEO_PATH),
        "json_ok": os.path.isfile(JSON_PATH),
        "video_source": "existing" if os.path.isfile(VIDEO_PATH) else None,
        "placeholder": False,
        "message": "",
        "video_path": VIDEO_PATH,
    }

    if result["video_ok"]:
        return result

    # Any other video already sitting in assets/
    local = _discover_first_expert_video(ASSETS_DIR)
    if local:
        VIDEO_PATH = local
        result["video_ok"] = True
        result["video_source"] = "assets"
        result["video_path"] = VIDEO_PATH
        result["message"] = f"Using display video:\n{VIDEO_PATH}"
        return result

    source_dir = resolve_expert_source_dir()
    src = _discover_first_expert_video(source_dir)
    target = os.path.join(ASSETS_DIR, _DISPLAY_VIDEO_CANDIDATES[0])
    if src:
        try:
            shutil.copy2(src, target)
            VIDEO_PATH = target
            result["video_ok"] = True
            result["video_source"] = "copied"
            result["video_path"] = VIDEO_PATH
            result["message"] = f"Copied display video from:\n{src}"
            return result
        except OSError as e:
            result["message"] = f"Could not copy expert video: {e}"

    # Fall back to JSON-timed placeholder so the app can still open
    fps, duration = 30.0, 17.0
    if result["json_ok"]:
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f).get("metadata", {})
            fps = float(meta.get("display_video_fps") or meta.get("fps") or 30.0)
            duration = float(meta.get("video_duration_seconds") or 17.0)
            df = int(meta.get("display_video_frames") or 0)
            if df > 0 and fps > 0:
                duration = df / fps
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    if _write_placeholder_video(target, fps, duration):
        VIDEO_PATH = target
        result["video_ok"] = True
        result["video_source"] = "placeholder"
        result["placeholder"] = True
        result["video_path"] = VIDEO_PATH
        result["message"] = (
            "Expert display video was missing. A silent placeholder was created "
            "so you can practice with angle scoring.\n\n"
            "Place your expert clip in assets/ (e.g. "
            "'Uderata pasaramba expert.mp4') or run:\n"
            "  python preprocess_multi_expert.py"
        )
        return result

    result["message"] = (
        f"Reference video not found under:\n{ASSETS_DIR}\n\n"
        "Expected something like:\n"
        "  Uderata pasaramba expert.mp4\n\n"
        "Or run:\n  python preprocess_multi_expert.py"
    )
    return result


EXPERT_SOURCE_DIR = resolve_expert_source_dir()
