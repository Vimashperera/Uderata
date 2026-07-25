"""
Paths and labels for the Udarata desktop learner (multi-step).

Override folder: set env UDARATA_EXPERT_VIDEOS to the full path of your expert clips.
"""
import json
import os
import shutil
import subprocess

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.normpath(os.path.join(APP_ROOT, "..", ".."))

ASSETS_DIR = os.path.join(APP_ROOT, "assets")
MODELS_DIR = os.path.join(APP_ROOT, "models")
DATA_DIR = os.path.join(APP_ROOT, "data")

DANCE_STYLE = "Udarata (Kandyan)"

# Shared pose settings (live practice + preprocess must match)
POSE_MODEL_COMPLEXITY = 1   # 0=lite, 1=full, 2=heavy — full is the Phase-1 standard
MIRROR_WEBCAM = True        # selfie-style flip so learner left ↔ expert left when facing camera

AUDIO_WAV_NAME = "expert_display.wav"  # legacy fallback name
PLAYBACK_MAX_WIDTH = 1280  # downscale 4K+ clips for smooth preview/practice UI

# ── Step registry ────────────────────────────────────────────────────────────
# reference_loops: how many times the expert clip plays in practice (1 = no loop).
# Keep video_keywords specific so Pa Saramba 01/02/03 never steal each other's clips.
STEPS = {
    "namaskaraya": {
        "id": "namaskaraya",
        "title": "Namaskaraya",
        "description": (
            "The traditional Kandyan greeting sequence. Follow the expert once through "
            "— posture, hand placement, and calm timing."
        ),
        "preview_blurb": (
            "Namaskaraya is the respectful greeting that opens Udarata practice. "
            "Watch the expert's posture, hand paths, and timing, then perform the "
            "sequence once alongside the reference."
        ),
        "tags": ["Arms", "Torso", "Greeting"],
        "difficulty": "Foundation",
        "json_name": "namaskaraya.json",
        "video_candidates": (
            "Uderata namaskaraya expert.mp4",
        ),
        "video_keywords": ("namaskaraya", "namaskara"),
        "reference_loops": 1,  # play once — do not loop in practice
        "expert_source_rel": None,
    },
    "pa_saramba_01": {
        "id": "pa_saramba_01",
        "title": "Pa Saramba 01",
        "description": (
            "Practice against a fused expert timeline built from master performances. "
            "Match joint angles, bone lines, and musical timing."
        ),
        "preview_blurb": (
            "Pa Saramba 01 is a core Udarata (Kandyan) movement. The fused expert "
            "timeline blends several master performances so you can match the tradition "
            "without copying a single dancer exactly. Focus on rhythm, posture, and "
            "clean lines."
        ),
        "tags": ["Legs", "Arms", "Torso"],
        "difficulty": "Focus",
        "json_name": "pa_saramba_01.json",
        "video_candidates": (
            "Uderata pasaramba expert.mp4",
            "expert_display.mp4",
        ),
        # Avoid bare "sarambha" — that also matches Pa Saramba 02/03 filenames
        "video_keywords": ("pasaramba expert", "uderata pasaramba", "pa saramba 01"),
        "reference_loops": 3,
        "expert_source_rel": "Pa saramba 01 - experts videos",
    },
    "pa_saramba_02": {
        "id": "pa_saramba_02",
        "title": "Pa Saramba 02",
        "description": (
            "Madya laya Pa Saramba 02 — follow the expert through the mid-tempo phrase "
            "with clear weight shifts and arm lines."
        ),
        "preview_blurb": (
            "Pa Saramba 02 (Madya Laya) builds on the first phrase with a longer mid-tempo "
            "sequence. Stay grounded through the knees, keep arm paths clean, and match "
            "the musical timing of the expert."
        ),
        "tags": ["Legs", "Arms", "Madya Laya"],
        "difficulty": "Focus",
        "json_name": "pa_saramba_02.json",
        "video_candidates": (
            "pa sarambha 02 - madya laya.mp4",
            "pa saramba 02 - madya laya.mp4",
            "Uderata pasaramba 02 expert.mp4",
        ),
        "video_keywords": ("sarambha 02", "saramba 02", "pasaramba 02"),
        "reference_loops": 1,  # play once — do not loop in practice
        "expert_source_rel": None,
    },
    "pa_saramba_03": {
        "id": "pa_saramba_03",
        "title": "Pa Saramba 03",
        "description": (
            "Madya laya Pa Saramba 03 — a longer phrase for stamina, form consistency, "
            "and timing across the full sequence."
        ),
        "preview_blurb": (
            "Pa Saramba 03 (Madya Laya) is the longest phrase in this set. Watch how the "
            "expert sustains posture and rhythm, then practice along for the full clip."
        ),
        "tags": ["Legs", "Arms", "Madya Laya"],
        "difficulty": "Challenge",
        "json_name": "pa_saramba_03.json",
        "video_candidates": (
            "pa sarambha 03 - madya laya.mp4",
            "pa saramba 03 - madya laya.mp4",
            "Uderata pasaramba 03 expert.mp4",
        ),
        "video_keywords": ("sarambha 03", "saramba 03", "pasaramba 03"),
        "reference_loops": 1,  # play once — do not loop in practice
        "expert_source_rel": None,
    },
}

STEP_ORDER = ("namaskaraya", "pa_saramba_01", "pa_saramba_02", "pa_saramba_03")


def get_step(step_id: str) -> dict:
    step = STEPS.get(step_id)
    if step is None:
        raise KeyError(f"Unknown step_id: {step_id}")
    return step


def list_steps() -> list:
    return [STEPS[sid] for sid in STEP_ORDER if sid in STEPS]


def step_json_path(step_id: str) -> str:
    return os.path.join(DATA_DIR, get_step(step_id)["json_name"])


def resolve_display_video_path(step_id: str | None = None) -> str:
    """Return the best available expert display video for a step under assets/."""
    step_id = step_id or STEP_ID
    step = get_step(step_id)
    for name in step["video_candidates"]:
        path = os.path.join(ASSETS_DIR, name)
        if os.path.isfile(path):
            return path
    return os.path.join(ASSETS_DIR, step["video_candidates"][0])


def is_playback_proxy(path: str) -> bool:
    """True for generated lightweight playback proxies (not pose sources)."""
    base = os.path.basename(path).lower()
    stem, _ = os.path.splitext(base)
    return stem.endswith("_720p") or "_720p." in base


def video_matches_step(path: str, step_id: str) -> bool:
    """True if a filename looks like it belongs to this step (excludes proxies)."""
    if is_playback_proxy(path):
        return False
    step = get_step(step_id)
    base = os.path.basename(path).lower()
    for name in step["video_candidates"]:
        if base == name.lower():
            return True
    # Never claim another step's explicit teaching clip
    for other_id, other in STEPS.items():
        if other_id == step_id:
            continue
        for name in other.get("video_candidates", ()):
            if base == name.lower():
                return False
    return any(k in base for k in step.get("video_keywords", ()))


# Active step defaults — updated by set_active_step / ensure_runtime_assets
STEP_ID = "namaskaraya"
STEP_TITLE = STEPS[STEP_ID]["title"]
JSON_PATH = step_json_path(STEP_ID)
_DISPLAY_VIDEO_CANDIDATES = STEPS[STEP_ID]["video_candidates"]
VIDEO_PATH = resolve_display_video_path(STEP_ID)


def set_active_step(step_id: str) -> dict:
    """Point module-level path globals at the selected step."""
    global STEP_ID, STEP_TITLE, JSON_PATH, VIDEO_PATH, _DISPLAY_VIDEO_CANDIDATES
    step = get_step(step_id)
    STEP_ID = step["id"]
    STEP_TITLE = step["title"]
    JSON_PATH = step_json_path(STEP_ID)
    _DISPLAY_VIDEO_CANDIDATES = step["video_candidates"]
    VIDEO_PATH = resolve_display_video_path(STEP_ID)
    return step


def _source_stem_for_audio(video_path: str) -> str:
    """Strip playback-proxy suffixes so audio stays tied to the source clip."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    if stem.endswith("_720p"):
        stem = stem[: -len("_720p")]
    return stem


def resolve_expert_audio_path(video_path: str | None = None) -> str:
    """
    WAV path for the beat track beside the display video.
    Prefers <video_stem>.wav; falls back to legacy expert_display.wav if present.
    """
    vp = os.path.abspath(video_path or VIDEO_PATH)
    base_dir = os.path.dirname(vp)
    stem = _source_stem_for_audio(vp)
    preferred = os.path.join(base_dir, f"{stem}.wav")
    legacy = os.path.join(base_dir, AUDIO_WAV_NAME)
    if os.path.isfile(preferred) and os.path.getsize(preferred) > 0:
        return preferred
    if os.path.isfile(legacy) and os.path.getsize(legacy) > 0:
        return legacy
    return preferred


def playback_proxy_path(video_path: str) -> str:
    stem, ext = os.path.splitext(os.path.abspath(video_path))
    if stem.endswith("_720p"):
        return f"{stem}{ext}"
    return f"{stem}_720p{ext}"


def ensure_playback_video(
    video_path: str | None = None,
    max_width: int = PLAYBACK_MAX_WIDTH,
    force: bool = False,
) -> str:
    """
    Return a UI-friendly playback path. For wide/4K sources, build a ~720p
    proxy once so OpenCV preview/practice stays smooth.
    """
    video_path = os.path.abspath(video_path or resolve_display_video_path())
    if not os.path.isfile(video_path):
        return video_path

    try:
        import cv2
    except ImportError:
        return video_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return video_path
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if width <= max_width:
        return video_path

    proxy = playback_proxy_path(video_path)
    if (
        not force
        and os.path.isfile(proxy)
        and os.path.getsize(proxy) > 0
    ):
        src_mtime = os.path.getmtime(video_path)
        proxy_mtime = os.path.getmtime(proxy)
        # Rebuild when the source was replaced, or frame counts diverge
        if proxy_mtime >= src_mtime:
            pcap = cv2.VideoCapture(proxy)
            if pcap.isOpened():
                pframes = int(pcap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                pcap.release()
                if frames <= 0 or abs(pframes - frames) <= 2:
                    return proxy

    last_err = ""
    for ff in _ffmpeg_candidates():
        try:
            r = subprocess.run(
                [
                    ff, "-y", "-i", video_path,
                    "-vf", f"scale={int(max_width)}:-2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-an",
                    proxy,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode == 0 and os.path.isfile(proxy) and os.path.getsize(proxy) > 0:
                print(
                    f"[config] Playback proxy ready ({width}px -> {max_width}px):\n  {proxy}"
                )
                return proxy
            last_err = (r.stderr or r.stdout or "").strip()
        except FileNotFoundError:
            continue
        except Exception as e:
            last_err = str(e)
            continue

    if last_err:
        print(f"[config] Playback proxy failed, using source: {last_err[-300:]}")
    return video_path


def _ffmpeg_candidates() -> list:
    """Prefer system ffmpeg, then the binary bundled with imageio-ffmpeg."""
    found = ["ffmpeg"]
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            found.append(bundled)
    except Exception:
        pass
    return found


def ensure_expert_audio(video_path: str | None = None, force: bool = False) -> str | None:
    """
    Ensure a WAV exists beside the display video.
    Returns the WAV path on success, or None if extraction is unavailable.
    """
    video_path = video_path or resolve_display_video_path()
    if not os.path.isfile(video_path):
        return None

    wav_path = resolve_expert_audio_path(video_path)
    # If resolve returned a legacy file and force=False, keep it
    stem = _source_stem_for_audio(video_path)
    extract_target = os.path.join(
        os.path.dirname(os.path.abspath(video_path)), f"{stem}.wav"
    )
    if not force:
        if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            try:
                if os.path.getmtime(wav_path) >= os.path.getmtime(video_path):
                    return wav_path
            except OSError:
                return wav_path
    else:
        wav_path = extract_target

    os.makedirs(os.path.dirname(extract_target), exist_ok=True)
    last_err = ""
    for ff in _ffmpeg_candidates():
        try:
            r = subprocess.run(
                [
                    ff, "-y", "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                    extract_target,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if (
                r.returncode == 0
                and os.path.isfile(extract_target)
                and os.path.getsize(extract_target) > 0
            ):
                return extract_target
            last_err = (r.stderr or r.stdout or "").strip()
            if os.path.isfile(extract_target) and os.path.getsize(extract_target) == 0:
                try:
                    os.remove(extract_target)
                except OSError:
                    pass
        except FileNotFoundError:
            continue
        except Exception as e:
            last_err = str(e)
            continue

    if last_err:
        print(f"[config] Could not extract expert audio: {last_err[-300:]}")
    return None


def _norm_folder_name(name: str) -> str:
    n = name.lower().strip()
    for ch in ("\u2013", "\u2014", "\u2212"):  # en-dash, em-dash, minus sign → hyphen
        n = n.replace(ch, "-")
    return " ".join(n.split())  # collapse weird spaces


def resolve_expert_source_dir(step_id: str | None = None) -> str:
    """
    Folder that contains expert .mp4 files for a step.

    Looks next to any ancestor of the app, then env UDARATA_EXPERT_VIDEOS, then
    the legacy two-levels-up workspace path.
    """
    step_id = step_id or STEP_ID
    step = get_step(step_id)
    rel = step.get("expert_source_rel") or "Pa saramba 01 - experts videos"

    env = os.environ.get("UDARATA_EXPERT_VIDEOS", "").strip()
    if env and os.path.isdir(env):
        return os.path.abspath(env)

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
                if any(k in el for k in step["video_keywords"]) and (
                    "expert" in el or "experts" in el
                ):
                    return os.path.abspath(full)

        p = parent

    return os.path.abspath(os.path.join(WORKSPACE_ROOT, rel))


def _discover_step_videos(source_dir: str, step_id: str) -> list:
    if not os.path.isdir(source_dir):
        return []
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    found = []
    for dp, _, files in os.walk(source_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in exts:
                continue
            full = os.path.join(dp, f)
            if video_matches_step(full, step_id):
                found.append(full)
    found.sort()
    return found


def _discover_first_expert_video(source_dir: str, step_id: str | None = None):
    step_id = step_id or STEP_ID
    found = _discover_step_videos(source_dir, step_id)
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
        frame, "Run: python preprocess_multi_expert.py --step <id>", (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (154, 154, 176), 1, cv2.LINE_AA,
    )
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    return os.path.isfile(path) and os.path.getsize(path) > 0


def ensure_runtime_assets(step_id: str | None = None) -> dict:
    """
    Create assets/models dirs and ensure a display video + JSON exist for a step.
    """
    global VIDEO_PATH

    step_id = step_id or STEP_ID
    step = set_active_step(step_id)
    json_path = step_json_path(step_id)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    video_path = resolve_display_video_path(step_id)
    result = {
        "step_id": step_id,
        "step_title": step["title"],
        "video_ok": os.path.isfile(video_path),
        "json_ok": os.path.isfile(json_path),
        "video_source": "existing" if os.path.isfile(video_path) else None,
        "placeholder": False,
        "message": "",
        "video_path": video_path,
        "playback_path": video_path,
        "json_path": json_path,
        "reference_loops": int(step.get("reference_loops", 3)),
    }

    def _finalize_media(path: str) -> dict:
        ensure_expert_audio(path)
        result["playback_path"] = ensure_playback_video(path)
        result["video_path"] = path
        return result

    if result["video_ok"]:
        if not result.get("placeholder"):
            return _finalize_media(video_path)
        return result

    # Another matching clip already in assets/
    local = _discover_first_expert_video(ASSETS_DIR, step_id)
    if local:
        set_active_step(step_id)
        VIDEO_PATH = local
        result["video_ok"] = True
        result["video_source"] = "assets"
        result["message"] = f"Using display video:\n{local}"
        return _finalize_media(local)

    source_dir = resolve_expert_source_dir(step_id)
    src = _discover_first_expert_video(source_dir, step_id)
    target = os.path.join(ASSETS_DIR, step["video_candidates"][0])
    if src:
        try:
            shutil.copy2(src, target)
            set_active_step(step_id)
            VIDEO_PATH = target
            result["video_ok"] = True
            result["video_source"] = "copied"
            result["message"] = f"Copied display video from:\n{src}"
            return _finalize_media(target)
        except OSError as e:
            result["message"] = f"Could not copy expert video: {e}"

    # Fall back to JSON-timed placeholder so the app can still open
    fps, duration = 30.0, 17.0
    if result["json_ok"]:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f).get("metadata", {})
            fps = float(meta.get("display_video_fps") or meta.get("fps") or 30.0)
            duration = float(meta.get("video_duration_seconds") or 17.0)
            df = int(meta.get("display_video_frames") or 0)
            if df > 0 and fps > 0:
                duration = df / fps
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    if _write_placeholder_video(target, fps, duration):
        set_active_step(step_id)
        VIDEO_PATH = target
        result["video_ok"] = True
        result["video_source"] = "placeholder"
        result["placeholder"] = True
        result["video_path"] = target
        result["message"] = (
            f"Expert display video for {step['title']} was missing. "
            "A silent placeholder was created so you can practice with angle scoring.\n\n"
            f"Place your expert clip in assets/ (e.g. '{step['video_candidates'][0]}') "
            "or run:\n"
            f"  python preprocess_multi_expert.py --step {step_id}"
        )
        return result

    result["message"] = (
        f"Reference video for {step['title']} not found under:\n{ASSETS_DIR}\n\n"
        f"Expected something like:\n  {step['video_candidates'][0]}\n\n"
        f"Or run:\n  python preprocess_multi_expert.py --step {step_id}"
    )
    return result


EXPERT_SOURCE_DIR = resolve_expert_source_dir(STEP_ID)
