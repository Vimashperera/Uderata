"""
preprocess_multi_expert.py
--------------------------
Phase-4 expert rebuild for Pa Saramba 01:

  1. Discover expert videos (source folder + assets + env override)
  2. Pick a canonical clip (prefer assets display video)
  3. Extract pose angles + torso-frame bones (shared VIDEO model)
  4. DTW-align every expert onto the canonical timeline
  5. Median-fuse + store per-joint variance / tolerance bands

Output:
  data/pa_saramba_01.json
  assets/<canonical display video>  (copied if needed)

Usage (from UdarataPaSaramba folder):
  python preprocess_multi_expert.py

Optional PowerShell:
  $env:UDARATA_EXPERT_VIDEOS = "E:\\SLIIT\\FINAL RESEARCH\\Expert data"
  python preprocess_multi_expert.py
"""
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (  # noqa: E402
    WORKSPACE_ROOT,
    APP_ROOT,
    ASSETS_DIR,
    resolve_expert_source_dir,
    resolve_display_video_path,
    JSON_PATH,
    VIDEO_PATH,
    STEP_TITLE,
    DANCE_STYLE,
    POSE_MODEL_COMPLEXITY,
    _DISPLAY_VIDEO_CANDIDATES,
)
from core.angle_calculator import ALL_JOINT_NAMES, compute_joint_angles  # noqa: E402
from core.motion_features import (  # noqa: E402
    BONE_NAMES,
    compute_bone_directions,
)
from core.expert_fusion import fuse_experts_canonical  # noqa: E402

POSE_CONFIDENCE = 0.7
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Extra known research folders (optional; skipped if missing)
_EXTRA_VIDEO_ROOTS = [
    os.path.join("E:\\", "SLIIT", "FINAL RESEARCH", "Expert data"),
]


def discover_videos(root: str):
    paths = []
    if not os.path.isdir(root):
        return paths
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                paths.append(os.path.join(dp, f))
    paths.sort()
    return paths


def discover_all_expert_videos() -> list:
    """Union of env/source dir, assets/, and known research folders."""
    roots = []
    env = os.environ.get("UDARATA_EXPERT_VIDEOS", "").strip()
    if env:
        roots.append(env)
    roots.append(resolve_expert_source_dir())
    roots.append(ASSETS_DIR)
    roots.extend(_EXTRA_VIDEO_ROOTS)

    preferred_names = {n.lower() for n in _DISPLAY_VIDEO_CANDIDATES}
    preferred_names.add("uderata pasaramba expert.mp4")

    by_size = {}  # size -> abspath (prefer teaching clip names)
    for root in roots:
        for p in discover_videos(root):
            ap = os.path.abspath(p)
            try:
                size_sig = os.path.getsize(ap)
            except OSError:
                continue
            base = os.path.basename(ap).lower()
            if size_sig not in by_size:
                by_size[size_sig] = ap
                continue
            # Prefer display/teaching filename when content size matches
            cur = os.path.basename(by_size[size_sig]).lower()
            if base in preferred_names and cur not in preferred_names:
                by_size[size_sig] = ap

    vids = sorted(by_size.values())
    return vids


def choose_canonical(vids: list) -> int:
    """Always prefer assets/'Uderata pasaramba expert.mp4' as the teaching timeline."""
    display = os.path.abspath(resolve_display_video_path())
    for i, p in enumerate(vids):
        if os.path.abspath(p) == display:
            return i
    # Ranked fallbacks if display path isn't in the discovered set
    ranked = [
        "uderata pasaramba expert.mp4",
        "expert_display.mp4",
        "pa_sarambha_expert.mp4",
        "pa saramba expert.mp4",
        "expert_video.mp4",
    ]
    lower_map = {os.path.basename(p).lower(): i for i, p in enumerate(vids)}
    for name in ranked:
        if name in lower_map:
            return lower_map[name]
    return 0


def extract_angles_for_video(video_path: str, extractor) -> tuple:
    """Returns (frame_records, fps, total_frames). Each record may include bones."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if hasattr(extractor, "reset_sequence"):
        extractor.reset_sequence()
    records = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = extractor.process_frame(frame, fps=float(fps))
        world = extractor.get_world_landmarks(results)
        angles = None
        bones = None
        if world is not None:
            key_ix = [11, 12, 23, 24, 25, 26, 27, 28]
            avg_vis = float(np.mean([world[i]["visibility"] for i in key_ix]))
            if avg_vis >= POSE_CONFIDENCE:
                ad = compute_joint_angles(
                    world, confidence_threshold=POSE_CONFIDENCE, use_world=True
                )
                angles = {
                    k: (round(v, 4) if v is not None else None) for k, v in ad.items()
                }
                bones = compute_bone_directions(
                    world, confidence_threshold=POSE_CONFIDENCE
                )
        records.append({
            "frame": fi,
            "angles": angles if angles is not None else {k: None for k in ALL_JOINT_NAMES},
            "bones": bones,
            "pose_detected": angles is not None,
        })
        fi += 1
    cap.release()
    return records, float(fps), fi


def main():
    print("=" * 70)
    print("  Udarata — Pa Saramba 01 — Phase-4 Canonical Expert Rebuild")
    print("=" * 70)

    source_dir = resolve_expert_source_dir()
    print(f"\nApp root:\n  {APP_ROOT}")
    print(f"Primary expert folder:\n  {source_dir}")
    print(f"  Exists: {os.path.isdir(source_dir)}")

    vids = discover_all_expert_videos()
    if not vids:
        print("\n[ERROR] No expert videos found.")
        print(
            "Place .mp4 files in assets/ or an expert folder, or set:\n"
            '  $env:UDARATA_EXPERT_VIDEOS = \"E:\\SLIIT\\FINAL RESEARCH\\Expert data\"\n'
            "  python preprocess_multi_expert.py\n"
        )
        sys.exit(1)

    canon_i = choose_canonical(vids)
    print(f"\nFound {len(vids)} expert video(s):")
    for i, p in enumerate(vids):
        mark = " [CANONICAL]" if i == canon_i else ""
        print(f"  • {p}{mark}")
    print()

    try:
        import mediapipe as mp  # noqa: F401
    except ImportError:
        print("[ERROR] mediapipe not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    from core.pose_extractor import PoseExtractor

    extractor = PoseExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=POSE_MODEL_COMPLEXITY,
        running_mode="VIDEO",
    )

    all_records = []
    metas = []
    t0 = time.time()
    for vp in vids:
        print(f"\nProcessing: {os.path.basename(vp)}")
        # Fresh landmarker per clip (MediaPipe VIDEO timestamps are sticky)
        extractor.reset_sequence()
        recs, fps, nfr = extract_angles_for_video(vp, extractor)
        cov = sum(1 for r in recs if r["pose_detected"]) / max(nfr, 1) * 100
        print(f"  frames={nfr}  fps={fps:.2f}  pose_coverage={cov:.1f}%")
        if cov < 5.0:
            print("  [WARN] Near-zero pose coverage — check video / model.")
        all_records.append(recs)
        metas.append({
            "file": os.path.basename(vp),
            "path": vp,
            "frames": nfr,
            "fps": fps,
            "pose_coverage_pct": round(cov, 2),
        })

    extractor.release()

    print("\nDTW-aligning experts to canonical timeline + median fusion…")
    fused_frames, fusion_meta = fuse_experts_canonical(
        all_records, canonical_index=canon_i
    )
    fused_fps = float(metas[canon_i]["fps"] if metas else 30.0)

    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # Never replace the designated teaching clip with another expert file.
    display_target = os.path.join(ASSETS_DIR, _DISPLAY_VIDEO_CANDIDATES[0])
    try:
        src = vids[canon_i]
        if not os.path.isfile(display_target):
            # Only seed assets if the missing teaching clip is absent
            if os.path.basename(src).lower() == os.path.basename(display_target).lower():
                shutil.copy2(src, display_target)
                print(f"\nSeeded assets teaching video:\n  {display_target}")
            else:
                print(
                    f"\n[WARN] Teaching clip missing — place "
                    f"'{_DISPLAY_VIDEO_CANDIDATES[0]}' in assets/ before relying on display sync."
                )
        else:
            print(f"\nTeaching display video (canonical reference):\n  {display_target}")
            if os.path.abspath(src) != os.path.abspath(display_target):
                print(
                    f"  Fusion canonical source: {src}\n"
                    f"  (assets teaching clip is never overwritten by other experts)"
                )
    except Exception as e:
        print(f"\n[WARN] Display video check failed: {e}")
        display_target = (
            display_target if os.path.isfile(display_target) else vids[canon_i]
        )

    wav_path = os.path.join(ASSETS_DIR, "expert_display.wav")
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-i", display_target,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                wav_path,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if r.returncode == 0 and os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            print(f"Reference audio (WAV):\n  {wav_path}")
        elif r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip()
            print(f"\n[INFO] ffmpeg WAV skipped: {(err[-300:] if err else 'no stderr')}")
            if os.path.isfile(wav_path) and os.path.getsize(wav_path) == 0:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
    except FileNotFoundError:
        print("\n[INFO] ffmpeg not in PATH — skipping WAV.")
    except Exception as e:
        print(f"\n[WARN] WAV extraction failed: {e}")

    disp_cap = cv2.VideoCapture(display_target)
    disp_frames = int(disp_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if disp_cap.isOpened() else 0
    disp_fps = (
        disp_cap.get(cv2.CAP_PROP_FPS) or fused_fps if disp_cap.isOpened() else fused_fps
    )
    disp_cap.release()

    # Prefer display clip length for sync when close to fused length
    duration = len(fused_frames) / max(fused_fps, 1e-6)
    if disp_frames > 0 and disp_fps > 0:
        duration = disp_frames / disp_fps

    output_data = {
        "metadata": {
            "dance_style": DANCE_STYLE,
            "step_name": STEP_TITLE,
            "fusion": fusion_meta.get("fusion", "canonical_dtw_median"),
            "canonical_index": fusion_meta.get("canonical_index", canon_i),
            "canonical_video": os.path.basename(vids[canon_i]),
            "n_experts": fusion_meta.get("n_experts", len(vids)),
            "align_reports": fusion_meta.get("align_reports", []),
            "has_variance_bands": True,
            "target_frames": len(fused_frames),
            "expert_videos": [os.path.basename(v) for v in vids],
            "expert_details": metas,
            "total_frames": len(fused_frames),
            "fps": round(fused_fps, 4),
            "display_video": os.path.basename(display_target),
            "display_video_frames": disp_frames,
            "display_video_fps": round(float(disp_fps), 4) if disp_frames else round(fused_fps, 4),
            "video_duration_seconds": round(duration, 4),
            "joint_names": ALL_JOINT_NAMES,
            "model_complexity": POSE_MODEL_COMPLEXITY,
            "pose_running_mode": "VIDEO",
            "pose_confidence_threshold": POSE_CONFIDENCE,
            "feature_schema": "angles+bones+variance_v1",
            "bone_names": BONE_NAMES,
        },
        "frames": fused_frames,
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Spot-check variance
    scales = []
    for fr in fused_frames:
        tol = fr.get("tolerance_scale") or {}
        scales.extend(tol.values())
    mean_tol = float(np.mean(scales)) if scales else 1.0

    elapsed = time.time() - t0
    print(f"\nSaved Phase-4 fused expert data ({len(fused_frames)} frames):")
    print(f"  {JSON_PATH}")
    print(f"  mean tolerance_scale={mean_tol:.3f} (1.0=tight, up to 2.5=loose)")
    print(f"Elapsed: {elapsed:.1f}s")
    print("\nNext:  python main.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
