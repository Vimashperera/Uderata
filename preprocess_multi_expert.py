"""
preprocess_multi_expert.py
--------------------------
Phase-4 expert rebuild for multi-style steps (Udarata / Sabaragamuwa):

  1. Discover expert videos for the selected step
  2. Pick a canonical clip (prefer assets display video)
  3. Extract pose angles + torso-frame bones (shared VIDEO model)
  4. DTW-align every expert onto the canonical timeline
  5. Median-fuse + store per-joint variance / tolerance bands

Output:
  data/<style>/<step>.json
  assets/<style>/<canonical display video>  (copied if needed)
  assets/<style>/<video_stem>.wav

Usage:
  python preprocess_multi_expert.py
  python preprocess_multi_expert.py --step namaskaraya
  python preprocess_multi_expert.py --style sabaragamuwa --step all
  python preprocess_multi_expert.py --step all

Optional PowerShell:
  $env:UDARATA_EXPERT_VIDEOS = "E:\\SLIIT\\FINAL RESEARCH\\Expert data"
  $env:SABARAGAMUWA_EXPERT_VIDEOS = "E:\\SLIIT\\Uderata system\\vids\\Sabaragamuwa MP4"
  python preprocess_multi_expert.py --step namaskaraya
"""
import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (  # noqa: E402
    APP_ROOT,
    POSE_MODEL_COMPLEXITY,
    STEP_ORDER,
    STEPS,
    STYLE_ORDER,
    STYLES,
    ensure_expert_audio,
    ensure_playback_video,
    get_step,
    get_style,
    resolve_display_video_path,
    resolve_expert_source_dir,
    set_active_step,
    step_assets_dir,
    step_json_path,
    video_matches_step,
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


def discover_all_expert_videos(step_id: str) -> list:
    """Union of env/source dir, style assets/, and known research folders — filtered to step."""
    step = get_step(step_id)
    style = get_style(step["style_id"])
    roots = []
    for key in (style.get("expert_env"), "DANCE_EXPERT_VIDEOS"):
        if not key:
            continue
        env = os.environ.get(key, "").strip()
        if env:
            roots.append(env)
    roots.append(resolve_expert_source_dir(step_id))
    roots.append(step_assets_dir(step_id))
    if step["style_id"] == "udarata":
        roots.extend(_EXTRA_VIDEO_ROOTS)

    preferred_names = {n.lower() for n in step["video_candidates"]}

    by_size = {}  # size -> abspath (prefer teaching clip names)
    for root in roots:
        for p in discover_videos(root):
            if not video_matches_step(p, step_id):
                continue
            ap = os.path.abspath(p)
            try:
                size_sig = os.path.getsize(ap)
            except OSError:
                continue
            base = os.path.basename(ap).lower()
            if size_sig not in by_size:
                by_size[size_sig] = ap
                continue
            cur = os.path.basename(by_size[size_sig]).lower()
            if base in preferred_names and cur not in preferred_names:
                by_size[size_sig] = ap

    return sorted(by_size.values())


def choose_canonical(vids: list, step_id: str) -> int:
    """Prefer the assets teaching clip for this step as the canonical timeline."""
    display = os.path.abspath(resolve_display_video_path(step_id))
    for i, p in enumerate(vids):
        if os.path.abspath(p) == display:
            return i
    step = get_step(step_id)
    ranked = [n.lower() for n in step["video_candidates"]]
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


def preprocess_step(step_id: str) -> bool:
    step = set_active_step(step_id)
    style = get_style(step["style_id"])
    json_path = step_json_path(step_id)
    assets_dir = step_assets_dir(step_id)
    title = step["title"]

    print("=" * 70)
    print(f"  {style['title']} — {title} — Phase-4 Canonical Expert Rebuild")
    print("=" * 70)

    source_dir = resolve_expert_source_dir(step_id)
    print(f"\nApp root:\n  {APP_ROOT}")
    print(f"Style / step:\n  {step['style_id']} / {step_id}")
    print(f"Primary expert folder:\n  {source_dir}")
    print(f"  Exists: {os.path.isdir(source_dir)}")

    vids = discover_all_expert_videos(step_id)
    if not vids:
        env_hint = style.get("expert_env") or "DANCE_EXPERT_VIDEOS"
        print(f"\n[ERROR] No expert videos found for step '{step_id}'.")
        print(
            f"Place a clip like '{step['video_candidates'][0]}' in "
            f"assets/{step['style_id']}/, or set:\n"
            f'  $env:{env_hint} = \"…\\path\\to\\expert videos\"\n'
            f"  python preprocess_multi_expert.py --step {step_id}\n"
        )
        return False

    canon_i = choose_canonical(vids, step_id)
    print(f"\nFound {len(vids)} expert video(s):")
    for i, p in enumerate(vids):
        mark = " [CANONICAL]" if i == canon_i else ""
        print(f"  • {p}{mark}")
    print()

    try:
        import mediapipe as mp  # noqa: F401
    except ImportError:
        print("[ERROR] mediapipe not installed. Run: pip install -r requirements.txt")
        return False

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

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    display_target = os.path.join(assets_dir, step["video_candidates"][0])
    try:
        src = vids[canon_i]
        if not os.path.isfile(display_target):
            if os.path.basename(src).lower() == os.path.basename(display_target).lower():
                shutil.copy2(src, display_target)
                print(f"\nSeeded assets teaching video:\n  {display_target}")
            else:
                # Copy canonical into the expected teaching name when absent
                shutil.copy2(src, display_target)
                print(f"\nSeeded assets teaching video from canonical:\n  {display_target}")
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

    wav_path = ensure_expert_audio(display_target, force=True)
    if wav_path:
        print(f"Reference audio (WAV):\n  {wav_path}")
    else:
        print(
            "\n[INFO] Could not extract reference WAV "
            "(install imageio-ffmpeg or put ffmpeg on PATH)."
        )

    playback = ensure_playback_video(display_target)
    if playback != display_target:
        print(f"Playback proxy (UI):\n  {playback}")

    disp_cap = cv2.VideoCapture(display_target)
    disp_frames = int(disp_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if disp_cap.isOpened() else 0
    disp_fps = (
        disp_cap.get(cv2.CAP_PROP_FPS) or fused_fps if disp_cap.isOpened() else fused_fps
    )
    disp_cap.release()

    duration = len(fused_frames) / max(fused_fps, 1e-6)
    if disp_frames > 0 and disp_fps > 0:
        duration = disp_frames / disp_fps

    output_data = {
        "metadata": {
            "dance_style": style["title"],
            "style_id": step["style_id"],
            "step_id": step_id,
            "step_name": title,
            "reference_loops": int(step.get("reference_loops", 3)),
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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    scales = []
    for fr in fused_frames:
        tol = fr.get("tolerance_scale") or {}
        scales.extend(tol.values())
    mean_tol = float(np.mean(scales)) if scales else 1.0

    elapsed = time.time() - t0
    print(f"\nSaved Phase-4 fused expert data ({len(fused_frames)} frames):")
    print(f"  {json_path}")
    print(f"  mean tolerance_scale={mean_tol:.3f} (1.0=tight, up to 2.5=loose)")
    print(f"  practice loops: {step.get('reference_loops', 3)}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 70)
    return True


def _step_ids_for(style_id: str | None, step_arg: str) -> list:
    if style_id:
        if style_id not in STYLES:
            print(
                f"[ERROR] Unknown style '{style_id}'. Known: {', '.join(STYLE_ORDER)}"
            )
            sys.exit(1)
        style_steps = list(STYLES[style_id]["step_order"])
    else:
        style_steps = list(STEP_ORDER)

    if step_arg.lower() == "all":
        return style_steps

    if step_arg not in STEPS:
        print(f"[ERROR] Unknown step '{step_arg}'. Known: {', '.join(STEP_ORDER)}")
        sys.exit(1)
    if style_id and STEPS[step_arg]["style_id"] != style_id:
        print(
            f"[ERROR] Step '{step_arg}' belongs to style "
            f"'{STEPS[step_arg]['style_id']}', not '{style_id}'."
        )
        sys.exit(1)
    return [step_arg]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build fused expert pose JSON for a step / style."
    )
    parser.add_argument(
        "--style",
        default=None,
        help=(
            "Optional style filter: udarata | sabaragamuwa. "
            f"Known: {', '.join(STYLE_ORDER)}"
        ),
    )
    parser.add_argument(
        "--step",
        default="pa_saramba_01",
        help=(
            "Step id to process, or 'all'. "
            f"Known: {', '.join(STEP_ORDER)}. Default: pa_saramba_01"
        ),
    )
    args = parser.parse_args(argv)
    ids = _step_ids_for(args.style, args.step)

    ok_any = False
    for sid in ids:
        if preprocess_step(sid):
            ok_any = True
        print()

    if not ok_any:
        sys.exit(1)
    print("Next:  python main.py")


if __name__ == "__main__":
    main()
