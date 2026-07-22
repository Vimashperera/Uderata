"""
enrich_bones_from_display.py
----------------------------
One-shot upgrade: add torso-frame bone directions to an existing
angles-only expert JSON by running pose on the display video.

Usage (from UdarataPaSaramba folder):
  python enrich_bones_from_display.py
"""
import json
import os
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (  # noqa: E402
    JSON_PATH,
    VIDEO_PATH,
    POSE_MODEL_COMPLEXITY,
    resolve_display_video_path,
)
from core.pose_extractor import PoseExtractor  # noqa: E402
from core.motion_features import (  # noqa: E402
    BONE_NAMES,
    compute_bone_directions,
    bones_to_vector,
)

POSE_CONFIDENCE = 0.5


def _interp_fill(arr: np.ndarray) -> np.ndarray:
    """Linear-fill NaNs along a 1-D series (edges use nearest finite)."""
    out = arr.astype(np.float64).copy()
    n = len(out)
    if n == 0:
        return out
    good = np.isfinite(out)
    if not good.any():
        return np.zeros(n, dtype=np.float64)
    idx = np.arange(n)
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def resample_series(series: np.ndarray, src_len: int, target_len: int) -> np.ndarray:
    """Resample a 1-D series (with NaNs) onto target_len via linear interp."""
    filled = _interp_fill(series[:src_len])
    if target_len == src_len:
        return filled
    xs = np.linspace(0.0, 1.0, src_len)
    xt = np.linspace(0.0, 1.0, target_len)
    return np.interp(xt, xs, filled)


def main():
    video_path = resolve_display_video_path()
    if not os.path.isfile(video_path):
        # Fall back to configured VIDEO_PATH / expert_display naming
        video_path = VIDEO_PATH
    if not os.path.isfile(video_path):
        print(f"[ERROR] Display video not found:\n  {video_path}")
        sys.exit(1)
    if not os.path.isfile(JSON_PATH):
        print(f"[ERROR] Expert JSON not found:\n  {JSON_PATH}")
        sys.exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        print("[ERROR] JSON has no frames.")
        sys.exit(1)

    already = sum(1 for fr in frames if fr.get("bones"))
    if already > len(frames) // 2:
        print(f"[INFO] JSON already has bones on {already}/{len(frames)} frames. Nothing to do.")
        return

    print("=" * 60)
    print("  Enrich expert JSON with Phase-2 bone directions")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"JSON : {JSON_PATH}")
    print(f"Target frames: {len(frames)}")

    extractor = PoseExtractor(
        model_complexity=POSE_MODEL_COMPLEXITY,
        running_mode="VIDEO",
    )
    extractor.reset_sequence()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open display video.")
        sys.exit(1)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    bone_rows = []
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = extractor.process_frame(frame, fps=fps)
        world = extractor.get_world_landmarks(results)
        if world is None:
            bone_rows.append(None)
            continue
        key_ix = [11, 12, 23, 24, 25, 26, 27, 28]
        avg_vis = float(np.mean([world[i]["visibility"] for i in key_ix]))
        if avg_vis < POSE_CONFIDENCE:
            bone_rows.append(None)
            continue
        bones = compute_bone_directions(world, confidence_threshold=POSE_CONFIDENCE)
        bone_rows.append(bones)
    cap.release()
    extractor.release()

    src_len = len(bone_rows)
    if src_len < 2:
        print("[ERROR] Too few video frames.")
        sys.exit(1)

    bone_dim = len(BONE_NAMES) * 3
    mat = np.full((src_len, bone_dim), np.nan)
    for i, bones in enumerate(bone_rows):
        if bones:
            mat[i] = bones_to_vector(bones)

    target = len(frames)
    resampled = np.zeros((target, bone_dim))
    for bi in range(bone_dim):
        resampled[:, bi] = resample_series(mat[:, bi], src_len, target)

    filled = 0
    for ti, fr in enumerate(frames):
        bones_out = {}
        row = resampled[ti]
        for bi, bname in enumerate(BONE_NAMES):
            vec = row[bi * 3:(bi + 1) * 3]
            if np.all(np.isfinite(vec)):
                n = float(np.linalg.norm(vec))
                if n > 1e-8:
                    u = vec / n
                    bones_out[bname] = [
                        round(float(u[0]), 5),
                        round(float(u[1]), 5),
                        round(float(u[2]), 5),
                    ]
                    continue
            bones_out[bname] = None
        fr["bones"] = bones_out
        if any(bones_out[b] is not None for b in BONE_NAMES):
            filled += 1

    meta = data.setdefault("metadata", {})
    meta["feature_schema"] = "angles+bones_v1"
    meta["bone_names"] = BONE_NAMES
    meta["bones_source"] = os.path.basename(video_path)
    meta["bones_enriched"] = True

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Bones written for {filled}/{target} frames.")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print(f"Saved: {JSON_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
