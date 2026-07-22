"""
expert_fusion.py
----------------
Phase-4 expert rebuild: DTW-align every expert to a canonical timeline,
then median-fuse with per-joint variance bands for adaptive scoring.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from core.angle_calculator import ALL_JOINT_NAMES, angles_to_vector
from core.motion_features import BONE_NAMES, bones_to_vector
from core.soft_dtw import dtw_path


def records_to_angle_matrix(recs: list) -> np.ndarray:
    """(T, J) angle matrix with NaN for missing."""
    T = len(recs)
    mat = np.full((T, len(ALL_JOINT_NAMES)), np.nan, dtype=np.float64)
    for ti, rec in enumerate(recs):
        ang = rec.get("angles") or {}
        for ji, jn in enumerate(ALL_JOINT_NAMES):
            v = ang.get(jn)
            if v is not None:
                mat[ti, ji] = float(v)
    return mat


def records_to_bone_matrix(recs: list) -> np.ndarray:
    """(T, 3*|bones|) bone direction matrix with NaN for missing."""
    dim = len(BONE_NAMES) * 3
    T = len(recs)
    mat = np.full((T, dim), np.nan, dtype=np.float64)
    for ti, rec in enumerate(recs):
        bones = rec.get("bones")
        if bones:
            mat[ti] = bones_to_vector(bones)
    return mat


def _angle_feat_matrix(angle_mat: np.ndarray) -> np.ndarray:
    """Scale angles to ~[0,1] for DTW; NaN → column median fill."""
    X = angle_mat / 180.0
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isfinite(col)
        if m.any():
            fill = float(np.nanmedian(col[m]))
            col = col.copy()
            col[~m] = fill
            X[:, j] = col
        else:
            X[:, j] = 0.0
    return X


def dtw_align_indices(src_angles: np.ndarray, canon_angles: np.ndarray) -> np.ndarray:
    """
    For each canonical frame j, return the source frame index i that best
    aligns to it (from open-end DTW path; gaps filled by interpolation).
    """
    Xs = _angle_feat_matrix(src_angles)
    Xc = _angle_feat_matrix(canon_angles)
    _, path = dtw_path(Xs, Xc, open_end=True)  # path: (src_i, canon_j)

    Tc = canon_angles.shape[0]
    buckets: List[List[int]] = [[] for _ in range(Tc)]
    for i, j in path:
        if 0 <= j < Tc:
            buckets[j].append(int(i))

    aligned = np.full(Tc, -1, dtype=np.int32)
    for j, idxs in enumerate(buckets):
        if idxs:
            aligned[j] = int(np.median(idxs))

    # Fill gaps by nearest valid / linear interpolate indices
    known = np.where(aligned >= 0)[0]
    if known.size == 0:
        # Degenerate: linear map
        Ts = max(src_angles.shape[0] - 1, 1)
        return np.clip(
            np.round(np.linspace(0, Ts, Tc)).astype(np.int32),
            0,
            src_angles.shape[0] - 1,
        )

    for j in range(Tc):
        if aligned[j] >= 0:
            continue
        # nearest known
        k = known[np.argmin(np.abs(known - j))]
        aligned[j] = aligned[k]

    # Smooth monotone-ish: ensure non-decreasing where possible
    for j in range(1, Tc):
        if aligned[j] < aligned[j - 1]:
            aligned[j] = aligned[j - 1]

    return np.clip(aligned, 0, src_angles.shape[0] - 1)


def warp_matrix_to_canonical(src: np.ndarray, index_map: np.ndarray) -> np.ndarray:
    """Gather src rows by index_map → (Tc, D)."""
    return src[index_map]


def _fill_nan_rows(mat: np.ndarray) -> np.ndarray:
    """Forward/backward fill NaN rows along time."""
    out = mat.copy()
    T, D = out.shape
    for j in range(D):
        col = out[:, j]
        good = np.isfinite(col)
        if not good.any():
            out[:, j] = 0.0
            continue
        idx = np.arange(T)
        out[:, j] = np.interp(idx, idx[good], col[good])
    return out


def fuse_aligned_stack(
    stacked: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    stacked: (E, T, D) → (median, nanstd) along expert axis.
    """
    med = np.nanmedian(stacked, axis=0)
    # ddof=0; with E=1 std is 0
    with np.errstate(all="ignore"):
        std = np.nanstd(stacked, axis=0)
    std = np.where(np.isfinite(std), std, 0.0)
    med = np.where(np.isfinite(med), med, np.nan)
    return med, std


def tolerance_scale_from_std(std_deg: float, base: float = 8.0) -> float:
    """
    Map expert disagreement (deg std) → multiplier on score thresholds.
    Agree tightly → ~1.0; disagree a lot → up to ~2.5.
    """
    s = max(0.0, float(std_deg))
    # 0° std → 1.0; 8° std → 2.0; cap 2.5
    return float(np.clip(1.0 + s / base, 1.0, 2.5))


def renorm_bone_row(row: np.ndarray) -> Dict[str, Optional[List[float]]]:
    out: Dict[str, Optional[List[float]]] = {}
    for bi, bname in enumerate(BONE_NAMES):
        vec = row[bi * 3:(bi + 1) * 3]
        if np.all(np.isfinite(vec)):
            n = float(np.linalg.norm(vec))
            if n > 1e-8:
                u = vec / n
                out[bname] = [
                    round(float(u[0]), 5),
                    round(float(u[1]), 5),
                    round(float(u[2]), 5),
                ]
                continue
        out[bname] = None
    return out


def fuse_experts_canonical(
    all_series: list,
    canonical_index: int = 0,
) -> Tuple[list, dict]:
    """
    DTW-align every expert onto the canonical expert's timeline, then
    median-fuse angles/bones and compute per-joint std / tolerance scales.

    Returns (frames_out, fusion_meta).
    """
    if not all_series:
        return [], {}

    canonical_index = int(np.clip(canonical_index, 0, len(all_series) - 1))
    canon = all_series[canonical_index]
    canon_ang = records_to_angle_matrix(canon)
    canon_bones = records_to_bone_matrix(canon)
    Tc = canon_ang.shape[0]

    warped_angles = []
    warped_bones = []
    align_reports = []

    for ei, recs in enumerate(all_series):
        src_ang = records_to_angle_matrix(recs)
        src_bones = records_to_bone_matrix(recs)
        if ei == canonical_index:
            idx_map = np.arange(Tc, dtype=np.int32)
            path_len = Tc
        else:
            idx_map = dtw_align_indices(src_ang, canon_ang)
            path_len = int(len(idx_map))

        wa = warp_matrix_to_canonical(src_ang, idx_map)
        wb = warp_matrix_to_canonical(src_bones, idx_map)
        warped_angles.append(wa)
        warped_bones.append(wb)
        align_reports.append({
            "expert_index": ei,
            "src_frames": int(src_ang.shape[0]),
            "canonical": bool(ei == canonical_index),
            "map_frames": path_len,
        })

    ang_stack = np.stack(warped_angles, axis=0)  # (E, T, J)
    bone_stack = np.stack(warped_bones, axis=0)

    ang_med, ang_std = fuse_aligned_stack(ang_stack)
    bone_med, bone_std = fuse_aligned_stack(bone_stack)
    ang_med = _fill_nan_rows(ang_med)
    bone_med = _fill_nan_rows(bone_med)

    # Single-expert: estimate mild temporal local variance as prior
    if ang_stack.shape[0] == 1:
        local = np.zeros_like(ang_med)
        for t in range(Tc):
            lo, hi = max(0, t - 2), min(Tc, t + 3)
            local[t] = np.nanstd(ang_med[lo:hi], axis=0)
        ang_std = np.maximum(ang_std, local * 0.5)

    frames_out = []
    for ti in range(Tc):
        ad = {}
        std_d = {}
        tol = {}
        for ji, jn in enumerate(ALL_JOINT_NAMES):
            val = ang_med[ti, ji]
            ad[jn] = round(float(val), 4) if np.isfinite(val) else None
            s = float(ang_std[ti, ji]) if np.isfinite(ang_std[ti, ji]) else 0.0
            std_d[jn] = round(s, 4)
            tol[jn] = round(tolerance_scale_from_std(s), 4)

        bones_out = renorm_bone_row(bone_med[ti])
        # Mean bone component std as a simple bone looseness cue
        bstd = bone_std[ti]
        bone_loose = float(np.nanmean(bstd)) if np.isfinite(bstd).any() else 0.0

        frames_out.append({
            "frame": ti,
            "angles": ad,
            "bones": bones_out,
            "angle_std": std_d,
            "tolerance_scale": tol,
            "bone_std_mean": round(bone_loose, 5),
            "pose_detected": any(ad[jn] is not None for jn in ALL_JOINT_NAMES),
        })

    meta = {
        "fusion": "canonical_dtw_median",
        "canonical_index": canonical_index,
        "canonical_frames": Tc,
        "n_experts": len(all_series),
        "align_reports": align_reports,
        "has_variance_bands": True,
    }
    return frames_out, meta
