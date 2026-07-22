"""
motion_features.py
------------------
Phase-2 motion representation for dance form comparison.

Builds richer features than unsigned joint angles alone:
  - Procrustes / torso-frame normalized 3D landmarks
  - Unit bone directions in the dancer's torso frame
  - Per-joint angular velocity (deg/s)
  - Hybrid feature vectors for matching + scoring

Designed so older JSON (angles-only) still works: bone terms activate
when expert frames include a "bones" dict; velocity always works from angles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from core.angle_calculator import ALL_JOINT_NAMES, landmarks_to_array

# Named bones: (name, proximal_idx, distal_idx) — MediaPipe Pose indices
BONE_DEFS: List[Tuple[str, int, int]] = [
    ("l_upper_arm", 11, 13),
    ("l_forearm", 13, 15),
    ("r_upper_arm", 12, 14),
    ("r_forearm", 14, 16),
    ("l_thigh", 23, 25),
    ("l_shin", 25, 27),
    ("r_thigh", 24, 26),
    ("r_shin", 26, 28),
    ("l_foot", 27, 31),
    ("r_foot", 28, 32),
]

BONE_NAMES = [b[0] for b in BONE_DEFS] + ["spine"]

# Blend weights for hybrid frame accuracy (sum = 1)
W_ANGLE = 0.55
W_BONE = 0.35
W_VEL = 0.10

# Scoring tolerances
BONE_GOOD_COS = 0.97   # ~14° between unit vectors
BONE_CLOSE_COS = 0.90  # ~26°
VEL_GOOD_DPS = 40.0    # deg/s
VEL_CLOSE_DPS = 90.0


def _safe_unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return v / n


def torso_frame_basis(pts: np.ndarray, vis: np.ndarray, conf: float = 0.5):
    """
    Orthonormal torso basis (right, up, forward) from shoulders/hips.
    Returns (origin_hip_mid, scale, R_3x3 with rows = basis axes) or None.
    """
    need = [11, 12, 23, 24]
    if any(i >= len(pts) or vis[i] < conf for i in need):
        return None

    hip_mid = 0.5 * (pts[23] + pts[24])
    sh_mid = 0.5 * (pts[11] + pts[12])
    up = _safe_unit(sh_mid - hip_mid)
    if up is None:
        return None

    right_raw = pts[24] - pts[23]  # left hip → right hip
    right_raw = right_raw - np.dot(right_raw, up) * up
    right = _safe_unit(right_raw)
    if right is None:
        # Fall back to shoulder width
        right_raw = pts[12] - pts[11]
        right_raw = right_raw - np.dot(right_raw, up) * up
        right = _safe_unit(right_raw)
        if right is None:
            return None

    forward = _safe_unit(np.cross(right, up))
    if forward is None:
        return None
    # Re-orthogonalize right
    right = _safe_unit(np.cross(up, forward))
    if right is None:
        return None

    scale = float(np.linalg.norm(sh_mid - hip_mid))
    if scale < 1e-6:
        # Mean limb length fallback
        lengths = []
        for _, a, b in BONE_DEFS[:8]:
            if vis[a] >= conf and vis[b] >= conf:
                lengths.append(float(np.linalg.norm(pts[b] - pts[a])))
        scale = float(np.mean(lengths)) if lengths else 1.0

    R = np.stack([right, up, forward], axis=0)  # local = R @ (p - origin) / scale
    return hip_mid, scale, R


def procrustes_normalize(
    landmarks: List[Dict],
    confidence_threshold: float = 0.5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Center at hip mid, scale by torso length, rotate into torso frame.

    Returns (local_pts (33,3), visibility (33,)) or None if torso unavailable.
    """
    if landmarks is None or len(landmarks) < 33:
        return None
    pts = landmarks_to_array(landmarks)
    vis = np.array([float(lm.get("visibility", 1.0)) for lm in landmarks], dtype=np.float64)
    basis = torso_frame_basis(pts, vis, confidence_threshold)
    if basis is None:
        return None
    origin, scale, R = basis
    local = ((pts - origin) / max(scale, 1e-6)) @ R.T
    return local, vis


def compute_bone_directions(
    landmarks: List[Dict],
    confidence_threshold: float = 0.5,
) -> Dict[str, Optional[List[float]]]:
    """
    Unit bone directions in the dancer's torso frame.
    Missing / low-confidence bones → None.
    """
    out: Dict[str, Optional[List[float]]] = {n: None for n in BONE_NAMES}
    normed = procrustes_normalize(landmarks, confidence_threshold)
    if normed is None:
        return out
    local, vis = normed

    for name, i, j in BONE_DEFS:
        if vis[i] < confidence_threshold or vis[j] < confidence_threshold:
            continue
        u = _safe_unit(local[j] - local[i])
        if u is not None:
            out[name] = [round(float(u[0]), 5), round(float(u[1]), 5), round(float(u[2]), 5)]

    # Spine: hip mid → shoulder mid in local frame
    if all(vis[i] >= confidence_threshold for i in (11, 12, 23, 24)):
        hip = 0.5 * (local[23] + local[24])
        sh = 0.5 * (local[11] + local[12])
        u = _safe_unit(sh - hip)
        if u is not None:
            out["spine"] = [round(float(u[0]), 5), round(float(u[1]), 5), round(float(u[2]), 5)]

    return out


def bones_to_vector(bones: Dict[str, Optional[List[float]]]) -> np.ndarray:
    """Flatten bone dirs to length 3*|BONE_NAMES|; missing → NaN."""
    vals = []
    for name in BONE_NAMES:
        v = bones.get(name) if bones else None
        if v is None or len(v) != 3:
            vals.extend([np.nan, np.nan, np.nan])
        else:
            vals.extend([float(v[0]), float(v[1]), float(v[2])])
    return np.array(vals, dtype=np.float64)


def compute_angular_velocity(
    current: Dict[str, Optional[float]],
    previous: Optional[Dict[str, Optional[float]]],
    dt_sec: float,
) -> Dict[str, Optional[float]]:
    """Per-joint angular velocity in deg/s. None if either sample missing."""
    dt = max(float(dt_sec), 1e-3)
    out: Dict[str, Optional[float]] = {}
    for name in ALL_JOINT_NAMES:
        c = current.get(name) if current else None
        p = previous.get(name) if previous else None
        if c is None or p is None:
            out[name] = None
        else:
            out[name] = float(c - p) / dt
    return out


def velocity_to_vector(vel: Dict[str, Optional[float]]) -> np.ndarray:
    vals = []
    for name in ALL_JOINT_NAMES:
        v = vel.get(name) if vel else None
        vals.append(float(v) if v is not None else np.nan)
    return np.array(vals, dtype=np.float64)


def angles_feature_vector(angles: Dict[str, Optional[float]]) -> np.ndarray:
    """Angles scaled to ~[0,1] with NaN for missing."""
    vals = []
    for name in ALL_JOINT_NAMES:
        v = angles.get(name) if angles else None
        vals.append(float(v) / 180.0 if v is not None else np.nan)
    return np.array(vals, dtype=np.float64)


def build_motion_feature_vector(
    angles: Dict[str, Optional[float]],
    bones: Optional[Dict[str, Optional[List[float]]]] = None,
    velocity: Optional[Dict[str, Optional[float]]] = None,
    include_bones: bool = True,
    include_velocity: bool = True,
) -> np.ndarray:
    """
    Concatenated feature vector for matching:
      [angles/180 | bone_dirs | angular_vel / 180]
    """
    parts = [angles_feature_vector(angles)]
    if include_bones:
        parts.append(bones_to_vector(bones or {}))
    if include_velocity:
        # Scale deg/s so typical motion sits near O(1)
        parts.append(velocity_to_vector(velocity or {}) / 180.0)
    return np.concatenate(parts)


def masked_cosine_bone_score(
    user_bones: Dict[str, Optional[List[float]]],
    expert_bones: Dict[str, Optional[List[float]]],
    tolerance_scale: float = 1.0,
) -> Optional[float]:
    """
    Mean cosine similarity of overlapping bones → 0–100 score.
    tolerance_scale > 1 loosens GOOD/CLOSE cosine gates (Phase-4 variance).
    Returns None if no overlapping valid bones.
    """
    scale = float(np.clip(tolerance_scale, 1.0, 2.5))
    good_cos = BONE_GOOD_COS - 0.08 * (scale - 1.0)
    close_cos = BONE_CLOSE_COS - 0.12 * (scale - 1.0)
    sims = []
    for name in BONE_NAMES:
        u = user_bones.get(name) if user_bones else None
        e = expert_bones.get(name) if expert_bones else None
        if u is None or e is None or len(u) != 3 or len(e) != 3:
            continue
        ua = np.asarray(u, dtype=np.float64)
        ea = np.asarray(e, dtype=np.float64)
        nu, ne = np.linalg.norm(ua), np.linalg.norm(ea)
        if nu < 1e-8 or ne < 1e-8:
            continue
        cos = float(np.clip(np.dot(ua, ea) / (nu * ne), -1.0, 1.0))
        sims.append(cos)
    if not sims:
        return None
    mean_cos = float(np.mean(sims))
    if mean_cos >= good_cos:
        return 100.0
    if mean_cos >= close_cos:
        t = (mean_cos - close_cos) / max(good_cos - close_cos, 1e-6)
        return 70.0 + 30.0 * t
    t = (mean_cos + 1.0) / max(close_cos + 1.0, 1e-6)
    return max(0.0, 70.0 * t)


def velocity_match_score(
    user_vel: Dict[str, Optional[float]],
    expert_vel: Dict[str, Optional[float]],
) -> Optional[float]:
    """Score how well angular velocities match (0–100)."""
    errs = []
    for name in ALL_JOINT_NAMES:
        u = user_vel.get(name) if user_vel else None
        e = expert_vel.get(name) if expert_vel else None
        if u is None or e is None:
            continue
        errs.append(abs(float(u) - float(e)))
    if not errs:
        return None
    mean_err = float(np.mean(errs))
    if mean_err <= VEL_GOOD_DPS:
        return 100.0
    if mean_err <= VEL_CLOSE_DPS:
        t = (mean_err - VEL_GOOD_DPS) / (VEL_CLOSE_DPS - VEL_GOOD_DPS)
        return 100.0 - 30.0 * t
    # degrade to 0 by ~200 deg/s
    return max(0.0, 70.0 * (1.0 - (mean_err - VEL_CLOSE_DPS) / 110.0))


def hybrid_frame_accuracy(
    angle_accuracy: float,
    bone_score: Optional[float],
    velocity_score: Optional[float],
) -> float:
    """
    Blend angle / bone / velocity scores.
    Missing optional terms redistribute their weight onto angle.
    """
    w_a, w_b, w_v = W_ANGLE, W_BONE, W_VEL
    if bone_score is None:
        w_a += w_b
        w_b = 0.0
        bone_score = 0.0
    if velocity_score is None:
        w_a += w_v
        w_v = 0.0
        velocity_score = 0.0
    total_w = w_a + w_b + w_v
    if total_w < 1e-6:
        return float(angle_accuracy)
    return float(
        (w_a * angle_accuracy + w_b * bone_score + w_v * velocity_score) / total_w
    )


def masked_feature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 over dimensions finite in both vectors."""
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("inf")
    return float(np.linalg.norm(a[mask] - b[mask]))


def expert_velocity_at(
    angle_matrix: np.ndarray,
    frame_idx: int,
    fps: float,
    joint_names: Optional[List[str]] = None,
) -> Dict[str, Optional[float]]:
    """Finite-difference angular velocity from expert angle matrix (T, J)."""
    names = joint_names or ALL_JOINT_NAMES
    dt = 1.0 / max(float(fps), 1e-3)
    n = angle_matrix.shape[0]
    i = int(np.clip(frame_idx, 0, n - 1))
    j = min(i + 1, n - 1) if i + 1 < n else max(i - 1, 0)
    out: Dict[str, Optional[float]] = {}
    for ji, name in enumerate(names):
        if ji >= angle_matrix.shape[1]:
            out[name] = None
            continue
        a0, a1 = angle_matrix[i, ji], angle_matrix[j, ji]
        if not (np.isfinite(a0) and np.isfinite(a1)) or i == j:
            out[name] = None
        else:
            sign = 1.0 if j > i else -1.0
            out[name] = float(sign * (a1 - a0) / dt)
    return out
