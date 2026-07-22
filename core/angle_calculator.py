"""
angle_calculator.py
-------------------
Scale-invariant 3D joint angle computation using vector dot products.
All angles computed from MediaPipe world landmarks (metric space).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Joint triplets: (vertex_idx, point_a_idx, point_b_idx) ────────────────────
# Angle is measured AT the vertex between the vectors vertex→a and vertex→b.
JOINT_TRIPLETS = {
    "left_knee":      (25, 23, 27),   # Left Knee (Hip → Knee → Ankle)
    "right_knee":     (26, 24, 28),   # Right Knee
    "left_elbow":     (13, 11, 15),   # Left Elbow (Shoulder → Elbow → Wrist)
    "right_elbow":    (14, 12, 16),   # Right Elbow
    "left_hip":       (23, 11, 25),   # Left Hip (Shoulder → Hip → Knee)
    "right_hip":      (24, 12, 26),   # Right Hip
    "left_shoulder":  (11, 13, 12),   # Left Shoulder (Elbow → Shoulder → R.Shoulder)
    "right_shoulder": (12, 14, 11),   # Right Shoulder (Elbow → Shoulder → L.Shoulder)
}

# Landmark indices used for spine tilt
SPINE_LANDMARKS = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
}

# All joint names this module produces
ALL_JOINT_NAMES = list(JOINT_TRIPLETS.keys()) + ["spine_tilt"]

# Pa Saramba 01 — joint importance weights (sum = 1.0)
JOINT_WEIGHTS: Dict[str, float] = {
    "left_knee":       0.18,
    "right_knee":      0.18,
    "left_hip":        0.12,
    "right_hip":       0.12,
    "left_shoulder":   0.10,
    "right_shoulder":  0.10,
    "left_elbow":      0.08,
    "right_elbow":     0.08,
    "spine_tilt":      0.04,
}

# Joints calibrated during countdown (camera / stance bias).
# Only spine_tilt is safe to treat as ~0° when standing upright.
# Hip angles are ~170° at rest — using absolute values as offsets broke scoring.
CALIBRATION_JOINTS = {"spine_tilt"}


def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return unit vector from point a to point b."""
    v = b - a
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros(3)
    return v / norm


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the angle in degrees between two 3D vectors.
    Uses arccos of the dot product (vectors need not be unit).

    Returns:
        Angle in degrees in [0, 180], or 0.0 if vectors are degenerate.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)   # Guard against floating-point error
    return float(np.degrees(np.arccos(cos_theta)))


def landmarks_to_array(landmarks: List[Dict]) -> np.ndarray:
    """
    Convert a list of 33 landmark dicts to a (33, 3) numpy array.

    Args:
        landmarks: List of dicts with keys x, y, z (world coordinates).

    Returns:
        np.ndarray of shape (33, 3).
    """
    pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks],
                   dtype=np.float64)
    return pts


def compute_joint_angles(
    landmarks: List[Dict],
    confidence_threshold: float = 0.5,
    use_world: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Compute all 9 joint angles from pose landmarks.

    Args:
        landmarks: List of 33 landmark dicts (from PoseExtractor.get_world_landmarks()
                   or get_landmarks()). Must contain x, y, z, visibility.
        confidence_threshold: Landmarks below this visibility are treated as missing.
        use_world: If True, assume coordinates are already in world space.

    Returns:
        Dict mapping joint_name -> angle in degrees, or None if landmarks
        were not detected with sufficient confidence.
    """
    if landmarks is None or len(landmarks) < 33:
        return {name: None for name in ALL_JOINT_NAMES}

    pts = landmarks_to_array(landmarks)
    vis = np.array([lm["visibility"] for lm in landmarks])

    angles: Dict[str, Optional[float]] = {}

    # ── Standard joint triplets ──────────────────────────────────────────────
    for joint_name, (vertex, pt_a, pt_b) in JOINT_TRIPLETS.items():
        if (vis[vertex] < confidence_threshold or
                vis[pt_a] < confidence_threshold or
                vis[pt_b] < confidence_threshold):
            angles[joint_name] = None
            continue

        vec_a = pts[pt_a] - pts[vertex]
        vec_b = pts[pt_b] - pts[vertex]
        angles[joint_name] = angle_between_vectors(vec_a, vec_b)

    # ── Spine tilt: angle between shoulder midpoint-hip midpoint vs vertical ─
    ls_idx = SPINE_LANDMARKS["left_shoulder"]
    rs_idx = SPINE_LANDMARKS["right_shoulder"]
    lh_idx = SPINE_LANDMARKS["left_hip"]
    rh_idx = SPINE_LANDMARKS["right_hip"]

    if all(vis[i] >= confidence_threshold for i in [ls_idx, rs_idx, lh_idx, rh_idx]):
        shoulder_mid = (pts[ls_idx] + pts[rs_idx]) / 2.0
        hip_mid      = (pts[lh_idx] + pts[rh_idx]) / 2.0
        spine_vec    = shoulder_mid - hip_mid

        # MediaPipe world landmarks: +Y is downward, so vertical up = [0, -1, 0]
        vertical = np.array([0.0, -1.0, 0.0])
        angles["spine_tilt"] = angle_between_vectors(spine_vec, vertical)
    else:
        angles["spine_tilt"] = None

    return angles


def angles_to_vector(angles: Dict[str, Optional[float]]) -> np.ndarray:
    """
    Convert an angles dict to a fixed-length numpy vector.
    Missing angles (None) become NaN so callers can mask them out of distances.

    Returns:
        np.ndarray of shape (len(ALL_JOINT_NAMES),)
    """
    vals = []
    for name in ALL_JOINT_NAMES:
        v = angles.get(name)
        vals.append(float(v) if v is not None else np.nan)
    return np.array(vals, dtype=np.float64)


def masked_angle_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean L2 distance using only joints finite in both vectors.
    Returns +inf if no overlapping valid joints.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("inf")
    diff = a[mask] - b[mask]
    return float(np.linalg.norm(diff))


def compute_joint_deviations(
    user_angles: Dict[str, Optional[float]],
    expert_angles: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    """
    Compute absolute deviation (in degrees) between user and expert for each joint.

    Args:
        user_angles:   Dict from compute_joint_angles() for the user frame.
        expert_angles: Dict from compute_joint_angles() for the expert frame.

    Returns:
        Dict mapping joint_name -> deviation in degrees, or None if either
        angle is unavailable.
    """
    deviations: Dict[str, Optional[float]] = {}
    for name in ALL_JOINT_NAMES:
        u = user_angles.get(name)
        e = expert_angles.get(name)
        if u is None or e is None:
            deviations[name] = None
        else:
            deviations[name] = abs(u - e)
    return deviations


def classify_deviation(
    deviation: Optional[float],
    good_thresh: float = 8.0,
    close_thresh: float = 15.0,
) -> str:
    """
    Classify a joint deviation into a status string.

    Returns:
        'good'    → deviation ≤ good_thresh
        'close'   → deviation ≤ close_thresh
        'poor'    → deviation > close_thresh
        'unknown' → deviation is None
    """
    if deviation is None:
        return "unknown"
    if deviation <= good_thresh:
        return "good"
    if deviation <= close_thresh:
        return "close"
    return "poor"


def joint_score_from_deviation(
    deviation: float,
    good_thresh: float = 8.0,
    close_thresh: float = 15.0,
) -> float:
    """
    Piecewise linear tiered scoring with configurable thresholds
    (Phase-4: scale thresholds by expert variance tolerance).
    """
    d = max(0.0, float(deviation))
    good = max(1.0, float(good_thresh))
    close = max(good + 1e-3, float(close_thresh))
    poor_end = close + 15.0
    zero_end = poor_end + 20.0

    if d <= good:
        return 100.0
    elif d <= close:
        return 100.0 - ((d - good) / (close - good)) * 30.0      # 100 → 70
    elif d <= poor_end:
        return 70.0 - ((d - close) / (poor_end - close)) * 40.0  # 70 → 30
    elif d <= zero_end:
        return 30.0 - ((d - poor_end) / (zero_end - poor_end)) * 30.0  # 30 → 0
    else:
        return 0.0


def thresholds_from_scale(scale: float = 1.0) -> Tuple[float, float]:
    """Map Phase-4 tolerance_scale → (good°, close°) thresholds."""
    s = float(np.clip(scale, 1.0, 2.5))
    return 8.0 * s, 15.0 * s


def compute_frame_accuracy(
    deviations: Dict[str, Optional[float]],
    joint_scores_override: Optional[Dict[str, Optional[float]]] = None,
) -> float:
    """
    Weighted, tiered frame accuracy.

    Joints with missing deviation, or override score of None (low confidence /
    unknown), are excluded from the weighted average — never treated as free points.
    """
    total_score = 0.0
    total_weight = 0.0

    for name in ALL_JOINT_NAMES:
        if joint_scores_override is not None and name in joint_scores_override:
            score = joint_scores_override[name]
            if score is None:
                continue
        else:
            dev = deviations.get(name)
            if dev is None:
                continue
            score = joint_score_from_deviation(dev)

        w = JOINT_WEIGHTS.get(name, 1.0 / len(ALL_JOINT_NAMES))
        total_score += float(score) * w
        total_weight += w

    if total_weight < 1e-6:
        return 0.0
    return float(min(100.0, total_score / total_weight))


def get_worst_joints(
    deviations: Dict[str, Optional[float]],
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """
    Return the top N joints with the highest deviation.
    """
    valid = [(name, dev) for name, dev in deviations.items() if dev is not None]
    valid.sort(key=lambda x: x[1], reverse=True)
    return valid[:top_n]


def generate_feedback_message(
    deviations: Dict[str, Optional[float]],
    accuracy: float,
) -> str:
    """
    Generate a human-readable feedback message based on worst-performing joints.
    """
    if accuracy >= 85.0:
        return "✨ Great form! Maintain this posture."

    worst = get_worst_joints(deviations, top_n=1)
    if not worst:
        return "Align your body with the expert pose."

    joint_name, dev = worst[0]
    dev_int = int(round(dev))

    MESSAGES = {
        "left_knee":      f"Bend your left knee deeper — currently {dev_int}° off.",
        "right_knee":     f"Bend your right knee deeper — currently {dev_int}° off.",
        "left_elbow":     f"Extend your left arm further — {dev_int}° deviation detected.",
        "right_elbow":    f"Extend your right arm further — {dev_int}° deviation detected.",
        "left_hip":       f"Adjust your left hip angle — {dev_int}° off from expert.",
        "right_hip":      f"Adjust your right hip angle — {dev_int}° off from expert.",
        "left_shoulder":  f"Open your left shoulder wider — {dev_int}° deviation.",
        "right_shoulder": f"Open your right shoulder wider — {dev_int}° deviation.",
        "spine_tilt":     f"Keep your torso more upright — {dev_int}° tilt deviation.",
    }

    return MESSAGES.get(joint_name, f"Adjust your {joint_name.replace('_', ' ')} — {dev_int}° off.")


# Human-readable names for UI display
JOINT_DISPLAY_NAMES = {
    "left_knee":      "Left Knee Bend",
    "right_knee":     "Right Knee Bend",
    "left_elbow":     "Left Elbow Flex",
    "right_elbow":    "Right Elbow Flex",
    "left_hip":       "Left Hip Angle",
    "right_hip":      "Right Hip Angle",
    "left_shoulder":  "Left Shoulder",
    "right_shoulder": "Right Shoulder",
    "spine_tilt":     "Spine / Torso Tilt",
}

# Unicode icons for report display
JOINT_ICONS = {
    "left_knee":      "🦵",
    "right_knee":     "🦵",
    "left_elbow":     "🦾",
    "right_elbow":    "🦾",
    "left_hip":       "🧍",
    "right_hip":      "🧍",
    "left_shoulder":  "🤸",
    "right_shoulder": "🤸",
    "spine_tilt":     "🧍",
}

# Corrective instructions for the report screen
CORRECTIVE_INSTRUCTIONS = {
    "left_knee":      "Bend your left knee more deeply during the squat phase. Aim for a 90–120° angle.",
    "right_knee":     "Bend your right knee more deeply during the squat phase. Aim for a 90–120° angle.",
    "left_elbow":     "Extend your left arm fully during arm movements. Keep the elbow soft, not locked.",
    "right_elbow":    "Extend your right arm fully during arm movements. Keep the elbow soft, not locked.",
    "left_hip":       "Shift your left hip lower and maintain pelvic tilt consistent with the expert.",
    "right_hip":      "Shift your right hip lower and maintain pelvic tilt consistent with the expert.",
    "left_shoulder":  "Raise and open your left shoulder to match the expert's arm plane.",
    "right_shoulder": "Raise and open your right shoulder to match the expert's arm plane.",
    "spine_tilt":     "Keep your spine more upright. Avoid leaning too far forward or backward.",
}
