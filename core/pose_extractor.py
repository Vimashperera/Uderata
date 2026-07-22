"""
pose_extractor.py
-----------------
MediaPipe Pose wrapper using the Tasks API (mediapipe >= 0.10).

Default running mode is VIDEO (detect_for_video + timestamps) so tracking
state is preserved across frames. Live practice and preprocess share the
same model complexity via config.POSE_MODEL_COMPLEXITY.
"""

import os
import urllib.request
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    import config as _app_config
    DEFAULT_MODEL_COMPLEXITY = int(getattr(_app_config, "POSE_MODEL_COMPLEXITY", 1))
except Exception:
    DEFAULT_MODEL_COMPLEXITY = 1

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_FILES = {
    0: ("pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    1: ("pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    2: ("pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"),
}

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
LANDMARK_INDICES = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (7, 8),
]


def _ensure_model(complexity: int) -> str:
    """Return path to the model file, downloading it if necessary."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    filename, url = MODEL_FILES.get(complexity, MODEL_FILES[1])
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.isfile(path):
        print(f"[PoseExtractor] Downloading model: {filename}")
        print(f"  From: {url}")
        print(f"  To  : {path}")
        try:
            urllib.request.urlretrieve(url, path, reporthook=_download_progress)
            print()
            print("[PoseExtractor] Model downloaded successfully.")
        except Exception as e:
            if os.path.isfile(path):
                os.remove(path)
            raise RuntimeError(
                f"Failed to download MediaPipe model.\n"
                f"  URL: {url}\n"
                f"  Error: {e}\n\n"
                f"Please download manually and place at:\n  {path}"
            ) from e
    return path


def _download_progress(count, block_size, total_size):
    pct = min(count * block_size / max(total_size, 1) * 100, 100)
    filled = int(pct / 2)
    bar = "#" * filled + "-" * (50 - filled)
    print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


class PoseExtractor:
    """
    MediaPipe Tasks PoseLandmarker wrapper.

    Prefer running_mode="VIDEO" for both offline preprocess and live webcam
    so the tracker keeps temporal state via monotonically increasing timestamps.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: Optional[int] = None,
        running_mode: str = "VIDEO",
    ):
        if model_complexity is None:
            model_complexity = DEFAULT_MODEL_COMPLEXITY

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = int(model_complexity)
        self.running_mode_name = running_mode.upper().strip()

        model_path = _ensure_model(self.model_complexity)

        BaseOptions = mp_python.BaseOptions
        PoseLandmarker = mp_vision.PoseLandmarker
        PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
        RunningMode = mp_vision.RunningMode

        if self.running_mode_name == "IMAGE":
            mode = RunningMode.IMAGE
        else:
            # Default: VIDEO (also used for live sequential webcam frames)
            mode = RunningMode.VIDEO
            self.running_mode_name = "VIDEO"

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.pose = PoseLandmarker.create_from_options(options)
        self.last_results = None
        self._timestamp_ms = 0
        self._frame_index = 0
        print(
            f"[PoseExtractor] Ready mode={self.running_mode_name} "
            f"complexity={self.model_complexity} model={os.path.basename(model_path)}"
        )

    def reset_sequence(self):
        """
        Call before a new video / practice session.

        MediaPipe PoseLandmarker keeps internal timestamp state, so simply
        zeroing our counters is not enough — recreate the landmarker so
        timestamps may restart from 0.
        """
        self._timestamp_ms = 0
        self._frame_index = 0
        try:
            if self.pose is not None:
                self.pose.close()
        except Exception:
            pass
        BaseOptions = mp_python.BaseOptions
        PoseLandmarker = mp_vision.PoseLandmarker
        PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
        RunningMode = mp_vision.RunningMode
        mode = (
            RunningMode.IMAGE
            if self.running_mode_name == "IMAGE"
            else RunningMode.VIDEO
        )
        model_path = _ensure_model(self.model_complexity)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.pose = PoseLandmarker.create_from_options(options)
        self.last_results = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
        fps: float = 30.0,
    ):
        """
        Process a BGR frame. Returns the raw PoseLandmarkerResult.

        For VIDEO mode, pass increasing timestamps (or let the extractor
        derive them from fps + frame index).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            if self.running_mode_name == "VIDEO":
                if timestamp_ms is None:
                    ts = int(round(self._frame_index * (1000.0 / max(float(fps), 1e-3))))
                    if self._frame_index > 0 and ts <= self._timestamp_ms:
                        ts = self._timestamp_ms + 1
                else:
                    ts = int(timestamp_ms)
                    if self._frame_index > 0 and ts <= self._timestamp_ms:
                        ts = self._timestamp_ms + 1
                self._timestamp_ms = ts
                results = self.pose.detect_for_video(mp_image, ts)
                self._frame_index += 1
            else:
                results = self.pose.detect(mp_image)

            self.last_results = results
            return results
        except Exception as e:
            print(f"[PoseExtractor] Inference error: {e}")
            return None

    def get_landmarks(self, results) -> Optional[List[Dict]]:
        """Normalised image-space landmarks → list of 33 dicts."""
        if results is None or not results.pose_landmarks:
            return None
        lms = results.pose_landmarks[0]
        return [{"x": lm.x, "y": lm.y, "z": lm.z,
                 "visibility": getattr(lm, "visibility", 1.0)} for lm in lms]

    def get_world_landmarks(self, results) -> Optional[List[Dict]]:
        """World-space (metric, hip-centred) landmarks → list of 33 dicts."""
        if results is None or not results.pose_world_landmarks:
            return None
        lms = results.pose_world_landmarks[0]
        return [{"x": lm.x, "y": lm.y, "z": lm.z,
                 "visibility": getattr(lm, "visibility", 1.0)} for lm in lms]

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: List[Dict],
        joint_colors: Optional[Dict[str, Tuple]] = None,
        dot_radius: int = 6,
        line_thickness: int = 3,
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        """Draw a plain skeleton on the frame (in-place)."""
        h, w = frame.shape[:2]
        default = (200, 200, 200)
        low_conf = (80, 80, 80)

        for (i, j) in POSE_CONNECTIONS:
            if i >= len(landmarks) or j >= len(landmarks):
                continue
            li, lj = landmarks[i], landmarks[j]
            if li["visibility"] < confidence_threshold or lj["visibility"] < confidence_threshold:
                color = low_conf
            else:
                color = default
            p1 = (int(li["x"] * w), int(li["y"] * h))
            p2 = (int(lj["x"] * w), int(lj["y"] * h))
            cv2.line(frame, p1, p2, color, line_thickness, cv2.LINE_AA)

        for idx, lm in enumerate(landmarks):
            cx, cy = int(lm["x"] * w), int(lm["y"] * h)
            name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else ""
            if lm["visibility"] < confidence_threshold:
                color = low_conf
            elif joint_colors and name in joint_colors:
                color = joint_colors[name]
            else:
                color = default
            cv2.circle(frame, (cx, cy), dot_radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), dot_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def draw_colored_skeleton(
        self,
        frame: np.ndarray,
        landmarks: List[Dict],
        angle_statuses: Dict[str, str],
        dot_radius: int = 6,
        line_thickness: int = 3,
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        """Draw skeleton with status-based colors per joint group."""
        STATUS_BGR = {
            "good": (136, 255, 0),
            "close": (0, 165, 255),
            "poor": (59, 59, 255),
            "unknown": (128, 128, 128),
        }
        JOINT_LANDMARK_MAP = {
            "left_knee": [23, 25, 27],
            "right_knee": [24, 26, 28],
            "left_elbow": [11, 13, 15],
            "right_elbow": [12, 14, 16],
            "left_hip": [11, 23, 25],
            "right_hip": [12, 24, 26],
            "left_shoulder": [13, 11, 12],
            "right_shoulder": [14, 12, 11],
        }
        priority = {"good": 0, "close": 1, "poor": 2, "unknown": -1}

        lm_colors: Dict[int, Tuple] = {}
        for jname, indices in JOINT_LANDMARK_MAP.items():
            status = angle_statuses.get(jname, "unknown")
            color = STATUS_BGR.get(status, STATUS_BGR["unknown"])
            for idx in indices:
                if idx not in lm_colors:
                    lm_colors[idx] = color
                else:
                    cur_status = next(
                        (s for s, c in STATUS_BGR.items() if c == lm_colors[idx]),
                        "unknown",
                    )
                    if priority.get(status, -1) > priority.get(cur_status, -1):
                        lm_colors[idx] = color

        h, w = frame.shape[:2]
        default = (200, 200, 200)
        low_conf = (80, 80, 80)

        for (i, j) in POSE_CONNECTIONS:
            if i >= len(landmarks) or j >= len(landmarks):
                continue
            li, lj = landmarks[i], landmarks[j]
            if li["visibility"] < confidence_threshold or lj["visibility"] < confidence_threshold:
                color = low_conf
            else:
                ci = lm_colors.get(i, default)
                cj = lm_colors.get(j, default)
                color = tuple((a + b) // 2 for a, b in zip(ci, cj))
            p1 = (int(li["x"] * w), int(li["y"] * h))
            p2 = (int(lj["x"] * w), int(lj["y"] * h))
            cv2.line(frame, p1, p2, color, line_thickness, cv2.LINE_AA)

        for idx, lm in enumerate(landmarks):
            cx, cy = int(lm["x"] * w), int(lm["y"] * h)
            if lm["visibility"] < confidence_threshold:
                color = low_conf
            else:
                color = lm_colors.get(idx, default)
            cv2.circle(frame, (cx, cy), dot_radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), dot_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, "pose") and self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass
            self.pose = None

    def __del__(self):
        self.release()
