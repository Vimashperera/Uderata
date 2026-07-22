"""
dtw_engine.py
-------------
Expert pose data loader and nearest-frame matching against the fused
reference sequence.

Supports Phase-1 angles (NaN-masked) and Phase-2 motion features
(bones + velocity) when present in the JSON.
"""

import json
from typing import List, Dict, Optional, Tuple
import numpy as np

from core.angle_calculator import ALL_JOINT_NAMES, angles_to_vector, masked_angle_distance
from core.motion_features import (
    build_motion_feature_vector,
    bones_to_vector,
    expert_velocity_at,
    masked_feature_distance,
)


class ExpertDataLoader:
    """Loads and manages the pre-processed expert pose data."""

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.frames: List[Dict] = []
        self.metadata: Dict = {}
        self.angle_matrix: Optional[np.ndarray] = None  # (N, J) NaN = missing
        self.bone_matrix: Optional[np.ndarray] = None   # (N, 3*B) or None
        self.feature_matrix: Optional[np.ndarray] = None  # (N, D) Soft-DTW features
        self.has_bones: bool = False
        self._loaded = False

    def load(self) -> bool:
        """Load expert data from JSON file."""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.metadata = data.get("metadata", {})
            self.frames = data.get("frames", [])

            if not self.frames:
                print("[ExpertDataLoader] No frames found in JSON.")
                return False

            angle_vecs = []
            bone_vecs = []
            bone_count = 0
            for frame_data in self.frames:
                angles_dict = frame_data.get("angles", {})
                angle_vecs.append(angles_to_vector(angles_dict))
                bones = frame_data.get("bones")
                if bones:
                    bone_count += 1
                    bone_vecs.append(bones_to_vector(bones))
                else:
                    bone_vecs.append(None)

            self.angle_matrix = np.array(angle_vecs, dtype=np.float64)
            self.has_bones = bone_count > max(3, len(self.frames) // 10)
            if self.has_bones:
                # Fill missing bone rows with NaN of correct width
                width = next(v.shape[0] for v in bone_vecs if v is not None)
                mat = np.full((len(bone_vecs), width), np.nan, dtype=np.float64)
                for i, v in enumerate(bone_vecs):
                    if v is not None:
                        mat[i] = v
                self.bone_matrix = mat
            else:
                self.bone_matrix = None

            # Phase-3: precompute full motion feature matrix for Soft-DTW
            self.feature_matrix = None
            try:
                feats = [self.feature_vector_at(i) for i in range(len(self.frames))]
                # Pad to common width
                width = max(f.shape[0] for f in feats)
                fm = np.full((len(feats), width), np.nan, dtype=np.float64)
                for i, f in enumerate(feats):
                    fm[i, : f.shape[0]] = f
                self.feature_matrix = fm
            except Exception as e:
                print(f"[ExpertDataLoader] feature_matrix build failed: {e}")
                self.feature_matrix = None

            self._loaded = True
            print(
                f"[ExpertDataLoader] Loaded {len(self.frames)} expert frames "
                f"(bones={'yes' if self.has_bones else 'no — angles+velocity only'}, "
                f"features={'yes' if self.feature_matrix is not None else 'no'})."
            )
            return True

        except FileNotFoundError:
            print(f"[ExpertDataLoader] File not found: {self.json_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"[ExpertDataLoader] JSON parse error: {e}")
            return False
        except Exception as e:
            print(f"[ExpertDataLoader] Unexpected error: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        return float(self.metadata.get("fps", 30.0))

    @property
    def duration_seconds(self) -> float:
        return float(self.metadata.get("video_duration_seconds", 0.0))

    def get_frame_angles(self, frame_idx: int) -> Optional[Dict]:
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx].get("angles", {})
        return None

    def get_frame_bones(self, frame_idx: int) -> Optional[Dict]:
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx].get("bones")
        return None

    def get_frame_tolerance(self, frame_idx: int) -> Dict[str, float]:
        """Per-joint tolerance_scale for adaptive scoring (default 1.0)."""
        if 0 <= frame_idx < len(self.frames):
            tol = self.frames[frame_idx].get("tolerance_scale") or {}
            return {n: float(tol.get(n, 1.0)) for n in ALL_JOINT_NAMES}
        return {n: 1.0 for n in ALL_JOINT_NAMES}

    def get_frame_bone_std_mean(self, frame_idx: int) -> float:
        """Mean bone-component std at frame (0 if absent)."""
        if 0 <= frame_idx < len(self.frames):
            v = self.frames[frame_idx].get("bone_std_mean")
            return float(v) if v is not None else 0.0
        return 0.0

    def get_nearest_expert_angles(self, user_frame_count: int) -> Optional[Dict]:
        if not self._loaded or not self.frames:
            return None
        idx = user_frame_count % len(self.frames)
        return self.frames[idx].get("angles", {})

    def get_angles_for_video_sync(
        self, video_frame_idx: int, video_total_frames: int
    ) -> Optional[Dict]:
        idx = self.sync_index(video_frame_idx, video_total_frames)
        if idx is None:
            return None
        return self.frames[idx].get("angles", {})

    def get_bones_for_video_sync(
        self, video_frame_idx: int, video_total_frames: int
    ) -> Optional[Dict]:
        idx = self.sync_index(video_frame_idx, video_total_frames)
        if idx is None:
            return None
        return self.frames[idx].get("bones")

    def sync_index(self, video_frame_idx: int, video_total_frames: int) -> Optional[int]:
        if not self._loaded or not self.frames:
            return None
        n = len(self.frames)
        vt = max(int(video_total_frames), 1)
        if n <= 1:
            return 0
        vf = min(max(int(video_frame_idx), 0), vt - 1)
        if vt <= 1:
            return 0
        idx = int(round(vf / (vt - 1) * (n - 1)))
        return max(0, min(n - 1, idx))

    def get_velocity_at(self, frame_idx: int) -> Dict[str, Optional[float]]:
        if self.angle_matrix is None:
            return {n: None for n in ALL_JOINT_NAMES}
        return expert_velocity_at(self.angle_matrix, frame_idx, self.fps)

    def feature_vector_at(
        self,
        frame_idx: int,
        user_velocity: Optional[Dict[str, Optional[float]]] = None,
    ) -> np.ndarray:
        """Expert motion feature vector at frame (for hybrid matching)."""
        angles = self.get_frame_angles(frame_idx) or {}
        bones = self.get_frame_bones(frame_idx) if self.has_bones else None
        vel = self.get_velocity_at(frame_idx)
        return build_motion_feature_vector(
            angles,
            bones=bones,
            velocity=vel,
            include_bones=self.has_bones,
            include_velocity=True,
        )

    def get_best_matching_expert_frame(
        self,
        user_angles: Dict[str, Optional[float]],
        last_best_idx: int = -1,
        window: int = 45,
        user_bones: Optional[Dict] = None,
        user_velocity: Optional[Dict[str, Optional[float]]] = None,
    ) -> Tuple[int, Dict]:
        """
        Find expert frame with minimum masked feature distance.
        Falls back to angle-only distance when Phase-2 features unavailable.
        """
        if not self._loaded or self.angle_matrix is None:
            return 0, {}

        n = len(self.frames)
        if last_best_idx < 0:
            lo, hi = 0, n
        else:
            lo = max(0, last_best_idx - window)
            hi = min(n, last_best_idx + window + 1)

        use_hybrid = user_velocity is not None or (self.has_bones and user_bones)
        if use_hybrid:
            user_vec = build_motion_feature_vector(
                user_angles,
                bones=user_bones if self.has_bones else None,
                velocity=user_velocity,
                include_bones=self.has_bones,
                include_velocity=user_velocity is not None,
            )
            dists = []
            for i in range(lo, hi):
                exp_vec = self.feature_vector_at(i)
                # Align lengths if velocity inclusion differs
                m = min(len(user_vec), len(exp_vec))
                dists.append(masked_feature_distance(user_vec[:m], exp_vec[:m]))
            dists = np.asarray(dists, dtype=np.float64)
        else:
            user_vec = angles_to_vector(user_angles)
            if not np.isfinite(user_vec).any():
                fallback = max(0, last_best_idx) if last_best_idx >= 0 else 0
                return fallback, self.frames[fallback].get("angles", {}) if self.frames else {}
            window_matrix = self.angle_matrix[lo:hi]
            dists = np.array(
                [masked_angle_distance(user_vec, window_matrix[i]) for i in range(len(window_matrix))],
                dtype=np.float64,
            )

        local_best = int(np.argmin(dists))
        best_idx = lo + local_best
        angles = self.frames[best_idx].get("angles", {})
        return best_idx, angles
