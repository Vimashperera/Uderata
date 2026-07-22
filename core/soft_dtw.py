"""
soft_dtw.py
-----------
Phase-3 temporal alignment: Soft-DTW distance + hard DTW path for
correspondence, with a music/video phase band constraint.

Form uses the DTW-aligned expert frame.
Timing scores lag of that alignment vs the expected clock index.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

from core.motion_features import masked_feature_distance


def softmin(values: np.ndarray, gamma: float) -> float:
    """Differentiable soft-min: -γ log Σ exp(-x_i / γ). Ignores non-finite entries."""
    g = max(float(gamma), 1e-6)
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return float("inf")
    # Numerically stable softmin
    m = float(np.min(finite))
    return float(m - g * np.log(np.sum(np.exp(-(finite - m) / g))))


def pairwise_cost_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    phase_targets: Optional[np.ndarray] = None,
    phase_weight: float = 0.0,
) -> np.ndarray:
    """
    (n, m) pairwise masked L2 costs between rows of X (n,d) and Y (m,d).

    phase_targets: optional length-n array of desired Y indices (local) for
    each X row; adds phase_weight * (j - target)^2 to discourage drift.
    """
    n, m = X.shape[0], Y.shape[0]
    C = np.empty((n, m), dtype=np.float64)
    big = 1e3
    for i in range(n):
        for j in range(m):
            d = masked_feature_distance(X[i], Y[j])
            if not np.isfinite(d):
                d = big
            if phase_targets is not None and phase_weight > 0:
                t = float(phase_targets[i])
                d = d + phase_weight * ((j - t) ** 2)
            C[i, j] = d
    return C


def soft_dtw(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float = 1.0,
    phase_targets: Optional[np.ndarray] = None,
    phase_weight: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    Soft-DTW cumulative cost (Cuturi & Blondel style).

    Returns (soft_distance, R) where R is (n+1, m+1) accumulated matrix.
    Note: soft-DTW can be < hard DTW (even slightly negative) due to softmin.
    """
    if X.size == 0 or Y.size == 0:
        return float("inf"), np.zeros((1, 1))
    C = pairwise_cost_matrix(X, Y, phase_targets=phase_targets, phase_weight=phase_weight)
    n, m = C.shape
    R = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            R[i, j] = C[i - 1, j - 1] + softmin(
                np.array([R[i - 1, j], R[i, j - 1], R[i - 1, j - 1]], dtype=np.float64),
                gamma,
            )
    return float(R[n, m]), R


def dtw_path(
    X: np.ndarray,
    Y: np.ndarray,
    phase_targets: Optional[np.ndarray] = None,
    phase_weight: float = 0.0,
    open_end: bool = True,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Classic DTW with backtracking. Returns (distance, path of (i,j) pairs).

    open_end=True (default): end at argmin_j D[n, j] so the last user frame
    can align to the best expert frame (needed for phase windows).
    """
    if X.size == 0 or Y.size == 0:
        return float("inf"), []
    C = pairwise_cost_matrix(X, Y, phase_targets=phase_targets, phase_weight=phase_weight)
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    if open_end:
        j_end = int(np.argmin(D[n, 1:])) + 1
        dist = float(D[n, j_end])
        i, j = n, j_end
    else:
        dist = float(D[n, m])
        i, j = n, m

    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        opts = [
            (D[i - 1, j - 1], i - 1, j - 1),
            (D[i - 1, j], i - 1, j),
            (D[i, j - 1], i, j - 1),
        ]
        _, i, j = min(opts, key=lambda t: t[0])
    path.reverse()
    return dist, path


def timing_score_from_lag(lag_frames: float, fps: float, good_ms: float = 120.0, poor_ms: float = 500.0) -> float:
    """
    Map absolute timing lag to 0–100.
    ≤ good_ms → 100; ≥ poor_ms → 0; linear in between.
    """
    ms = abs(float(lag_frames)) * (1000.0 / max(float(fps), 1e-3))
    if ms <= good_ms:
        return 100.0
    if ms >= poor_ms:
        return 0.0
    t = (ms - good_ms) / (poor_ms - good_ms)
    return float(100.0 * (1.0 - t))


class PhaseConstrainedAligner:
    """
    Online aligner: Soft-DTW cost + DTW path for the latest user frame,
    searching only inside [clock_idx ± phase_band].
    """

    def __init__(
        self,
        user_buf_len: int = 24,
        phase_band: int = 36,
        compute_every: int = 4,
        gamma: float = 2.0,
        fps: float = 30.0,
    ):
        self.user_buf_len = int(user_buf_len)
        self.phase_band = int(phase_band)
        self.compute_every = int(compute_every)
        self.gamma = float(gamma)
        self.fps = float(fps)

        self._buf: Deque[np.ndarray] = deque(maxlen=self.user_buf_len)
        self._frames_since = 0
        self._aligned_idx: int = 0
        self._clock_idx: int = 0
        self._lag_frames: float = 0.0
        self._soft_cost: float = 0.0
        self._timing_score: float = 100.0
        self._form_ready: bool = False

    def reset(self, fps: Optional[float] = None):
        if fps is not None:
            self.fps = float(fps)
        self._buf.clear()
        self._frames_since = 0
        self._aligned_idx = 0
        self._clock_idx = 0
        self._lag_frames = 0.0
        self._soft_cost = 0.0
        self._timing_score = 100.0
        self._form_ready = False

    def update(
        self,
        user_feat: np.ndarray,
        expert_feats: np.ndarray,
        clock_idx: Optional[int],
    ) -> Dict:
        """
        Push one user feature vector and optionally recompute alignment.

        expert_feats: (T, D) full expert feature matrix
        clock_idx: expected expert frame from music/video (None = free search)
        """
        self._buf.append(np.asarray(user_feat, dtype=np.float64).ravel())
        self._frames_since += 1

        T = int(expert_feats.shape[0])
        if T <= 0 or len(self._buf) < 4:
            return self.snapshot()

        if clock_idx is None:
            # Free search around last alignment
            center = self._aligned_idx if self._form_ready else T // 2
            lo = max(0, center - self.phase_band * 2)
            hi = min(T, center + self.phase_band * 2)
            self._clock_idx = center
        else:
            c = int(np.clip(clock_idx, 0, T - 1))
            self._clock_idx = c
            lo = max(0, c - self.phase_band)
            hi = min(T, c + self.phase_band + 1)

        if hi - lo < 4:
            lo, hi = 0, T

        should = (self._frames_since >= self.compute_every) or (not self._form_ready)
        if should:
            self._frames_since = 0
            X = np.stack(list(self._buf), axis=0)
            Y = expert_feats[lo:hi]
            d = min(X.shape[1], Y.shape[1])
            X = X[:, :d]
            Y = Y[:, :d]

            # Linear phase targets inside the window (keeps path near the beat)
            phase_targets = None
            phase_weight = 0.0
            if clock_idx is not None:
                n = X.shape[0]
                clock_local = float(self._clock_idx - lo)
                # Assume user buffer spans ~n frames ending at "now"
                phase_targets = clock_local - (n - 1) + np.arange(n, dtype=np.float64)
                # Mild bias: enough to stop window-edge drift, weak enough to
                # still measure real early/late lag for Timing.
                phase_weight = 0.22

            soft_cost, _ = soft_dtw(
                X, Y, gamma=self.gamma,
                phase_targets=phase_targets, phase_weight=phase_weight,
            )
            hard_dist, path = dtw_path(
                X, Y,
                phase_targets=phase_targets, phase_weight=phase_weight,
                open_end=True,
            )
            if path:
                _ui, yj = path[-1]
                aligned = lo + int(yj)
            else:
                aligned = self._clock_idx
                hard_dist = soft_cost

            self._aligned_idx = int(np.clip(aligned, 0, T - 1))
            self._soft_cost = float(soft_cost if np.isfinite(soft_cost) else hard_dist)

            # Timing: independent instantaneous phase match of the latest frame
            # (avoids Soft-DTW path absorbing true early/late into form warp)
            if clock_idx is None:
                self._lag_frames = 0.0
                self._timing_score = 100.0
            else:
                u = X[-1]
                best_j = 0
                best_c = float("inf")
                clock_local = float(self._clock_idx - lo)
                for j in range(Y.shape[0]):
                    d = masked_feature_distance(u, Y[j])
                    if not np.isfinite(d):
                        continue
                    # Tiny prior so ties break toward the beat
                    c = d + 0.002 * ((j - clock_local) ** 2)
                    if c < best_c:
                        best_c = c
                        best_j = j
                timing_idx = lo + best_j
                self._lag_frames = float(timing_idx - self._clock_idx)
                self._timing_score = timing_score_from_lag(self._lag_frames, self.fps)
            self._form_ready = True

        return self.snapshot()

    def snapshot(self) -> Dict:
        lag_ms = self._lag_frames * (1000.0 / max(self.fps, 1e-3))
        return {
            "aligned_idx": int(self._aligned_idx),
            "clock_idx": int(self._clock_idx),
            "lag_frames": float(self._lag_frames),
            "lag_ms": float(lag_ms),
            "timing_score": float(self._timing_score),
            "soft_cost": float(self._soft_cost),
            "ready": bool(self._form_ready),
        }
