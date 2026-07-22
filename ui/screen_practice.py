"""
screen_practice.py - Screen 3: Live Practice
3-column layout (486/308/486). Threads: Capture, Pose (+ UI update loop).

Flow: after countdown, the expert reference video (+ audio) plays while the
student dances. The reference loops REFERENCE_LOOPS times; the session ends
when the last loop finishes (or the user stops early).
"""
import os, time, threading, collections
from typing import Callable, Optional, Dict, List
import cv2, numpy as np
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

from core.pose_extractor import PoseExtractor
from core.angle_calculator import (
    compute_joint_angles, compute_joint_deviations, compute_frame_accuracy,
    classify_deviation, generate_feedback_message, get_worst_joints,
    joint_score_from_deviation, thresholds_from_scale,
    angles_to_vector, ALL_JOINT_NAMES, JOINT_DISPLAY_NAMES,
    JOINT_WEIGHTS, CALIBRATION_JOINTS
)
from core.dtw_engine import ExpertDataLoader
from core.motion_features import (
    compute_bone_directions,
    compute_angular_velocity,
    masked_cosine_bone_score,
    velocity_match_score,
    hybrid_frame_accuracy,
    build_motion_feature_vector,
)
from core.soft_dtw import PhaseConstrainedAligner
from ui.theme import C, font_display, font_ui

try:
    import pygame; pygame.mixer.init(); PYGAME=True
except Exception: PYGAME=False

BGR_GOOD=(136,255,0); BGR_CLOSE=(0,165,255); BGR_POOR=(59,59,255)

DISP_W,DISP_H = 480,270
FRAME_MS = 33
REFERENCE_LOOPS = 3   # expert video loops; session ends after the last loop

DRAW_CONNS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),   # upper body
    (11,23),(12,24),(23,24),                    # torso
    (23,25),(25,27),(24,26),(26,28),            # legs
    (27,29),(28,30),                            # ankles
]

SHORT_NAMES = {
    "left_knee":"L.Knee","right_knee":"R.Knee","left_elbow":"L.Elbow",
    "right_elbow":"R.Elbow","left_hip":"L.Hip","right_hip":"R.Hip",
    "left_shoulder":"L.Shldr","right_shoulder":"R.Shldr","spine_tilt":"Spine"
}


class _FrameSlot:
    """Thread-safe single-frame slot (newest frame always available, old ones dropped)."""
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None
        self._new   = threading.Event()
    def write(self, frame):
        with self._lock: self._frame = frame
        self._new.set()
    def push(self, frame):
        """Old alias for write."""
        self.write(frame)
    def read(self, timeout=0.05):
        if not self._new.wait(timeout): return None
        with self._lock:
            self._new.clear()
            return self._frame
    def latest(self):
        with self._lock: return self._frame


class CaptureThread(threading.Thread):
    """Thread 1: captures webcam frames into a slot at 30fps."""
    def __init__(self, slot: _FrameSlot, stop_event: threading.Event, index=0,
                 mirror: Optional[bool] = None):
        super().__init__(daemon=True, name="CaptureThread")
        self.slot = slot
        self.stop = stop_event
        self.index = index
        self.cap  = None
        self.error = None
        if mirror is None:
            try:
                import config as _cfg
                mirror = bool(getattr(_cfg, "MIRROR_WEBCAM", True))
            except Exception:
                mirror = True
        self.mirror = mirror

    def run(self):
        print(f"[CaptureThread] Starting on index {self.index} mirror={self.mirror}")
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY

        # Try all common indices with multiple backends
        indices_to_try = list(dict.fromkeys([self.index, 0, 1, 2, 3, 4]))
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
        opened = False
        for idx in indices_to_try:
            for bk in backends:
                self.cap = cv2.VideoCapture(idx, bk)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"[CaptureThread] Opened camera index {idx} backend {bk}")
                        opened = True
                        break
                self.cap.release()
            if opened:
                break

        if not opened:
            print("[CaptureThread] CRITICAL: Could not open any camera")
            self.error = "No camera found"
            self.fallback_no_camera = True
            return

        self.cap.set(cv2.CAP_PROP_FPS, 30)
        consecutive_failures = 0

        print(f"[CaptureThread] Entering while loop. stop_is_set={self.stop.is_set()}")
        while not self.stop.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                consecutive_failures = 0
                # Fix corrupted frames from virtual cameras: ensure 3-channel BGR
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                # Selfie-style mirror so learner left matches expert left when facing camera
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                self.slot.push(frame)
                time.sleep(0.01)
            else:
                consecutive_failures += 1
                print(f"[CaptureThread] Read failed (#{consecutive_failures}), retrying...")
                if consecutive_failures >= 20:
                    print("[CaptureThread] Too many failures — giving up")
                    self.error = "Camera read failed repeatedly"
                    break
                time.sleep(0.5)

        if self.cap:
            self.cap.release()
        print("[CaptureThread] Finished")


class PoseThread(threading.Thread):
    """Thread 2: MediaPipe inference + FIX 6B/6C/6D/6E/6F scoring pipeline."""

    # FIX 6C — EMA alpha (0.4 = stronger smoothing)
    EMA_ALPHA = 0.4
    # Median filter window size
    MED_WIN   = 5
    # Confidence gate threshold (FIX 6E)
    CONF_GATE = 0.6
    # Best-frame search every N frames (FIX 6B)
    FRAME_SEARCH_EVERY = 5

    def __init__(self, raw_slot: _FrameSlot, result_slot: _FrameSlot,
                 stop_event: threading.Event, expert_loader: ExpertDataLoader,
                 frame_acc_list: list, joint_hist: dict, angle_buf: collections.deque,
                 joint_dev_hist: Optional[dict] = None):
        super().__init__(daemon=True, name="PoseThread")
        self.raw      = raw_slot
        self.result   = result_slot
        self.stop     = stop_event
        self.loader   = expert_loader
        self.acc_list = frame_acc_list
        self.j_hist   = joint_hist
        self.j_dev_hist = joint_dev_hist if joint_dev_hist is not None else {n: [] for n in ALL_JOINT_NAMES}
        self.buf      = angle_buf
        self._frame_count = 0
        self._deviations: Dict = {}
        self._statuses:   Dict = {}
        self._joint_scores: Dict = {}
        self._live_accuracy: float = 0.0
        self._lock = threading.Lock()
        self.extractor: Optional[PoseExtractor] = None

        # Optional: UI sets expert sync (angles + bones + frame idx) during practice loops
        self._expert_override: Optional[Dict] = None
        self._expert_override_bones: Optional[Dict] = None
        self._expert_override_idx: int = -1

        # FIX 6B — DTW best-frame tracking
        self._best_expert_idx: int = -1

        # FIX 6C — per-joint EMA state and median buffer
        self._ema: Dict[str, float] = {n: 0.0 for n in ALL_JOINT_NAMES}
        self._med_buf: Dict[str, collections.deque] = {
            n: collections.deque(maxlen=self.MED_WIN) for n in ALL_JOINT_NAMES
        }

        # FIX 6E — last valid score per joint (None until a confident measurement)
        self._last_valid_score: Dict[str, Optional[float]] = {
            n: None for n in ALL_JOINT_NAMES
        }

        # FIX 6F — calibration baseline offsets (set during countdown)
        self._baseline_offsets: Dict[str, float] = {n: 0.0 for n in ALL_JOINT_NAMES}
        # Calibration accumulator
        self._calib_frames: List[Dict] = []
        self._calibrating = False

        # Phase-2: previous smoothed angles for angular velocity
        self._prev_smooth_angles: Optional[Dict[str, Optional[float]]] = None
        self._prev_angle_time: float = 0.0

        # Phase-3: Soft-DTW form / timing
        self._form_score: float = 0.0
        self._timing_score: float = 100.0
        self._lag_ms: float = 0.0
        self._aligner = PhaseConstrainedAligner(
            user_buf_len=24,
            phase_band=36,
            compute_every=4,
            gamma=2.0,
            fps=float(getattr(expert_loader, "fps", 30.0) or 30.0),
        )
        self._form_hist: List[float] = []
        self._timing_hist: List[float] = []
        self._lag_hist: List[float] = []

    # ─────────────────────────────────────────────────────────────────────
    # Public API for calibration (called from countdown)
    # ─────────────────────────────────────────────────────────────────────

    def start_calibration(self):
        """FIX 6F — Begin accumulating frames for baseline offset."""
        self._calib_frames.clear()
        self._calibrating = True

    def stop_calibration(self) -> str:
        """FIX 6F — Stop accumulation, compute offsets. Returns warning if needed."""
        self._calibrating = False
        if len(self._calib_frames) < 3:
            return ""   # Not enough data

        # Average each calibration joint over the collected frames
        warning = ""
        for jname in CALIBRATION_JOINTS:
            vals = [f.get(jname) for f in self._calib_frames if f.get(jname) is not None]
            if not vals:
                continue
            offset = float(np.mean(vals))
            # Cap at ±20°
            if abs(offset) > 20.0:
                warning = "Please stand further back so your full body is visible."
                offset  = np.clip(offset, -20.0, 20.0)
            self._baseline_offsets[jname] = offset

        self._calib_frames.clear()
        return warning

    def reset_calibration(self):
        """Reset offsets to zero (called from on_show)."""
        self._baseline_offsets = {n: 0.0 for n in ALL_JOINT_NAMES}
        self._calib_frames.clear()
        self._calibrating = False

    def set_expert_angle_override(self, angles: Optional[Dict]):
        """Back-compat: angles-only override (clears bones/index)."""
        self.set_expert_sync(angles, bones=None, frame_idx=-1)

    def set_expert_sync(
        self,
        angles: Optional[Dict],
        bones: Optional[Dict] = None,
        frame_idx: int = -1,
    ):
        """When set, use this expert frame (practice loop clock sync)."""
        with self._lock:
            self._expert_override = dict(angles) if angles else None
            self._expert_override_bones = dict(bones) if bones else None
            self._expert_override_idx = int(frame_idx) if frame_idx is not None else -1

    def run(self):
        # Avoid loading MediaPipe model in no-camera / test mode to keep startup fast
        no_cam = os.environ.get("NO_CAMERA", "0") in ("1", "true", "True")
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        pose_log = os.path.join(log_dir, 'pose.log')
        try:
            pose_fh = open(pose_log, 'a', encoding='utf-8')
        except Exception:
            pose_fh = None

        if not no_cam:
            # Initialize extractor asynchronously so we don't block startup or UI
            def _init_extractor():
                try:
                    try:
                        import config as _cfg
                        complexity = int(getattr(_cfg, "POSE_MODEL_COMPLEXITY", 1))
                    except Exception:
                        complexity = 1
                    ex = PoseExtractor(
                        model_complexity=complexity,
                        running_mode="VIDEO",
                    )
                    ex.reset_sequence()
                    self.extractor = ex
                    msg = (
                        f"[PoseThread] PoseExtractor ready "
                        f"(VIDEO, complexity={complexity})\n"
                    )
                    print(msg.strip())
                    if pose_fh:
                        try: pose_fh.write(msg); pose_fh.flush()
                        except Exception: pass
                except Exception as e:
                    msg = f"[PoseThread] PoseExtractor init failed: {e}\n"
                    print(msg.strip())
                    if pose_fh:
                        try: pose_fh.write(msg); pose_fh.flush()
                        except Exception: pass
                    self.extractor = None

            init_thread = threading.Thread(target=_init_extractor, daemon=True)
            init_thread.start()
        else:
            self.extractor = None

        while not self.stop.is_set():
            frame = self.raw.read(timeout=0.05)
            if frame is None: continue
            annotated, user_angles, statuses, deviations, joint_scores = self._process(frame)
            self.result.write(annotated)
            with self._lock:
                self._statuses    = statuses
                self._deviations  = deviations
                self._joint_scores = joint_scores
                if self.acc_list:
                    self._live_accuracy = float(self.acc_list[-1])
                self._frame_count += 1
        if pose_fh:
            try: pose_fh.close()
            except Exception: pass
        if self.extractor:
            try:
                self.extractor.release()
            except Exception:
                pass

    def _process(self, frame):
        # If extractor is not available (no-camera/test mode or init failure), behave as if no landmarks
        if self.extractor:
            results   = self.extractor.process_frame(frame)
            world_lms = self.extractor.get_world_landmarks(results)
            norm_lms  = self.extractor.get_landmarks(results)
        else:
            results = None
            world_lms = None
            norm_lms = None
        raw_angles = compute_joint_angles(world_lms) if world_lms else {n: None for n in ALL_JOINT_NAMES}
        user_bones = (
            compute_bone_directions(world_lms, confidence_threshold=0.5)
            if world_lms else None
        )

        # ── FIX 6C: EMA + median smoothing ───────────────────────────────
        smooth_angles: Dict[str, Optional[float]] = {}
        for jname in ALL_JOINT_NAMES:
            raw = raw_angles.get(jname)
            if raw is None:
                smooth_angles[jname] = None
                continue
            # EMA
            prev_ema = self._ema.get(jname, raw)
            ema_val  = self.EMA_ALPHA * raw + (1 - self.EMA_ALPHA) * prev_ema
            self._ema[jname] = ema_val
            # Median
            self._med_buf[jname].append(ema_val)
            buf_sorted = sorted(self._med_buf[jname])
            smooth_angles[jname] = buf_sorted[len(buf_sorted) // 2]

        # Phase-2 angular velocity (deg/s)
        now_t = time.time()
        dt = (now_t - self._prev_angle_time) if self._prev_angle_time > 0 else (1.0 / 30.0)
        user_vel = compute_angular_velocity(
            smooth_angles, self._prev_smooth_angles, dt
        )
        self._prev_smooth_angles = dict(smooth_angles)
        self._prev_angle_time = now_t

        # ── FIX 6F: accumulate calibration frames ────────────────────────
        if self._calibrating:
            self._calib_frames.append({k: v for k, v in smooth_angles.items() if v is not None})

        # ── Phase-3 Soft-DTW: form alignment vs music/video clock ─────────
        with self._lock:
            override = self._expert_override
            override_bones = self._expert_override_bones
            override_idx = self._expert_override_idx

        clock_idx = override_idx if (override is not None and override_idx >= 0) else None
        user_feat = build_motion_feature_vector(
            smooth_angles,
            bones=user_bones if self.loader.has_bones else None,
            velocity=user_vel,
            include_bones=self.loader.has_bones,
            include_velocity=True,
        )

        align_info = {"aligned_idx": -1, "timing_score": 100.0, "lag_ms": 0.0, "ready": False}
        if self.loader.is_loaded and self.loader.feature_matrix is not None:
            self._aligner.fps = float(self.loader.fps or 30.0)
            align_info = self._aligner.update(
                user_feat,
                self.loader.feature_matrix,
                clock_idx,
            )

        # Prefer Soft-DTW aligned frame for FORM; fall back to clock / nearest
        expert_bones = None
        expert_idx = -1
        if align_info.get("ready") and align_info.get("aligned_idx", -1) >= 0:
            expert_idx = int(align_info["aligned_idx"])
            expert_frame_angles = self.loader.get_frame_angles(expert_idx) or {}
            expert_bones = self.loader.get_frame_bones(expert_idx)
            self._best_expert_idx = expert_idx
        elif override is not None:
            expert_frame_angles = override
            expert_bones = override_bones
            expert_idx = override_idx if override_idx >= 0 else 0
        elif self._frame_count % self.FRAME_SEARCH_EVERY == 0:
            best_idx, expert_frame_angles = self.loader.get_best_matching_expert_frame(
                smooth_angles,
                last_best_idx=self._best_expert_idx,
                user_bones=user_bones,
                user_velocity=user_vel,
            )
            self._best_expert_idx = best_idx
            expert_idx = best_idx
            expert_bones = self.loader.get_frame_bones(best_idx)
        else:
            if self._best_expert_idx >= 0:
                expert_idx = self._best_expert_idx
                expert_frame_angles = self.loader.get_frame_angles(expert_idx) or {}
                expert_bones = self.loader.get_frame_bones(expert_idx)
            else:
                expert_frame_angles = self.loader.get_nearest_expert_angles(self._frame_count) or {}
                expert_idx = self._frame_count % max(self.loader.total_frames, 1)

        timing_score = float(align_info.get("timing_score", 100.0))
        lag_ms = float(align_info.get("lag_ms", 0.0))
        # Without a clock phase, timing is not judged
        if clock_idx is None:
            timing_score = 100.0
            lag_ms = 0.0

        expert_vel = (
            self.loader.get_velocity_at(expert_idx)
            if expert_idx >= 0 and self.loader.is_loaded
            else {n: None for n in ALL_JOINT_NAMES}
        )

        # ── FIX 6F: apply calibration baseline offsets to expert angles ───
        adjusted_expert: Dict[str, Optional[float]] = {}
        for jname in ALL_JOINT_NAMES:
            ev = expert_frame_angles.get(jname) if expert_frame_angles else None
            if ev is None:
                adjusted_expert[jname] = None
            elif jname in CALIBRATION_JOINTS:
                adjusted_expert[jname] = ev + self._baseline_offsets.get(jname, 0.0)
            else:
                adjusted_expert[jname] = ev

        # ── Deviations ────────────────────────────────────────────────────
        deviations = compute_joint_deviations(smooth_angles, adjusted_expert)
        tol_map = (
            self.loader.get_frame_tolerance(expert_idx)
            if expert_idx >= 0 and self.loader.is_loaded
            else {n: 1.0 for n in ALL_JOINT_NAMES}
        )
        statuses = {}
        for n, d in deviations.items():
            good_t, close_t = thresholds_from_scale(tol_map.get(n, 1.0))
            statuses[n] = classify_deviation(d, good_t, close_t)

        # ── FIX 6E: confidence gating per joint ───────────────────────────
        joint_scores: Dict[str, Optional[float]] = {}
        JOINT_LANDMARK_INDICES = {
            "left_knee":      [23, 25, 27], "right_knee":     [24, 26, 28],
            "left_elbow":     [11, 13, 15], "right_elbow":    [12, 14, 16],
            "left_hip":       [11, 23, 25], "right_hip":      [12, 24, 26],
            "left_shoulder":  [13, 11, 12], "right_shoulder": [14, 12, 11],
            "spine_tilt":     [11, 12, 23, 24],
        }
        for jname in ALL_JOINT_NAMES:
            dev = deviations.get(jname)
            if dev is None:
                joint_scores[jname] = None
                statuses[jname] = "unknown"
                continue
            lm_idxs = JOINT_LANDMARK_INDICES.get(jname, [])
            if norm_lms and lm_idxs:
                min_conf = min(
                    (norm_lms[i]["visibility"] for i in lm_idxs if i < len(norm_lms)),
                    default=1.0
                )
            else:
                min_conf = 1.0

            if min_conf >= self.CONF_GATE:
                good_t, close_t = thresholds_from_scale(tol_map.get(jname, 1.0))
                score = joint_score_from_deviation(dev, good_t, close_t)
                self._last_valid_score[jname] = score
                joint_scores[jname] = score
            else:
                joint_scores[jname] = None
                statuses[jname] = "unknown"

        # Angle-only accuracy, then Phase-2 hybrid FORM (bones + velocity)
        angle_acc = compute_frame_accuracy(deviations, joint_scores_override=joint_scores)
        bone_score = None
        if user_bones and expert_bones:
            # Map bone_std_mean (~0–0.3 typical) → scale 1.0–2.0
            bstd = (
                self.loader.get_frame_bone_std_mean(expert_idx)
                if expert_idx >= 0 and self.loader.is_loaded
                else 0.0
            )
            bone_tol = float(np.clip(1.0 + 5.0 * bstd, 1.0, 2.0))
            bone_score = masked_cosine_bone_score(
                user_bones, expert_bones, tolerance_scale=bone_tol
            )
        vel_score = velocity_match_score(user_vel, expert_vel)
        form_score = hybrid_frame_accuracy(angle_acc, bone_score, vel_score)

        # Phase-3 overall: mostly form, timing as separate coach signal
        # Overall for gauge history = 0.7 form + 0.3 timing (when clock active)
        if clock_idx is not None:
            frame_acc = 0.70 * form_score + 0.30 * timing_score
        else:
            frame_acc = form_score
            timing_score = 100.0

        with self._lock:
            self._form_score = float(form_score)
            self._timing_score = float(timing_score)
            self._lag_ms = float(lag_ms)

        self.acc_list.append(frame_acc)
        self._form_hist.append(float(form_score))
        self._timing_hist.append(float(timing_score))
        self._lag_hist.append(float(lag_ms))
        for jname in ALL_JOINT_NAMES:
            sc = joint_scores.get(jname)
            if sc is not None:
                self.j_hist[jname].append(sc)
            d = deviations.get(jname)
            if d is not None and joint_scores.get(jname) is not None:
                self.j_dev_hist[jname].append(d)
        self.buf.append(angles_to_vector(smooth_angles))

        # ── Draw skeleton overlay ─────────────────────────────────────────
        annotated = frame.copy()
        if norm_lms:
            h, w = annotated.shape[:2]
            for (i, j) in DRAW_CONNS:
                if i >= len(norm_lms) or j >= len(norm_lms): continue
                li, lj = norm_lms[i], norm_lms[j]
                if li["visibility"] < 0.5 or lj["visibility"] < 0.5: continue
                p1 = (int(li["x"]*w), int(li["y"]*h))
                p2 = (int(lj["x"]*w), int(lj["y"]*h))
                color = BGR_GOOD
                for jname, idxs in [("left_knee",[23,25,27]),("right_knee",[24,26,28]),
                                     ("left_elbow",[11,13,15]),("right_elbow",[12,14,16]),
                                     ("left_hip",[11,23,25]),("right_hip",[12,24,26])]:
                    if i in idxs or j in idxs:
                        s = statuses.get(jname, "unknown")
                        color = BGR_GOOD if s=="good" else (BGR_CLOSE if s=="close" else BGR_POOR)
                        break
                cv2.line(annotated, p1, p2, color, 2, cv2.LINE_AA)
            for idx in range(min(33, len(norm_lms))):
                lm = norm_lms[idx]
                if lm["visibility"] < 0.5: continue
                cx, cy = int(lm["x"]*w), int(lm["y"]*h)
                cv2.circle(annotated, (cx, cy), 4, (200,200,200), -1, cv2.LINE_AA)
            worst = get_worst_joints(deviations, top_n=1)
            if worst:
                jn, dev = worst[0]
                tip = f"{JOINT_DISPLAY_NAMES.get(jn,jn)}: {dev:.0f}d off"
                cv2.putText(annotated, tip, (8, DISP_H-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,200,0), 1, cv2.LINE_AA)

        return annotated, smooth_angles, statuses, deviations, joint_scores

    def get_state(self):
        with self._lock:
            return (
                dict(self._statuses),
                dict(self._deviations),
                self._frame_count,
                dict(self._joint_scores),
                float(self._live_accuracy),
                float(self._form_score),
                float(self._timing_score),
                float(self._lag_ms),
            )

    def get_phase3_histories(self) -> Dict:
        with self._lock:
            return {
                "form": list(self._form_hist),
                "timing": list(self._timing_hist),
                "lag_ms": list(self._lag_hist),
            }

    @property
    def feedback(self):
        with self._lock:
            devs = dict(self._deviations)
        if not devs: return "Initialising…"
        acc = compute_frame_accuracy(devs)
        return generate_feedback_message(devs, acc)


def _letterbox(img, target_w, target_h):
    """Resizes image to fit target while maintaining aspect ratio, padding with black."""
    if img is None: return None
    h, w = img.shape[:2]
    sc = min(target_w/w, target_h/h)
    nw,nh = int(w*sc),int(h*sc)
    r = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
    out = np.zeros((target_h,target_w,3),dtype=np.uint8)
    y0,x0=(target_h-nh)//2,(target_w-nw)//2
    out[y0:y0+nh,x0:x0+nw]=r
    return out

def _beep():
    if not PYGAME: return
    try:
        sr=22050; t=np.linspace(0,0.15,int(sr*0.15),False)
        tone=(np.sin(2*np.pi*880*t)*32767).astype(np.int16)
        pygame.sndarray.make_sound(np.column_stack([tone,tone])).play()
    except Exception: pass


class PracticeScreen(ctk.CTkFrame):
    """Screen 3 - Live Practice (3-column fixed layout, 3 threads)."""

    EXP_W,EXP_H = 480,270   # expert video display size
    CAM_W,CAM_H = 480,270   # webcam display size

    def __init__(self, master, video_path:str, json_path:str, step_title: str,
                 on_session_end:Callable[[dict],None], on_back:Callable, **kwargs):
        super().__init__(master, fg_color=C["bg"], **kwargs)
        self.video_path    = video_path
        self.json_path     = json_path
        self._step_title   = step_title
        self.on_session_end = on_session_end
        self.on_back        = on_back

        # Shared data
        self._raw_slot    = _FrameSlot()
        self._result_slot = _FrameSlot()
        self._stop_event  = threading.Event()
        self._frame_acc   : List[float] = []
        self._joint_hist  : Dict[str,List[float]] = {n:[] for n in ALL_JOINT_NAMES}
        self._joint_dev_hist : Dict[str,List[float]] = {n:[] for n in ALL_JOINT_NAMES}
        self._angle_buf   = collections.deque(maxlen=90)

        # Expert loader
        self._expert_loader = ExpertDataLoader(json_path)

        # Threads
        self._cap_thread  : Optional[CaptureThread] = None
        self._pose_thread : Optional[PoseThread]    = None

        # Expert video cap (UI thread)
        self._exp_cap      : Optional[cv2.VideoCapture] = None
        self._exp_total    = 0
        self._exp_cur      = 0
        self._exp_photo    = None
        self._cam_photo    = None
        # Persistent blank placeholder images — used for resets to avoid stale pyimage errors
        _blank = Image.new("RGB", (self.EXP_W, self.EXP_H), (0, 0, 0))
        self._blank_photo  = ImageTk.PhotoImage(_blank)

        # Session state
        self._running      = False
        self._session_start= 0.0
        self._after_id     = None
        self._ending       = False

        # FPS counter (instance vars, not class vars)
        self._fps_counter  = 0
        self._fps_t0       = 0.0
        self._fps_val      = 0.0

        # Angle display vars (for col1 table)
        self._exp_angle_vars: Dict[str,tk.StringVar] = {}
        # Joint status dots (col2)
        self._dot_labels: Dict[str,ctk.CTkLabel] = {}
        self._dev_labels: Dict[str,ctk.CTkLabel] = {}
        # Gauge canvas
        self._gauge_canvas = None

        # Practice: expert video loops REFERENCE_LOOPS times, then session ends
        self._session_phase = "idle"       # idle | practice
        self._practice_loop = 0            # 1..REFERENCE_LOOPS while playing
        self._rep_start_time = 0.0
        self._exp_fps_ref = 30.0
        self._ref_duration_sec = 1.0
        self._audio_wav_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)), "expert_display.wav")
        self._loop_audio_started = False
        self._frozen_exp_photo = None
        self._audio_poll_id = None
        self._rep_audio_active = False
        self._finish_after_id = None
        self._countdown_after_id = None
        self._check_webcam_id = None
        self._session_gen = 0          # bumps each start; ignores stale after() callbacks
        self._loop_frames_read = 0     # frames shown in current expert loop

        self._build_ui()

    # ═══════════════════════ UI BUILD ═════════════════════════════════════════

    def _build_ui(self):
        self.columnconfigure(0, minsize=486, weight=0)
        self.columnconfigure(1, minsize=1,   weight=0)
        self.columnconfigure(2, minsize=308, weight=0)
        self.columnconfigure(3, minsize=1,   weight=0)
        self.columnconfigure(4, minsize=486, weight=0)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        # Top bar
        self._build_topbar()
        # Dividers
        ctk.CTkFrame(self,fg_color=C["divider"],width=1,corner_radius=0).grid(
            row=1,column=1,sticky="ns")
        ctk.CTkFrame(self,fg_color=C["divider"],width=1,corner_radius=0).grid(
            row=1,column=3,sticky="ns")
        # 3 columns
        self._build_col1()
        self._build_col2()
        self._build_col3()
        # Bottom feedback bar
        self._build_bottombar()

    def _build_topbar(self):
        wrap = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0)
        wrap.grid(row=0, column=0, columnspan=5, sticky="ew")
        wrap.columnconfigure(0, weight=1)

        ctk.CTkFrame(wrap, fg_color=C["gold"], height=2, corner_radius=0).grid(
            row=0, column=0, sticky="ew"
        )

        bar = ctk.CTkFrame(wrap, fg_color=C["surface"], height=44, corner_radius=0)
        bar.grid(row=1, column=0, sticky="ew")
        bar.columnconfigure(2, weight=1)

        ctk.CTkButton(
            bar, text="← Back", width=80, height=30,
            fg_color="transparent", hover_color=C["elevated"],
            text_color=C["muted"], font=font_ui(10),
            border_width=1, border_color=C["divider"], corner_radius=6,
            command=self._stop_and_back,
        ).grid(row=0, column=0, padx=8, pady=6)

        ctk.CTkLabel(
            bar, text=f"{self._step_title} — Live Practice",
            text_color=C["ivory"], font=font_ui(12, "bold"),
        ).grid(row=0, column=1, padx=8)

        self._timer_lbl = ctk.CTkLabel(
            bar, text="⏱ 0:00", text_color=C["gold"], font=font_ui(12, "bold")
        )
        self._timer_lbl.grid(row=0, column=3, padx=8)

        self._fps_lbl = ctk.CTkLabel(
            bar, text="Live: —fps", text_color=C["muted"], font=font_ui(9)
        )
        self._fps_lbl.grid(row=0, column=4, padx=12)

    def _build_col1(self):
        col = ctk.CTkFrame(self,fg_color=C["panel"],corner_radius=0)
        col.grid(row=1,column=0,sticky="nsew")
        col.columnconfigure(0,weight=1)

        ctk.CTkLabel(col,text="Expert Reference",text_color=C["gold"],
            font=("Segoe UI",11,"bold")).pack(pady=(8,4))

        # Expert video label (fixed 480x270)
        vc = ctk.CTkFrame(col,fg_color="#000",corner_radius=6,
                          width=self.EXP_W,height=self.EXP_H)
        vc.pack(padx=3,pady=2)
        vc.pack_propagate(False)
        self._exp_lbl = ctk.CTkLabel(vc,text="Loading…",fg_color="#000",
            text_color=C["muted"])
        self._exp_lbl.pack(fill="both",expand=True)

        # Angle readout table
        ctk.CTkLabel(col,text="Current Expert Angles",text_color=C["muted"],
            font=("Segoe UI",9,"bold")).pack(pady=(6,2))

        tbl = ctk.CTkFrame(col,fg_color=C["surface"],corner_radius=6)
        tbl.pack(fill="x",padx=6,pady=2)

        for jname in ALL_JOINT_NAMES:
            row = ctk.CTkFrame(tbl,fg_color="transparent")
            row.pack(fill="x",padx=6,pady=1)
            ctk.CTkLabel(row,text=SHORT_NAMES.get(jname,jname),
                text_color=C["muted"],font=("Segoe UI",9),
                width=70,anchor="w").pack(side="left")
            var = tk.StringVar(value="—°")
            self._exp_angle_vars[jname] = var
            ctk.CTkLabel(row,textvariable=var,text_color=C["gold"],
                font=("Segoe UI",9,"bold"),width=55,anchor="e").pack(side="right")

        self._rep_status = ctk.CTkLabel(
            col,
            text="",
            text_color=C["gold"],
            font=("Georgia", 26, "bold"),
        )
        self._rep_status.pack(pady=(4, 2))

    def _build_col2(self):
        col = ctk.CTkFrame(self,fg_color=C["surface"],corner_radius=0)
        col.grid(row=1,column=2,sticky="nsew")
        col.columnconfigure(0,weight=1)

        # Timer
        self._hud_timer = ctk.CTkLabel(col,text="0:00",
            text_color=C["gold"],font=("Georgia",22,"bold"))
        self._hud_timer.pack(pady=(14,4))

        ctk.CTkLabel(col,text="FORM",text_color=C["muted"],
            font=("Segoe UI",8,"bold")).pack()

        # Gauge canvas
        self._gauge_canvas = tk.Canvas(col,width=150,height=150,
            bg=C["surface"],highlightthickness=0)
        self._gauge_canvas.pack(pady=4)
        self._draw_gauge(0)

        self._gauge_pct = ctk.CTkLabel(col,text="—",
            text_color=C["good"],font=("Georgia",16,"bold"))
        self._gauge_pct.pack()

        self._timing_lbl = ctk.CTkLabel(
            col, text="Timing: —",
            text_color=C["muted"], font=("Segoe UI", 10, "bold"),
        )
        self._timing_lbl.pack(pady=(2, 0))
        self._lag_lbl = ctk.CTkLabel(
            col, text="",
            text_color=C["muted"], font=("Segoe UI", 9),
        )
        self._lag_lbl.pack()

        ctk.CTkFrame(col,fg_color=C["divider"],height=1).pack(fill="x",padx=10,pady=8)
        ctk.CTkLabel(col,text="JOINT STATUS",text_color=C["muted"],
            font=("Segoe UI",8,"bold")).pack()

        # 9 joint rows
        jframe = ctk.CTkFrame(col,fg_color="transparent")
        jframe.pack(fill="x",padx=8,pady=4)
        for jname in ALL_JOINT_NAMES:
            r = ctk.CTkFrame(jframe,fg_color="transparent")
            r.pack(fill="x",pady=1)
            dot = ctk.CTkLabel(r,text="●",text_color=C["muted"],
                font=("Segoe UI",10),width=16)
            dot.pack(side="left")
            ctk.CTkLabel(r,text=SHORT_NAMES.get(jname,jname),
                text_color=C["muted"],font=("Segoe UI",8),
                anchor="w",width=58).pack(side="left")
            dev_lbl = ctk.CTkLabel(r,text="—",text_color=C["muted"],
                font=("Segoe UI",8),anchor="e",width=40)
            dev_lbl.pack(side="right")
            self._dot_labels[jname] = dot
            self._dev_labels[jname] = dev_lbl

        ctk.CTkFrame(col,fg_color=C["divider"],height=1).pack(fill="x",padx=10,pady=6)

        # Start/End button
        self._action_btn = ctk.CTkButton(col,text="▶ START",width=140,height=42,
            font=font_ui(12,"bold"),fg_color=C["gold"],
            hover_color=C["gold_hover"],text_color=C["ink"],
            corner_radius=8,command=self._start_session)
        self._action_btn.pack(pady=6)

    def _build_col3(self):
        col = ctk.CTkFrame(self,fg_color=C["panel"],corner_radius=0)
        col.grid(row=1,column=4,sticky="nsew")
        col.columnconfigure(0,weight=1)
        # Store ref for countdown overlay
        self._col3 = col

        ctk.CTkLabel(col,text="Your Performance",text_color=C["gold"],
            font=("Segoe UI",11,"bold")).pack(pady=(8,4))

        vc = ctk.CTkFrame(col,fg_color="#000",corner_radius=6,
                          width=self.CAM_W,height=self.CAM_H)
        vc.pack(padx=3,pady=2)
        vc.pack_propagate(False)
        self._cam_frame = vc
        self._cam_lbl = ctk.CTkLabel(vc,text="Camera starting…",fg_color="#000",
            text_color=C["muted"])
        self._cam_lbl.pack(fill="both", expand=True)

        # Separate overlay — NEVER write "Session Complete!" onto the camera label
        self._complete_overlay = ctk.CTkLabel(
            vc, text="Session Complete!",
            font=("Georgia", 22, "bold"),
            text_color=C["gold"], fg_color="#000000",
            width=self.CAM_W, height=self.CAM_H,
        )
        # hidden until session ends
        self._complete_overlay.place_forget()

        # Countdown overlay label (hidden initially)
        self._countdown_lbl = ctk.CTkLabel(col,text="",
            font=("Georgia",80,"bold"),text_color=C["gold"],fg_color="transparent")

    def _build_bottombar(self):
        bar = ctk.CTkFrame(self,fg_color=C["card"],height=44,corner_radius=0)
        bar.grid(row=2,column=0,columnspan=5,sticky="ew")
        bar.columnconfigure(0,weight=1)
        self._feedback_lbl = ctk.CTkLabel(bar,
            text="Press START when you are ready.",
            text_color=C["offwhite"],font=("Segoe UI",11,"italic"))
        self._feedback_lbl.grid(row=0,column=0,padx=20,pady=10)

    # ═══════════════════════ GAUGE ════════════════════════════════════════════

    def _draw_gauge(self, value:float):
        c = self._gauge_canvas
        if c is None: return
        c.delete("all")
        cx,cy,r = 75,75,60
        c.create_arc(cx-r,cy-r,cx+r,cy+r,start=220,extent=-260,
            outline=C["divider"],width=10,style="arc")
        if value > 0:
            ext = -260*(value/100)
            col = C["good"] if value>=75 else (C["close"] if value>=55 else C["poor"])
            c.create_arc(cx-r,cy-r,cx+r,cy+r,start=220,extent=ext,
                outline=col,width=10,style="arc")
        c.create_text(cx,cy,text=f"{value:.0f}%",
            fill=C["offwhite"],font=("Georgia",18,"bold"))


    # ═══════════════════════ SESSION CONTROL ══════════════════════════════════

    def _cancel_session_timers(self):
        """Cancel delayed callbacks so Practice Again can't inherit a finished session."""
        for attr in ("_after_id", "_audio_poll_id", "_finish_after_id",
                     "_countdown_after_id", "_check_webcam_id"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)

    def _open_expert_video(self) -> bool:
        """(Re)open expert video from the start. Prefer reopen over seek (more reliable)."""
        if self._exp_cap is not None:
            try:
                self._exp_cap.release()
            except Exception:
                pass
            self._exp_cap = None
        if not os.path.isfile(self.video_path):
            return False
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False
        self._exp_cap = cap
        self._exp_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._exp_fps_ref = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._ref_duration_sec = max(
            self._exp_total / max(self._exp_fps_ref, 1e-3), 0.5
        )
        self._exp_cur = 0
        self._loop_frames_read = 0
        return True

    def _start_session(self):
        try:
            self._cancel_session_timers()
            self._session_gen += 1
            self._ending = False
            self._running = False
            self._session_phase = "idle"
            self._practice_loop = 0
            self._loop_frames_read = 0

            self._action_btn.configure(state="disabled", text="Starting…")
            # Load expert data
            if not self._expert_loader.is_loaded:
                if not self._expert_loader.load():
                    self._show_error("Expert data not found.\nRun:  python preprocess_multi_expert.py")
                    self._action_btn.configure(state="normal", text="▶ START")
                    return
            # Open expert video fresh from the start
            if not self._open_expert_video():
                self._show_error(
                    f"Expert video not found / cannot open:\n{self.video_path}"
                )
                self._action_btn.configure(state="normal", text="▶ START")
                return
            # Reset data
            self._frame_acc.clear()
            for k in self._joint_hist: self._joint_hist[k].clear()
            for k in self._joint_dev_hist: self._joint_dev_hist[k].clear()
            self._angle_buf.clear()
            # Stop old threads using their own event, then create a fresh event for new session
            old_stop = self._stop_event
            old_stop.set()
            if self._cap_thread is not None:
                self._cap_thread.join(timeout=3.0)
                self._cap_thread = None
            if self._pose_thread is not None:
                self._pose_thread.join(timeout=3.0)
                self._pose_thread = None
            # Give DirectShow a moment to free the webcam before reopening
            time.sleep(0.35)
            self._hide_complete_overlay()
            self._reset_cam_panel("Camera starting…")
            # Fresh stop event — completely isolated from the old session
            self._stop_event = threading.Event()
            print(f"[START] New stop_event created, is_set={self._stop_event.is_set()} gen={self._session_gen}")
            # Start fresh threads with the new stop event
            self._raw_slot    = _FrameSlot()
            self._result_slot = _FrameSlot()
            self._cap_thread  = CaptureThread(self._raw_slot, self._stop_event)
            self._pose_thread = PoseThread(
                self._raw_slot, self._result_slot, self._stop_event,
                self._expert_loader, self._frame_acc, self._joint_hist, self._angle_buf,
                self._joint_dev_hist)
            self._cap_thread.start()
            self._check_webcam_id = self.after(800, self._check_webcam_then_countdown)
        except Exception as e:
            import traceback
            self._show_error(f"Error starting session:\n{traceback.format_exc()}")
            self._action_btn.configure(state="normal", text="▶ START")

    def _check_webcam_then_countdown(self):
        self._check_webcam_id = None
        # If capture thread reported an error, check if it elected to fallback to no-camera
        if self._cap_thread:
            if getattr(self._cap_thread, 'fallback_no_camera', False):
                self._feedback_lbl.configure(
                    text="⚠ Webcam not found — running in no-camera mode.",
                    text_color="#FFA500")
                self.after(3000, lambda: self._feedback_lbl.configure(
                    text="Press START when you are ready.", text_color=C["offwhite"]))
            elif getattr(self._cap_thread, 'error', None):
                # Genuine error with no fallback available
                self._show_error(
                    f"Webcam not found.\n\nTroubleshooting:\n"
                    "• Check USB connection\n• Allow camera access\n"
                    "• Close other apps using the camera")
                self._stop_event.set()
                self._action_btn.configure(state="normal", text="▶ START")
                return
        # Start pose thread and proceed with countdown (pose thread will operate on synthetic frames if needed)
        if self._pose_thread and not self._pose_thread.is_alive():
            self._pose_thread.start()
        self._run_countdown(5)

    def _run_countdown(self, n: int):
        self._countdown_after_id = None
        if self._ending:
            return
        if n == 2 and self._pose_thread:
            # FIX 6F: start capturing calibration frames at count 2
            self._pose_thread.start_calibration()
        if n <= 0:
            self._countdown_lbl.pack_forget()
            if self._pose_thread:
                # FIX 6F: compute and apply baseline offsets
                warn = self._pose_thread.stop_calibration()
                if warn:
                    self._feedback_lbl.configure(
                        text=f"⚠  {warn}", text_color="#FFA500")
                    self.after(3000, lambda: self._feedback_lbl.configure(
                        text="Press START when you are ready.",
                        text_color=C["offwhite"]))
            self._begin_tracking()
            return
        self._countdown_lbl.configure(text=str(n))
        self._countdown_lbl.pack(pady=8)
        self._countdown_lbl.lift()
        _beep()
        self._countdown_after_id = self.after(1000, lambda: self._run_countdown(n - 1))

    def _begin_tracking(self):
        # Ensure expert video is open at frame 0 for this session
        if self._exp_cap is None or not self._exp_cap.isOpened():
            if not self._open_expert_video():
                self._show_error("Cannot reopen expert video for practice.")
                self._action_btn.configure(state="normal", text="▶ START")
                return
        else:
            # Reopen rather than seek — avoids EOF stuck state after prior session
            if not self._open_expert_video():
                self._show_error("Cannot reopen expert video for practice.")
                self._action_btn.configure(state="normal", text="▶ START")
                return

        self._running = True
        self._ending = False
        self._session_start = time.time()
        self._session_phase = "practice"
        self._loop_audio_started = False
        self._rep_audio_active = False
        self._frozen_exp_photo = None
        self._practice_loop = 1
        self._exp_cur = 0
        self._loop_frames_read = 0
        self._rep_start_time = time.time()
        if self._pose_thread:
            self._pose_thread.set_expert_angle_override(None)
        # Clear leftover completion overlay before live frames
        self._hide_complete_overlay()
        self._reset_cam_panel("Get ready…")
        self._rep_status.configure(
            text=f"Loop {self._practice_loop} / {REFERENCE_LOOPS}",
            text_color=C["gold"],
        )
        self._feedback_lbl.configure(
            text=f"Follow the expert — loop {self._practice_loop} of {REFERENCE_LOOPS}.",
            text_color=C["offwhite"],
        )
        self._action_btn.configure(
            text="⏹ End Session",
            fg_color=C["gold_deep"], hover_color=C["gold_hover"],
            text_color=C["ivory"], state="normal",
            command=self._end_session_user,
        )
        self._start_loop_audio()
        self._ui_update()

    def _end_session_user(self):
        if not self._ending:
            self._ending = True
            self._show_complete_overlay("user")

    def _ensure_mixer(self) -> bool:
        if not PYGAME:
            return False
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        except Exception:
            try:
                pygame.mixer.init()
            except Exception:
                return False
        return True

    def _stop_any_music(self):
        if not PYGAME:
            return
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def _start_loop_audio(self):
        """Start/restart music in sync with the current expert video loop."""
        self._loop_audio_started = False
        self._rep_audio_active = False
        if not PYGAME or not os.path.isfile(self._audio_wav_path):
            return
        if not self._ensure_mixer():
            return
        try:
            self._stop_any_music()
            pygame.mixer.music.load(self._audio_wav_path)
            pygame.mixer.music.play(loops=0)
            self._loop_audio_started = True
            self._rep_audio_active = True
        except Exception:
            pass

    def _on_expert_loop_ended(self):
        """Called when one full pass of the expert video finishes."""
        if self._session_phase != "practice" or self._ending or not self._running:
            return
        # Ignore spurious EOF before any real frames (common after Practice Again reopen)
        if self._loop_frames_read < 5:
            print(
                f"[Practice] Spurious EOF at loop {self._practice_loop} "
                f"(frames={self._loop_frames_read}) — reopening video"
            )
            if not self._open_expert_video():
                self._ending = True
                self._show_complete_overlay("video_error")
            return
        if self._practice_loop < REFERENCE_LOOPS:
            self._practice_loop += 1
            self._rep_start_time = time.time()
            if not self._open_expert_video():
                self._ending = True
                self._show_complete_overlay("video_error")
                return
            self._rep_status.configure(
                text=f"Loop {self._practice_loop} / {REFERENCE_LOOPS}",
                text_color=C["gold"],
            )
            self._feedback_lbl.configure(
                text=f"Follow the expert — loop {self._practice_loop} of {REFERENCE_LOOPS}.",
                text_color=C["offwhite"],
            )
            self._start_loop_audio()
        else:
            if not self._ending:
                self._ending = True
                self._show_complete_overlay("loops_complete")

    def _show_complete_overlay(self, ended_by:str):
        """Show 'Session Complete!' for 1.5s then transition."""
        gen = self._session_gen
        self._running = False
        self._cancel_session_timers()
        self._stop_any_music()
        if self._pose_thread:
            self._pose_thread.set_expert_angle_override(None)
        # Stop threads
        self._stop_event.set()
        # Show overlay ABOVE camera (do not destroy camera label content permanently)
        self._show_complete_overlay_ui()
        self._finish_after_id = self.after(
            1500, lambda: self._finish_session(ended_by, gen)
        )

    def _finish_session(self, ended_by:str, gen: Optional[int] = None):
        self._finish_after_id = None
        # Ignore stale callback from a previous Practice Again generation
        if gen is not None and gen != self._session_gen:
            print(f"[Practice] Ignoring stale finish gen={gen} current={self._session_gen}")
            return
        duration = time.time() - self._session_start
        # Clean up resources
        if self._exp_cap:
            self._exp_cap.release(); self._exp_cap = None
        # Build session_data
        p3 = {}
        if self._pose_thread and hasattr(self._pose_thread, "get_phase3_histories"):
            p3 = self._pose_thread.get_phase3_histories()
        session_data = {
            "step_name":       self._step_title,
            "duration_seconds": round(duration, 2),
            "frame_accuracies": list(self._frame_acc),
            "form_accuracies":  p3.get("form", list(self._frame_acc)),
            "timing_accuracies": p3.get("timing", []),
            "lag_ms_history":   p3.get("lag_ms", []),
            "joint_histories":  {k: list(v) for k,v in self._joint_hist.items()},
            "joint_deviations": {k: list(v) for k,v in self._joint_dev_hist.items()},
            "ended_by":         ended_by,
            "reference_loops_target": REFERENCE_LOOPS,
            "reference_loop_at_end": self._practice_loop,
            "practice_reps_target": REFERENCE_LOOPS,  # alias for older report code
            "practice_rep_at_end": self._practice_loop,
            "session_phase_at_end": getattr(self, "_session_phase", "idle"),
        }
        self.on_session_end(session_data)

    def _stop_and_back(self):
        self._running = False
        if self._audio_poll_id:
            try:
                self.after_cancel(self._audio_poll_id)
            except Exception:
                pass
            self._audio_poll_id = None
        self._stop_any_music()
        if self._after_id:
            self.after_cancel(self._after_id)
        self._stop_event.set()
        if self._exp_cap: self._exp_cap.release(); self._exp_cap = None
        self.on_back()

    # ═══════════════════════ UI UPDATE LOOP ═══════════════════════════════════

    def _ui_update(self):
        if not self._running: return
        t0 = time.perf_counter()
        
        # print(f"[UI] Tick - result_slot empty={self._result_slot._data is None}, raw_slot empty={self._raw_slot._data is None}")

        # ── Expert panel: looping reference video (student practices along) ──
        if self._session_phase == "practice" and self._exp_cap is not None:
            ret, exp_frame = self._exp_cap.read()
            if not ret:
                self._on_expert_loop_ended()
                if not self._running or self._ending:
                    return
            else:
                self._exp_cur += 1
                self._loop_frames_read += 1
                lb = _letterbox(exp_frame, self.EXP_W, self.EXP_H)
                photo = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)))
                self._exp_photo = photo
                try:
                    self._exp_lbl.configure(image=photo, text="")
                except Exception:
                    pass
                ea = self._expert_loader.get_angles_for_video_sync(
                    self._exp_cur, self._exp_total)
                if ea:
                    for jname, var in self._exp_angle_vars.items():
                        v = ea.get(jname)
                        var.set(f"{v:.1f}°" if v is not None else "—°")
                    if self._pose_thread:
                        sx = self._expert_loader.sync_index(self._exp_cur, self._exp_total)
                        eb = self._expert_loader.get_bones_for_video_sync(
                            self._exp_cur, self._exp_total)
                        self._pose_thread.set_expert_sync(
                            ea, bones=eb, frame_idx=sx if sx is not None else -1)

        # ── Webcam / pose panel ───────────────────────────────────────────────
        # Keep completion overlay off while a live session is running
        if self._running and not self._ending:
            self._hide_complete_overlay()

        result_frame = self._result_slot.latest()
        if result_frame is not None:
            lb2 = _letterbox(result_frame, self.CAM_W, self.CAM_H)
            photo2 = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(lb2, cv2.COLOR_BGR2RGB)))
            self._cam_photo = photo2
            try:
                self._cam_lbl.configure(image=photo2, text="")
            except Exception:
                pass
        else:
            # If processed result not ready yet, show raw capture
            raw_frame = self._raw_slot.latest()
            if raw_frame is not None:
                lb2 = _letterbox(raw_frame, self.CAM_W, self.CAM_H)
                photo2 = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(lb2, cv2.COLOR_BGR2RGB)))
                self._cam_photo = photo2
                try:
                    self._cam_lbl.configure(image=photo2, text="")
                except Exception:
                    pass
            else:
                # No frame yet — keep a clear waiting state (never leave Session Complete)
                try:
                    cur = str(self._cam_lbl.cget("text") or "")
                    if "Session Complete" in cur or cur == "":
                        self._cam_lbl.configure(
                            image=self._blank_photo,
                            text="Waiting for camera…",
                            text_color=C["muted"],
                        )
                        self._cam_photo = self._blank_photo
                except Exception:
                    pass
        
        # ── Performance stats ─────────────────────────────────────────────────
        if self._pose_thread:
            (statuses, deviations, fc, joint_scores, live_acc,
             form_score, timing_score, lag_ms) = self._pose_thread.get_state()
            # Gauge shows FORM (pose quality after Soft-DTW alignment)
            acc = float(form_score) if form_score else float(live_acc or 0.0)
            if acc <= 0 and joint_scores:
                usable = {j: s for j, s in joint_scores.items() if s is not None}
                total_w = sum(JOINT_WEIGHTS.get(j, 0) for j in usable)
                if total_w > 1e-6:
                    acc = sum(usable[j] * JOINT_WEIGHTS.get(j, 0) for j in usable) / total_w
            self._draw_gauge(acc)
            color = C["good"] if acc>=75 else (C["close"] if acc>=55 else C["poor"])
            self._gauge_pct.configure(text=f"{acc:.0f}%", text_color=color)

            # Timing (phase lag vs music/video clock)
            tcol = C["good"] if timing_score>=75 else (C["close"] if timing_score>=55 else C["poor"])
            self._timing_lbl.configure(
                text=f"Timing: {timing_score:.0f}%", text_color=tcol)
            if abs(lag_ms) >= 1.0:
                sign = "early" if lag_ms < 0 else "late"
                self._lag_lbl.configure(
                    text=f"{abs(lag_ms):.0f} ms {sign}", text_color=C["muted"])
            else:
                self._lag_lbl.configure(text="on beat", text_color=C["muted"])

            # Joint dots
            dot_colors = {"good":C["good"],"close":C["close"],"poor":C["poor"],"unknown":C["muted"]}
            for jname in ALL_JOINT_NAMES:
                s   = statuses.get(jname,"unknown")
                dev = deviations.get(jname)
                self._dot_labels[jname].configure(text_color=dot_colors.get(s,C["muted"]))
                self._dev_labels[jname].configure(
                    text=f"{dev:.0f}°" if dev is not None else "—")
            # Feedback
            fb = self._pose_thread.feedback
            if timing_score < 55 and abs(lag_ms) > 150:
                tip = "Speed up a little." if lag_ms < 0 else "Hold back — you're ahead of the beat."
                fb = f"{fb}  ·  {tip}"
            self._feedback_lbl.configure(text=fb)

        # ── Timer ─────────────────────────────────────────────────────────────
        elapsed = time.time() - self._session_start
        self._hud_timer.configure(text=f"{int(elapsed)//60}:{int(elapsed)%60:02d}")
        self._timer_lbl.configure(text=f"⏱ {int(elapsed)//60}:{int(elapsed)%60:02d}")

        # ── FPS counter ───────────────────────────────────────────────────────
        self._fps_counter += 1
        if self._fps_counter >= 30:
            now = time.time()
            if self._fps_t0 > 0:
                self._fps_val = 30 / (now - self._fps_t0)
                self._fps_lbl.configure(text=f"Live: {self._fps_val:.0f}fps")
            self._fps_t0 = now; self._fps_counter = 0

        # ── Schedule next update ──────────────────────────────────────────────
        if not self._running or self._ending:
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000
        delay = max(1, int(FRAME_MS - elapsed_ms))
        self._after_id = self.after(delay, self._ui_update)

    # ═══════════════════════ HELPERS ═════════════════════════════════════════

    def _hide_complete_overlay(self):
        if getattr(self, "_complete_overlay", None) is not None:
            try:
                self._complete_overlay.place_forget()
            except Exception:
                pass

    def _show_complete_overlay_ui(self):
        if getattr(self, "_complete_overlay", None) is not None:
            try:
                self._complete_overlay.configure(text="Session Complete!")
                self._complete_overlay.place(relx=0.5, rely=0.5, anchor="center")
                self._complete_overlay.lift()
            except Exception:
                pass

    def _reset_cam_panel(self, message: str = "Camera starting…"):
        """Restore camera label to a clean state (never leave Session Complete stuck)."""
        self._hide_complete_overlay()
        self._cam_photo = self._blank_photo
        try:
            self._cam_lbl.configure(
                image=self._blank_photo,
                text=message,
                text_color=C["muted"],
                font=("Segoe UI", 12),
            )
            self._cam_lbl.lift()
        except Exception:
            pass

    def _show_error(self, msg:str):
        import tkinter.messagebox as mb
        mb.showerror("Error", msg)

    def on_show(self):
        """Called every time this screen becomes visible. Fully resets to pre-session state."""
        # Invalidate any pending finish/countdown from the previous session
        self._session_gen += 1
        self._cancel_session_timers()
        # 1. Stop any running UI loop
        self._running = False
        self._stop_any_music()
        # 2. Signal current threads to stop (keep refs so _start_session can join them)
        self._stop_event.set()
        # 3. Release expert video cap
        if self._exp_cap:
            self._exp_cap.release()
            self._exp_cap = None
        # 6. Reset session flags
        self._ending       = False
        self._session_phase = "idle"
        self._practice_loop = 0
        self._loop_frames_read = 0
        self._loop_audio_started = False
        self._rep_audio_active = False
        self._frozen_exp_photo = None
        self._exp_cur      = 0
        self._fps_counter  = 0
        self._fps_t0       = 0.0
        # 7. Reset session data lists
        self._frame_acc.clear()
        for k in self._joint_hist:
            self._joint_hist[k].clear()
        for k in self._joint_dev_hist:
            self._joint_dev_hist[k].clear()
        self._angle_buf.clear()
        # 8. Reset action button → START
        if hasattr(self, '_action_btn'):
            self._action_btn.configure(
                text="▶ START", state="normal",
                fg_color=C["gold"], hover_color=C["gold_hover"],
                text_color=C["ink"], command=self._start_session)
        # 9. Reset display labels using blank placeholder to avoid stale pyimage errors
        if hasattr(self, '_cam_lbl'):
            self._reset_cam_panel("Camera starting…")
            # Immediately show probe image if available so UI gives visual feedback
            probe_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'camera_probe_0.jpg')
            if os.path.isfile(probe_path):
                try:
                    from PIL import Image as PILImage
                    img = PILImage.open(probe_path).resize((self.CAM_W, self.CAM_H))
                    photo = ImageTk.PhotoImage(img)
                    self._cam_lbl.configure(image=photo, text='Probe image (no live frame)')
                    self._cam_photo = photo
                except Exception:
                    pass
        if hasattr(self, '_exp_lbl'):
            self._exp_photo = self._blank_photo
            try:
                self._exp_lbl.configure(image=self._blank_photo,
                    text="Loading…", text_color=C["muted"])
            except Exception:
                pass
        if hasattr(self, '_rep_status'):
            self._rep_status.configure(text="", text_color=C["gold"])
        if hasattr(self, '_feedback_lbl'):
            self._feedback_lbl.configure(
                text="Press START when you are ready.",
                text_color=C["offwhite"])
        if hasattr(self, '_hud_timer'):
            self._hud_timer.configure(text="0:00")
        if hasattr(self, '_timer_lbl'):
            self._timer_lbl.configure(text="⏱ 0:00")
        if hasattr(self, '_fps_lbl'):
            self._fps_lbl.configure(text="Live: —fps")
        if hasattr(self, '_gauge_pct'):
            self._gauge_pct.configure(text="—", text_color=C["good"])
        self._draw_gauge(0)
        if hasattr(self, '_timing_lbl'):
            self._timing_lbl.configure(text="Timing: —", text_color=C["muted"])
        if hasattr(self, '_lag_lbl'):
            self._lag_lbl.configure(text="", text_color=C["muted"])
        # 10. Reset joint status indicators
        for jname in ALL_JOINT_NAMES:
            if jname in self._dot_labels:
                self._dot_labels[jname].configure(text_color=C["muted"])
            if jname in self._dev_labels:
                self._dev_labels[jname].configure(text="—")
        # 11. Reset expert angle vars
        for var in self._exp_angle_vars.values():
            var.set("—°")
        # 12. Hide countdown
        if hasattr(self, '_countdown_lbl'):
            self._countdown_lbl.pack_forget()

    def on_hide(self):
        """Called when navigating away from this screen."""
        self._session_gen += 1
        self._running = False
        self._cancel_session_timers()
        self._stop_any_music()
        self._stop_event.set()
        if self._exp_cap:
            self._exp_cap.release()
            self._exp_cap = None

    def destroy(self):
        self.on_hide()
        super().destroy()
