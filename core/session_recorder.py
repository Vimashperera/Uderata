"""
Background session recording.

Frames are queued from the practice UI and written by RecordingThread.
Disk I/O never runs on the live scoring path.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

import config
from core.angle_calculator import JOINT_DISPLAY_NAMES

RECORD_FPS = 20
FRAME_SIZE = (960, 400)  # width, height
TOP_H = 40
VIDEO_H = 270
VIDEO_W = 480
BOTTOM_H = 90
MIN_FREE_BYTES = 100 * 1024 * 1024

# Display order requested for the burned-in joint row.
JOINT_ROW = (
    ("left_knee", "LK"),
    ("right_knee", "RK"),
    ("left_hip", "LH"),
    ("right_hip", "RH"),
    ("left_shoulder", "LS"),
    ("right_shoulder", "RS"),
    ("left_elbow", "LE"),
    ("right_elbow", "RE"),
    ("spine_tilt", "SP"),
)

_BG = (46, 26, 26)          # BGR for #1A1A2E
_GOLD = (0, 165, 201)       # BGR gold
_WHITE = (255, 255, 255)
_GREEN = (80, 180, 60)
_ORANGE = (0, 140, 255)
_RED = (60, 60, 220)
_GRAY = (90, 90, 90)
_BAR_BG = (50, 50, 50)

_INDEX_LOCK = threading.Lock()


def recordings_dir() -> str:
    return os.path.join(config.APP_ROOT, "recordings")


def index_path() -> str:
    return os.path.join(recordings_dir(), "index.json")


def ensure_recordings_dir() -> str:
    path = recordings_dir()
    os.makedirs(path, exist_ok=True)
    return path


def free_bytes(path: str) -> int:
    usage = shutil.disk_usage(path if os.path.isdir(path) else os.path.dirname(path) or path)
    return int(usage.free)


def slugify(title: str) -> str:
    raw = "".join(ch if ch.isalnum() else "_" for ch in (title or "").lower())
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw.strip("_") or "session"


def new_recording_path(step_title: str) -> str:
    folder = ensure_recordings_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{slugify(step_title)}_{stamp}.mp4"
    return os.path.join(folder, name)


def cleanup_incomplete_recordings() -> int:
    """
    Drop tiny, recently touched mp4 files left by a force-close
    before VideoWriter.release().
    """
    folder = recordings_dir()
    if not os.path.isdir(folder):
        return 0
    now = time.time()
    removed = 0
    for name in os.listdir(folder):
        if not name.lower().endswith(".mp4"):
            continue
        path = os.path.join(folder, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if (now - st.st_mtime) < 3600 and st.st_size < 1_000_000:
            try:
                os.remove(path)
                removed += 1
                print(f"[recording] Removed incomplete file: {name}")
            except OSError as exc:
                print(f"[recording] Could not remove {name}: {exc}")
    return removed


def _score_color(score: Optional[float]):
    if score is None:
        return _GRAY
    if score >= 75:
        return _GREEN
    if score >= 55:
        return _ORANGE
    return _RED


def _fit(frame, width: int, height: int) -> np.ndarray:
    if frame is None or getattr(frame, "size", 0) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    h, w = frame.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    y0, x0 = (height - nh) // 2, (width - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def _wrap_two_lines(text: str, max_width: int, font_scale: float = 0.4) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        width = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        if width <= max_width and len(lines) < 1:
            current = trial
        else:
            lines.append(current)
            current = word
            if len(lines) == 2:
                break
    if len(lines) < 2:
        lines.append(current)
    return lines[:2]


def build_recording_frame(
    expert_frame,
    student_frame,
    joint_scores: Optional[Dict] = None,
    overall_score: float = 0.0,
    feedback: str = "",
    timestamp: float = 0.0,
    step_name: str = "",
    header_left: str = "",
    mistake_text: str = "",
) -> np.ndarray:
    """Stack already-rendered panels into a 960×400 BGR review frame."""
    joint_scores = joint_scores or {}
    top = np.full((TOP_H, FRAME_SIZE[0], 3), _BG, dtype=np.uint8)
    left = (header_left or config.APP_NAME)[:42]
    cv2.putText(top, left, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)

    step = (step_name or "Step")[:28]
    step_size = cv2.getTextSize(step, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(
        top, step,
        ((FRAME_SIZE[0] - step_size[0]) // 2, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GOLD, 1, cv2.LINE_AA,
    )

    elapsed = max(0, int(timestamp))
    clock = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
    clock_size = cv2.getTextSize(clock, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(
        top, clock,
        (FRAME_SIZE[0] - clock_size[0] - 12, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA,
    )

    expert = _fit(expert_frame, VIDEO_W, VIDEO_H)
    student = _fit(student_frame, VIDEO_W, VIDEO_H).copy()
    if mistake_text:
        banner = student.copy()
        cv2.rectangle(banner, (0, 0), (VIDEO_W, 36), (0, 0, 160), -1)
        student = cv2.addWeighted(banner, 0.55, student, 0.45, 0)
        cv2.putText(
            student, mistake_text[:42], (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA,
        )

    middle = np.hstack((expert, student))
    bottom = np.full((BOTTOM_H, FRAME_SIZE[0], 3), _BG, dtype=np.uint8)

    cv2.putText(bottom, "Accuracy", (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
    score = float(overall_score or 0.0)
    score = max(0.0, min(100.0, score))
    bar_x, bar_y, bar_w, bar_h = 16, 36, 260, 20
    cv2.rectangle(bottom, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _BAR_BG, -1)
    fill = int(bar_w * (score / 100.0))
    if fill > 0:
        cv2.rectangle(
            bottom, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
            _score_color(score), -1,
        )
    pct = f"{score:.0f}%"
    pct_size = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(
        bottom, pct,
        (bar_x + (bar_w - pct_size[0]) // 2, bar_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA,
    )

    any_low = False
    origin_x = 300
    slot = 360 / len(JOINT_ROW)
    for i, (jname, abbr) in enumerate(JOINT_ROW):
        raw = joint_scores.get(jname)
        score_j = None if raw is None else float(raw)
        if score_j is not None and score_j < 55:
            any_low = True
        cx = int(origin_x + slot * i + slot / 2)
        cv2.circle(bottom, (cx, 38), 8, _score_color(score_j), -1, cv2.LINE_AA)
        label_size = cv2.getTextSize(abbr, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        cv2.putText(
            bottom, abbr,
            (cx - label_size[0] // 2, 68),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, _WHITE, 1, cv2.LINE_AA,
        )

    lines = _wrap_two_lines(feedback or "", 270, 0.4)
    for i, line in enumerate(lines):
        cv2.putText(
            bottom, line, (670, 36 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA,
        )
    if any_low:
        cv2.rectangle(bottom, (660, 8), (948, 82), _RED, 2)

    frame = np.vstack((top, middle, bottom))
    if frame.shape[1] != FRAME_SIZE[0] or frame.shape[0] != FRAME_SIZE[1]:
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
    return frame


class RecordingThread(threading.Thread):
    def __init__(self, output_path: str, fps: int = RECORD_FPS, frame_size=FRAME_SIZE):
        super().__init__(daemon=True)
        self.queue: queue.Queue = queue.Queue(maxsize=60)
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.frames_written = 0
        self.failed = False
        self.ready = threading.Event()

    def run(self):
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, float(self.fps), self.frame_size,
            )
            if self.writer is None or not self.writer.isOpened():
                self.failed = True
                print(f"[recording] VideoWriter failed to open: {self.output_path}")
                self.ready.set()
                return
            self.ready.set()
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                    frame = cv2.resize(frame, self.frame_size)
                self.writer.write(frame)
                self.frames_written += 1
        except Exception as exc:
            self.failed = True
            print(f"[recording] Recording thread error: {exc}")
            self.ready.set()
        finally:
            if self.writer is not None:
                try:
                    self.writer.release()
                except Exception:
                    pass

    def add_frame(self, frame):
        if self.failed:
            return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self):
        try:
            self.queue.put(None, timeout=2.0)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass


def load_index() -> List[dict]:
    path = index_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def append_recording_index(entry: dict) -> dict:
    """
    Append one recording. Returns stats computed from stored rows for this step.
    new_personal_best is true only when a prior recording for the same step
    exists and this overall_accuracy is higher.
    """
    step_id = entry.get("step_id") or ""
    with _INDEX_LOCK:
        rows = load_index()
        prior = [
            row for row in rows
            if (row.get("step_id") or "") == step_id
        ]
        prior_scores = [
            float(row["overall_accuracy"])
            for row in prior
            if row.get("overall_accuracy") is not None
        ]
        previous_best = max(prior_scores) if prior_scores else None
        current = float(entry.get("overall_accuracy") or 0.0)
        rows.append(entry)
        folder = ensure_recordings_dir()
        target = os.path.join(folder, "index.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
    same = [row for row in rows if (row.get("step_id") or "") == step_id]
    scores = [
        float(row["overall_accuracy"])
        for row in same
        if row.get("overall_accuracy") is not None
    ]
    return {
        "recordings_count": len(same),
        "best_accuracy": max(scores) if scores else current,
        "new_personal_best": previous_best is not None and current > previous_best,
    }


def mistake_label(joint_name: str) -> str:
    display = JOINT_DISPLAY_NAMES.get(joint_name, joint_name.replace("_", " "))
    return f"⚠ {display} needs correction"
