"""
Local SQLite session history (single installation, no accounts).

Stores already-computed practice results only — does not change scoring.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

import config
from core.angle_calculator import ALL_JOINT_NAMES

_LOCK = threading.Lock()


def history_db_path() -> str:
    return os.path.join(config.APP_ROOT, "history.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(history_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _LOCK:
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    style_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    step_name TEXT,
                    form_score REAL NOT NULL,
                    timing_score REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    star_rating INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    avg_lag_ms REAL NOT NULL,
                    ended_by TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_step_time
                    ON sessions(step_id, timestamp);
                CREATE TABLE IF NOT EXISTS joint_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL
                        REFERENCES sessions(id) ON DELETE CASCADE,
                    joint_name TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    avg_deviation_deg REAL NOT NULL,
                    flagged INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_joint_scores_session
                    ON joint_scores(session_id);
                """
            )
            conn.commit()
        finally:
            conn.close()


def star_rating_from_form(form_overall: float) -> int:
    """Same thresholds as the Report screen (do not change scoring)."""
    if form_overall >= 90:
        return 5
    if form_overall >= 75:
        return 4
    if form_overall >= 60:
        return 3
    if form_overall >= 45:
        return 2
    return 1


def summarize_session(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive aggregate metrics from in-memory session_data using the same
    formulas as ReportScreen._compute (means only — no new scoring).
    """
    sd = session_data
    accs = sd.get("frame_accuracies", []) or []
    form_accs = sd.get("form_accuracies") or accs
    timing_accs = sd.get("timing_accuracies") or []
    lag_hist = sd.get("lag_ms_history") or []

    overall = float(np.mean(accs)) if accs else 0.0
    form_overall = float(np.mean(form_accs)) if form_accs else overall
    timing_overall = float(np.mean(timing_accs)) if timing_accs else 100.0
    avg_lag = float(np.mean(np.abs(lag_hist))) if lag_hist else 0.0

    jh = sd.get("joint_histories", {}) or {}
    jd = sd.get("joint_deviations", {}) or {}
    joint_acc: Dict[str, float] = {}
    joint_dev: Dict[str, float] = {}
    for jname in ALL_JOINT_NAMES:
        vals = jh.get(jname, []) or []
        joint_acc[jname] = float(np.mean(vals)) if vals else 0.0
        dev_vals = jd.get(jname, []) or []
        if dev_vals:
            joint_dev[jname] = round(float(np.mean(dev_vals)), 1)
        elif vals:
            joint_dev[jname] = round((100 - joint_acc[jname]) / 100 * 15, 1)
        else:
            joint_dev[jname] = 0.0

    sorted_joints = sorted(joint_acc.items(), key=lambda x: x[1])
    flagged = {name for name, _ in sorted_joints[:3]}

    return {
        "overall": overall,
        "form": form_overall,
        "timing": timing_overall,
        "avg_lag_ms": avg_lag,
        "stars": star_rating_from_form(form_overall),
        "joint_acc": joint_acc,
        "joint_dev": joint_dev,
        "flagged_joints": flagged,
        "duration": float(sd.get("duration_seconds", 0.0) or 0.0),
        "ended_by": sd.get("ended_by", "user"),
        "step_name": sd.get("step_name", "Step"),
        "step_id": sd.get("step_id") or "",
        "style_id": sd.get("style_id") or "",
    }


def get_previous_session(step_id: str) -> Optional[Dict[str, Any]]:
    """Most recent stored session for this step, or None."""
    if not step_id:
        return None
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                """
                SELECT * FROM sessions
                WHERE step_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (step_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def record_session(session_data: Dict[str, Any], summary: Optional[Dict[str, Any]] = None) -> int:
    """
    Persist one completed session. Returns new session id.
    Call after practice scoring finishes (not during live scoring).
    """
    init_db()
    summary = summary or summarize_session(session_data)
    step_id = summary.get("step_id") or session_data.get("step_id") or ""
    style_id = summary.get("style_id") or session_data.get("style_id") or ""
    if not step_id or not style_id:
        raise ValueError("session_data must include step_id and style_id to record history")

    ts = session_data.get("timestamp") or datetime.now(timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )

    with _LOCK:
        conn = _connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO sessions (
                    timestamp, style_id, step_id, step_name,
                    form_score, timing_score, overall_score, star_rating,
                    duration_seconds, avg_lag_ms, ended_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    style_id,
                    step_id,
                    summary.get("step_name") or session_data.get("step_name"),
                    float(summary["form"]),
                    float(summary["timing"]),
                    float(summary["overall"]),
                    int(summary["stars"]),
                    float(summary["duration"]),
                    float(summary["avg_lag_ms"]),
                    summary.get("ended_by"),
                ),
            )
            session_id = int(cur.lastrowid)
            joint_acc = summary.get("joint_acc") or {}
            joint_dev = summary.get("joint_dev") or {}
            flagged = summary.get("flagged_joints") or set()
            rows = [
                (
                    session_id,
                    jname,
                    float(joint_acc.get(jname, 0.0)),
                    float(joint_dev.get(jname, 0.0)),
                    1 if jname in flagged else 0,
                )
                for jname in ALL_JOINT_NAMES
            ]
            conn.executemany(
                """
                INSERT INTO joint_scores (
                    session_id, joint_name, accuracy, avg_deviation_deg, flagged
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return session_id
        finally:
            conn.close()


def list_sessions_for_step(step_id: str) -> List[Dict[str, Any]]:
    """All sessions for a step, oldest → newest."""
    if not step_id:
        return []
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM sessions
                WHERE step_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (step_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def step_aggregate_stats(step_id: str) -> Dict[str, Any]:
    sessions = list_sessions_for_step(step_id)
    if not sessions:
        return {
            "count": 0,
            "best_form": None,
            "best_overall": None,
            "latest_form": None,
            "latest_timing": None,
            "latest_overall": None,
        }
    forms = [float(s["form_score"]) for s in sessions]
    overalls = [float(s["overall_score"]) for s in sessions]
    latest = sessions[-1]
    return {
        "count": len(sessions),
        "best_form": max(forms),
        "best_overall": max(overalls),
        "latest_form": float(latest["form_score"]),
        "latest_timing": float(latest["timing_score"]),
        "latest_overall": float(latest["overall_score"]),
    }


def most_flagged_joints(step_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Joints most often flagged (top-3 weakest) across sessions for this step."""
    if not step_id:
        return []
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT js.joint_name AS joint_name,
                       SUM(js.flagged) AS flag_count,
                       AVG(js.accuracy) AS avg_accuracy
                FROM joint_scores js
                JOIN sessions s ON s.id = js.session_id
                WHERE s.step_id = ?
                GROUP BY js.joint_name
                HAVING flag_count > 0
                ORDER BY flag_count DESC, avg_accuracy ASC
                LIMIT ?
                """,
                (step_id, int(limit)),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def format_comparison_line(current_form: float, prior: Optional[Dict[str, Any]]) -> Optional[str]:
    """Factual vs-last-session line, or None if no prior session."""
    if not prior:
        return None
    prev = float(prior["form_score"])
    delta = current_form - prev
    sign = "+" if delta >= 0 else "–"
    # Use en-dash for negative display; magnitude always positive after sign
    mag = abs(delta)
    return f"Form: {current_form:.0f}% ({sign}{mag:.0f}% vs. last session)"


def format_timing_comparison_line(
    current_timing: float, prior: Optional[Dict[str, Any]]
) -> Optional[str]:
    if not prior:
        return None
    prev = float(prior["timing_score"])
    delta = current_timing - prev
    sign = "+" if delta >= 0 else "–"
    mag = abs(delta)
    return f"Timing: {current_timing:.0f}% ({sign}{mag:.0f}% vs. last session)"
