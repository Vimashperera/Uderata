"""
screen_preview.py  —  Screen 2: Expert Video Preview
LEFT video + controls | RIGHT info + CTA.
Plays the expert clip with its beat track (expert_display.wav).
The stage grows and shrinks with the window.
"""

import os
import time
from typing import Callable, Optional

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

from config import ensure_expert_audio, resolve_expert_audio_path
from ui.theme import C, font_display, font_ui

try:
    import pygame
    pygame.mixer.init()
    PYGAME = True
except Exception:
    PYGAME = False

# Poll often; frames advance from the media clock (not a fixed per-frame delay)
TICK_MS = 8


def _letterbox(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    res = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((th, tw, 3), dtype=np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = res
    return out


def _fmt(sec: float) -> str:
    return f"{int(sec)//60}:{int(sec)%60:02d}"


class PreviewScreen(ctk.CTkFrame):
    """Screen 2 — Expert Video Preview."""

    def __init__(self, master, video_path: str,
                 on_start_practice: Callable, on_back: Callable,
                 step_title: str = "Expert",
                 preview_blurb: str = "", **kwargs):
        super().__init__(master, fg_color=C["bg"], **kwargs)
        self.video_path = video_path
        self.on_start_practice = on_start_practice
        self.on_back = on_back
        self._step_title = step_title
        self._preview_blurb = preview_blurb or (
            "Study the expert's form carefully before starting practice."
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._playing = False
        self._speed = 1.0
        self._cur = 0
        self._total = 0
        self._fps = 30.0
        self._photo = None
        self._after = None
        self._audio_wav_path = resolve_expert_audio_path(video_path)
        self._audio_active = False
        self._play_origin = 0.0          # wall clock origin for A/V sync
        self._audio_start_offset = 0.0   # seconds into the track when play() was called
        self._pause_elapsed = 0.0        # media time frozen while paused
        self._vw = 640
        self._vh = 360

        self._build_ui()
        self.bind("<Configure>", self._schedule_fit)

    def configure_step(self, video_path: str, step_title: str, preview_blurb: str = "",
                       audio_path: str | None = None):
        """Point this screen at another step (menu selection)."""
        self._stop()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.video_path = video_path
        self._step_title = step_title
        self._preview_blurb = preview_blurb or self._preview_blurb
        self._audio_wav_path = audio_path or resolve_expert_audio_path(video_path)
        self._cur = 0
        self._total = 0
        self._playing = False
        self._speed = 1.0
        if hasattr(self, "_nav_title"):
            self._nav_title.configure(text=step_title, text_color=C["gold"])
        if hasattr(self, "_info_title"):
            self._info_title.configure(text=step_title)
        if hasattr(self, "_info_blurb"):
            self._info_blurb.configure(text=self._preview_blurb)
        if hasattr(self, "_play_btn"):
            self._play_btn.configure(text="Play")
        self.restyle()
        if hasattr(self, "_video_lbl"):
            self._video_lbl.configure(image=None, text="Press Play to begin")
        if hasattr(self, "_progress"):
            self._progress.set(0)
        if hasattr(self, "_time_lbl"):
            self._time_lbl.configure(text="0:00 / 0:00")

    def _build_ui(self):
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=2)

        self._rule = ctk.CTkFrame(self, fg_color=C["gold"], height=3, corner_radius=0)
        self._rule.grid(row=0, column=0, columnspan=3, sticky="ew")

        left = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0)
        left.grid(row=1, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        self._left = left

        nav = ctk.CTkFrame(left, fg_color="transparent")
        nav.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 4))

        ctk.CTkButton(
            nav, text="Steps", width=96, height=34,
            font=font_ui(13), fg_color=C["bg"],
            hover_color=C["elevated"], text_color=C["ivory"],
            border_width=1, border_color=C["divider"], corner_radius=4,
            command=self._on_back,
        ).pack(side="left")

        self._nav_title = ctk.CTkLabel(
            nav, text=self._step_title,
            text_color=C["gold"], font=font_ui(16, "bold"),
        )
        self._nav_title.pack(side="left", padx=16)

        stage_host = ctk.CTkFrame(left, fg_color="transparent")
        stage_host.grid(row=1, column=0, sticky="nsew", padx=20, pady=8)
        stage_host.columnconfigure(0, weight=1)
        stage_host.rowconfigure(0, weight=1)

        self._vc = ctk.CTkFrame(
            stage_host, fg_color=C["stage"], corner_radius=2,
            width=self._vw, height=self._vh,
        )
        self._vc.grid(row=0, column=0)
        self._vc.grid_propagate(False)

        self._video_lbl = ctk.CTkLabel(
            self._vc, text="Press Play to begin",
            fg_color=C["stage"], text_color=C["stage_text"],
            font=font_ui(14),
        )
        self._video_lbl.pack(fill="both", expand=True)

        self._progress = ctk.CTkProgressBar(
            left, width=self._vw, height=4,
            fg_color=C["divider"], progress_color=C["gold"], corner_radius=0,
        )
        self._progress.grid(row=2, column=0, pady=(6, 0))
        self._progress.set(0)

        ctrl_row = ctk.CTkFrame(left, fg_color="transparent")
        ctrl_row.grid(row=3, column=0, pady=(12, 2))

        self._play_btn = ctk.CTkButton(
            ctrl_row, text="Play", width=110, height=36,
            font=font_ui(13, "bold"),
            fg_color=C["gold"], hover_color=C["gold_hover"],
            text_color=C["ink"], corner_radius=4,
            command=self._toggle_play,
        )
        self._play_btn.grid(row=0, column=0, padx=4)

        self._replay_btn = ctk.CTkButton(
            ctrl_row, text="Replay", width=100, height=36,
            font=font_ui(13), fg_color=C["surface"],
            hover_color=C["elevated"], text_color=C["ivory"],
            border_width=1, border_color=C["divider"],
            corner_radius=4, command=self._replay,
        )
        self._replay_btn.grid(row=0, column=1, padx=4)

        self._speed_btn = ctk.CTkButton(
            ctrl_row, text="1×", width=72, height=36,
            font=font_ui(13), fg_color=C["surface"],
            hover_color=C["elevated"], text_color=C["ivory"],
            border_width=1, border_color=C["divider"],
            corner_radius=4, command=self._toggle_speed,
        )
        self._speed_btn.grid(row=0, column=2, padx=4)

        self._time_lbl = ctk.CTkLabel(
            ctrl_row, text="0:00 / 0:00",
            text_color=C["muted"], font=font_ui(13),
        )
        self._time_lbl.grid(row=0, column=3, padx=12)

        ctk.CTkLabel(
            left, text="Study the expert, then start practice.",
            text_color=C["muted"], font=font_ui(12),
        ).grid(row=4, column=0, pady=(2, 16))

        ctk.CTkFrame(self, fg_color=C["divider"], width=1, corner_radius=0).grid(
            row=1, column=1, sticky="ns"
        )

        right = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        right.grid(row=1, column=2, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self._right = right

        self._scroll = ctk.CTkScrollableFrame(right, fg_color="transparent", corner_radius=0)
        self._scroll.grid(row=0, column=0, sticky="nsew")
        self._build_info(self._scroll)

        cta = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=0, height=92)
        cta.grid(row=1, column=0, sticky="ew")
        cta.columnconfigure(0, weight=1)
        cta.grid_propagate(False)
        self._cta_bar = cta

        ctk.CTkFrame(cta, fg_color=C["divider"], height=1, corner_radius=0).grid(
            row=0, column=0, sticky="ew"
        )
        ctk.CTkLabel(
            cta, text="When the sequence is familiar, start practice.",
            text_color=C["muted"], font=font_ui(12),
        ).grid(row=1, column=0, pady=(10, 0))

        self._start_btn = ctk.CTkButton(
            cta, text="Start practice",
            height=40, font=font_ui(14, "bold"),
            fg_color=C["gold"], hover_color=C["gold_hover"],
            text_color=C["ink"], corner_radius=4,
            command=self._on_start,
        )
        self._start_btn.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 12))

    def _build_info(self, p: ctk.CTkScrollableFrame):
        self._wrap_labels = []
        self._info_title = ctk.CTkLabel(
            p, text=self._step_title,
            text_color=C["gold"], font=font_display(22, "bold"), anchor="w",
        )
        self._info_title.pack(fill="x", padx=20, pady=(20, 6))

        ctk.CTkFrame(p, fg_color=C["gold"], height=2, corner_radius=0).pack(
            fill="x", padx=20, pady=(0, 12)
        )

        self._info_blurb = ctk.CTkLabel(
            p,
            text=self._preview_blurb,
            text_color=C["ivory"], font=font_ui(13),
            wraplength=320, justify="left", anchor="w",
        )
        self._info_blurb.pack(fill="x", padx=20, pady=(0, 16))
        self._wrap_labels.append(self._info_blurb)

        ctk.CTkLabel(
            p, text="Key focus",
            text_color=C["ivory"], font=font_ui(14, "bold"), anchor="w",
        ).pack(fill="x", padx=20, pady=(0, 8))

        for title, desc in [
            ("Knee Bend Depth", "Bend both knees deeply — aim below 120°"),
            ("Arm Extension", "Fully sweep arms; keep elbows soft"),
            ("Torso Alignment", "Stay upright — no forward lean"),
        ]:
            row = ctk.CTkFrame(
                p, fg_color=C["card"], corner_radius=4,
                border_width=1, border_color=C["divider"],
            )
            row.pack(fill="x", padx=20, pady=4)
            col = ctk.CTkFrame(row, fg_color="transparent")
            col.pack(side="left", pady=12, padx=14, fill="x", expand=True)
            ctk.CTkLabel(
                col, text=title, text_color=C["ivory"],
                font=font_ui(13, "bold"), anchor="w",
            ).pack(anchor="w")
            desc_lbl = ctk.CTkLabel(
                col, text=desc, text_color=C["muted"],
                font=font_ui(12), anchor="w", wraplength=300, justify="left",
            )
            desc_lbl.pack(anchor="w", pady=(2, 0))
            self._wrap_labels.append(desc_lbl)

        ctk.CTkLabel(
            p, text="Tips",
            text_color=C["ivory"], font=font_ui(14, "bold"), anchor="w",
        ).pack(fill="x", padx=20, pady=(16, 8))

        for tip in [
            "Weight shifts from foot to foot on each beat.",
            "Arms reach full extension before pulling back.",
            "The torso stays centred — only limbs move expressively.",
        ]:
            row = ctk.CTkFrame(
                p, fg_color=C["tip"], corner_radius=4,
                border_width=1, border_color=C["border"],
            )
            row.pack(fill="x", padx=20, pady=4)
            tip_lbl = ctk.CTkLabel(
                row, text=tip, text_color=C["ivory"],
                font=font_ui(13), wraplength=300,
                justify="left", anchor="w",
            )
            tip_lbl.pack(anchor="w", padx=14, pady=10)
            self._wrap_labels.append(tip_lbl)

        ctk.CTkFrame(p, fg_color="transparent", height=12).pack()

    def restyle(self):
        """Recolor chrome after the active tradition changes. Playback state stays."""
        self.configure(fg_color=C["bg"])
        if hasattr(self, "_rule"):
            self._rule.configure(fg_color=C["gold"])
        if hasattr(self, "_left"):
            self._left.configure(fg_color=C["surface"])
        if hasattr(self, "_right"):
            self._right.configure(fg_color=C["bg"])
        if hasattr(self, "_nav_title"):
            self._nav_title.configure(text_color=C["gold"])
        if hasattr(self, "_progress"):
            self._progress.configure(progress_color=C["gold"], fg_color=C["divider"])
        if hasattr(self, "_play_btn"):
            self._play_btn.configure(
                fg_color=C["gold"], hover_color=C["gold_hover"], text_color=C["ink"],
            )
        if hasattr(self, "_start_btn"):
            self._start_btn.configure(
                fg_color=C["gold"], hover_color=C["gold_hover"], text_color=C["ink"],
            )
        if hasattr(self, "_cta_bar"):
            self._cta_bar.configure(fg_color=C["surface"])
        if hasattr(self, "_scroll"):
            for child in self._scroll.winfo_children():
                child.destroy()
            self._build_info(self._scroll)

    def _schedule_fit(self, event=None):
        if event is not None and event.widget is not self:
            return
        job = getattr(self, "_fit_job", None)
        if job is not None:
            try:
                self.after_cancel(job)
            except Exception:
                pass
        self._fit_job = self.after(70, self._apply_fit)

    def _apply_fit(self):
        self._fit_job = None
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 480 or h < 320:
            return
        pane_w = int(w * 0.60) - 56
        avail_h = h - 210
        vw = max(280, pane_w)
        vh = int(vw * 9 / 16)
        if vh > max(160, avail_h):
            vh = max(160, avail_h)
            vw = int(vh * 16 / 9)
        vw, vh = int(vw), int(vh)
        if abs(vw - self._vw) >= 10 or abs(vh - self._vh) >= 10:
            self._vw, self._vh = vw, vh
            self._vc.configure(width=vw, height=vh)
            self._progress.configure(width=max(vw, 240))
        wrap = max(200, int(w * 0.34) - 72)
        for lbl in getattr(self, "_wrap_labels", []):
            try:
                lbl.configure(wraplength=wrap)
            except Exception:
                pass


    def _open_video(self) -> bool:
        if not os.path.isfile(self.video_path):
            self._video_lbl.configure(
                text=f"Video not found:\n{self.video_path}", text_color=C["poor"])
            return False
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            self._video_lbl.configure(text="Cannot open video.", text_color=C["poor"])
            return False
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._cur = 0
        self._audio_wav_path = resolve_expert_audio_path(self.video_path)
        if not os.path.isfile(self._audio_wav_path):
            ensure_expert_audio(self.video_path)
        return True

    def _ensure_mixer(self) -> bool:
        if not PYGAME:
            return False
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            return pygame.mixer.get_init() is not None
        except Exception:
            try:
                pygame.mixer.init()
                return True
            except Exception:
                return False

    def _stop_audio(self):
        self._audio_active = False
        if not PYGAME:
            return
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def _arm_clock(self, from_sec: float = 0.0):
        """Align wall-clock origin so media time starts at from_sec."""
        self._audio_start_offset = max(0.0, float(from_sec))
        self._pause_elapsed = self._audio_start_offset
        # At speed s, media_time = (now - origin) * s  →  origin = now - media/s
        spd = max(self._speed, 1e-6)
        self._play_origin = time.perf_counter() - self._audio_start_offset / spd

    def _media_elapsed(self) -> float:
        """Seconds into the clip according to audio (preferred) or wall clock."""
        if self._speed == 1.0 and PYGAME and self._audio_active:
            try:
                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms >= 0:
                    return self._audio_start_offset + pos_ms / 1000.0
            except Exception:
                pass
        return (time.perf_counter() - self._play_origin) * self._speed

    def _start_audio(self, from_sec: float = 0.0):
        """Play beat track in sync with the expert video (1× only)."""
        self._stop_audio()
        self._arm_clock(from_sec)
        if self._speed != 1.0:
            return
        if not PYGAME or not os.path.isfile(self._audio_wav_path):
            return
        if not self._ensure_mixer():
            return
        try:
            pygame.mixer.music.load(self._audio_wav_path)
            try:
                pygame.mixer.music.play(loops=0, start=max(0.0, float(from_sec)))
            except TypeError:
                pygame.mixer.music.play(loops=0)
            self._audio_active = True
            # Re-arm after play() so origin matches audible start as closely as possible
            self._arm_clock(from_sec)
        except Exception:
            self._audio_active = False

    def _pause_audio(self):
        self._pause_elapsed = self._media_elapsed()
        if not PYGAME or not self._audio_active:
            return
        try:
            pygame.mixer.music.pause()
        except Exception:
            pass

    def _resume_audio(self):
        from_sec = self._pause_elapsed if self._pause_elapsed > 0 else (
            self._cur / max(self._fps, 1.0)
        )
        if self._speed != 1.0:
            self._arm_clock(from_sec)
            return
        if not PYGAME:
            self._arm_clock(from_sec)
            return
        if not self._audio_active:
            self._start_audio(from_sec=from_sec)
            return
        try:
            pygame.mixer.music.unpause()
            self._arm_clock(from_sec)
        except Exception:
            self._start_audio(from_sec=from_sec)

    def _show_frame(self, frame: np.ndarray):
        lb = _letterbox(frame, self._vw, self._vh)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        photo = ctk.CTkImage(
            light_image=pil_img, dark_image=pil_img,
            size=(self._vw, self._vh),
        )
        self._video_lbl.configure(image=photo, text="")
        self._photo = photo
        self._progress.set(self._cur / max(self._total, 1))
        self._time_lbl.configure(
            text=f"{_fmt(self._cur / self._fps)} / {_fmt(self._total / self._fps)}"
        )

    def _restart_loop(self):
        """Rewind video + audio to the start for seamless looping."""
        if self._cap is None:
            return False
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._cur = 0
        self._start_audio(from_sec=0.0)
        return True

    def _tick(self):
        if not self._playing or self._cap is None:
            return

        fps = max(self._fps, 1e-6)
        total = max(self._total, 1)
        elapsed = self._media_elapsed()
        duration = total / fps

        if elapsed >= duration:
            if not self._restart_loop():
                return
            elapsed = 0.0

        target = min(int(elapsed * fps), total - 1)

        # Catch up to the media clock (seek when far behind)
        frame = None
        behind = target - self._cur + 1
        if behind > 10:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
            self._cur = max(0, target)
            ret, frame = self._cap.read()
            if not ret:
                if not self._restart_loop():
                    return
                self._after = self.after(TICK_MS, self._tick)
                return
            self._cur = target + 1
        else:
            while self._cur <= target:
                ret, frame = self._cap.read()
                if not ret:
                    if not self._restart_loop():
                        return
                    self._after = self.after(TICK_MS, self._tick)
                    return
                self._cur += 1

        if frame is not None:
            self._show_frame(frame)

        self._after = self.after(TICK_MS, self._tick)

    def _toggle_play(self):
        if self._cap is None and not self._open_video():
            return
        self._playing = not self._playing
        self._play_btn.configure(text="Pause" if self._playing else "Play")
        if self._playing:
            if self._cur <= 1:
                self._start_audio(from_sec=0.0)
            else:
                self._resume_audio()
            self._tick()
        else:
            if self._after:
                self.after_cancel(self._after)
                self._after = None
            self._pause_audio()

    def _replay(self):
        if self._cap is None and not self._open_video():
            return
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._cur = 0
        self._start_audio(from_sec=0.0)
        if not self._playing:
            self._toggle_play()
        elif self._speed != 1.0:
            self._stop_audio()
            self._arm_clock(0.0)

    def _toggle_speed(self):
        # Freeze media time across the speed change
        at = self._media_elapsed() if self._playing else (
            self._cur / max(self._fps, 1.0)
        )
        self._speed = 0.5 if self._speed == 1.0 else 1.0
        self._speed_btn.configure(
            text="0.5×" if self._speed == 0.5 else "1×"
        )
        if not self._playing:
            self._pause_elapsed = at
            return
        if self._speed == 1.0:
            self._start_audio(from_sec=at)
        else:
            self._stop_audio()
            self._arm_clock(at)

    def _stop(self):
        self._playing = False
        if self._after:
            self.after_cancel(self._after)
            self._after = None
        self._stop_audio()
        self._pause_elapsed = 0.0

    def _on_back(self):
        self._stop()
        self.on_back()

    def _on_start(self):
        self._stop()
        self.on_start_practice()

    def on_show(self):
        self.restyle()
        self._schedule_fit()
        self._audio_wav_path = resolve_expert_audio_path(self.video_path)
        if self._cap is None:
            self._open_video()

    def on_hide(self):
        self._stop()

    def destroy(self):
        self._stop()
        if self._cap:
            self._cap.release()
        super().destroy()
