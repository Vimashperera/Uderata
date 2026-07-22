"""
screen_preview.py  —  Screen 2: Expert Video Preview
Black & gold layout: LEFT video + controls | RIGHT info + CTA.
"""

import os
from typing import Callable, Optional

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

from ui.theme import C, font_display, font_ui

VIDEO_MAX_W = 640
VIDEO_MAX_H = 360
FRAME_MS = 33


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
                 on_start_practice: Callable, on_back: Callable, **kwargs):
        super().__init__(master, fg_color=C["bg"], **kwargs)
        self.video_path = video_path
        self.on_start_practice = on_start_practice
        self.on_back = on_back

        self._cap: Optional[cv2.VideoCapture] = None
        self._playing = False
        self._speed = 1.0
        self._cur = 0
        self._total = 0
        self._fps = 30.0
        self._photo = None
        self._after = None

        self._build_ui()

    def _build_ui(self):
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=2)

        ctk.CTkFrame(self, fg_color=C["gold"], height=3, corner_radius=0).grid(
            row=0, column=0, columnspan=3, sticky="ew"
        )

        left = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0)
        left.grid(row=1, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        nav = ctk.CTkFrame(left, fg_color="transparent")
        nav.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 4))

        ctk.CTkButton(
            nav, text="← Menu", width=88, height=30,
            font=font_ui(10), fg_color="transparent",
            hover_color=C["elevated"], text_color=C["muted"],
            border_width=1, border_color=C["divider"], corner_radius=6,
            command=self._on_back,
        ).pack(side="left")

        ctk.CTkLabel(
            nav, text="Expert Reference — Pa Saramba 01",
            text_color=C["gold"], font=font_ui(13, "bold"),
        ).pack(side="left", padx=12)

        vc_outer = ctk.CTkFrame(left, fg_color="transparent")
        vc_outer.grid(row=1, column=0, pady=10)

        vc = ctk.CTkFrame(
            vc_outer, fg_color="#000000", corner_radius=6,
            border_width=1, border_color=C["border"],
            width=VIDEO_MAX_W, height=VIDEO_MAX_H,
        )
        vc.pack()
        vc.pack_propagate(False)

        self._video_lbl = ctk.CTkLabel(
            vc, text="Press Play to begin",
            fg_color="#000000", text_color=C["muted"],
            font=font_ui(12),
        )
        self._video_lbl.pack(fill="both", expand=True)

        prog_row = ctk.CTkFrame(left, fg_color="transparent")
        prog_row.grid(row=2, column=0, pady=(2, 0))
        self._progress = ctk.CTkProgressBar(
            prog_row, width=VIDEO_MAX_W, height=4,
            fg_color=C["divider"], progress_color=C["gold"], corner_radius=2,
        )
        self._progress.pack()
        self._progress.set(0)

        ctrl_row = ctk.CTkFrame(left, fg_color="transparent")
        ctrl_row.grid(row=3, column=0, pady=10)

        ctrl = ctk.CTkFrame(
            ctrl_row, fg_color=C["card"], corner_radius=10,
            border_width=1, border_color=C["divider"],
            width=VIDEO_MAX_W, height=56,
        )
        ctrl.pack()
        ctrl.pack_propagate(False)

        inner = ctk.CTkFrame(ctrl, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        self._play_btn = ctk.CTkButton(
            inner, text="▶  Play", width=115, height=36,
            font=font_ui(12, "bold"),
            fg_color=C["gold"], hover_color=C["gold_hover"],
            text_color=C["ink"], corner_radius=8,
            command=self._toggle_play,
        )
        self._play_btn.grid(row=0, column=0, padx=4)

        ctk.CTkButton(
            inner, text="↺  Replay", width=105, height=36,
            font=font_ui(11), fg_color=C["elevated"],
            hover_color=C["gold_deep"], text_color=C["ivory"],
            corner_radius=8, command=self._replay,
        ).grid(row=0, column=1, padx=4)

        self._speed_btn = ctk.CTkButton(
            inner, text="⚡ 1×", width=80, height=36,
            font=font_ui(11), fg_color=C["elevated"],
            hover_color=C["gold_deep"], text_color=C["ivory"],
            corner_radius=8, command=self._toggle_speed,
        )
        self._speed_btn.grid(row=0, column=2, padx=4)

        self._time_lbl = ctk.CTkLabel(
            inner, text="0:00 / 0:00",
            text_color=C["muted"], font=font_ui(10), width=90,
        )
        self._time_lbl.grid(row=0, column=3, padx=8)

        ctk.CTkLabel(
            left, text="Study the expert's form carefully before starting.",
            text_color=C["muted"], font=font_ui(10, "italic"),
        ).grid(row=4, column=0, pady=6)

        ctk.CTkFrame(self, fg_color=C["divider"], width=1, corner_radius=0).grid(
            row=1, column=1, sticky="ns"
        )

        right = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        right.grid(row=1, column=2, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(right, fg_color="transparent", corner_radius=0)
        scroll.grid(row=0, column=0, sticky="nsew")
        self._build_info(scroll)

        cta = ctk.CTkFrame(
            right, fg_color=C["surface"], corner_radius=0, height=72,
            border_width=0,
        )
        cta.grid(row=1, column=0, sticky="ew")
        cta.columnconfigure(0, weight=1)
        cta.grid_propagate(False)

        ctk.CTkFrame(cta, fg_color=C["gold_dim"], height=1, corner_radius=0).grid(
            row=0, column=0, sticky="ew"
        )
        ctk.CTkLabel(
            cta, text="When you feel confident, start practice.",
            text_color=C["muted"], font=font_ui(10, "italic"),
        ).grid(row=1, column=0, pady=(8, 0))

        ctk.CTkButton(
            cta, text="I'm Ready — Start Practice  →",
            height=38, font=font_ui(13, "bold"),
            fg_color=C["gold"], hover_color=C["gold_hover"],
            text_color=C["ink"], corner_radius=8,
            command=self._on_start,
        ).grid(row=2, column=0, sticky="ew", padx=14, pady=(4, 10))

    def _build_info(self, p: ctk.CTkScrollableFrame):
        ctk.CTkLabel(
            p, text="Pa Saramba 01",
            text_color=C["gold"], font=font_display(18, "bold"), anchor="w",
        ).pack(fill="x", padx=16, pady=(16, 4))

        ctk.CTkFrame(p, fg_color=C["gold"], height=2, corner_radius=0).pack(
            fill="x", padx=16, pady=(0, 10)
        )

        ctk.CTkLabel(
            p,
            text=(
                "Pa Saramba 01 is a core Udarata (Kandyan) movement. The fused expert "
                "timeline blends several master performances so you can match the tradition "
                "without copying a single dancer exactly. Focus on rhythm, posture, and "
                "clean lines."
            ),
            text_color=C["ivory"], font=font_ui(10),
            wraplength=330, justify="left", anchor="w",
        ).pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkLabel(
            p, text="Key Focus Points",
            text_color=C["gold"], font=font_ui(12, "bold"), anchor="w",
        ).pack(fill="x", padx=16, pady=(0, 6))

        for title, desc in [
            ("Knee Bend Depth", "Bend both knees deeply — aim below 120°"),
            ("Arm Extension", "Fully sweep arms; keep elbows soft"),
            ("Torso Alignment", "Stay upright — no forward lean"),
        ]:
            row = ctk.CTkFrame(
                p, fg_color=C["card"], corner_radius=8,
                border_width=1, border_color=C["divider"],
            )
            row.pack(fill="x", padx=16, pady=3)
            col = ctk.CTkFrame(row, fg_color="transparent")
            col.pack(side="left", pady=10, padx=12, fill="x", expand=True)
            ctk.CTkLabel(
                col, text=title, text_color=C["ivory"],
                font=font_ui(10, "bold"), anchor="w",
            ).pack(anchor="w")
            ctk.CTkLabel(
                col, text=desc, text_color=C["muted"],
                font=font_ui(9), anchor="w",
            ).pack(anchor="w")

        ctk.CTkLabel(
            p, text="Tips",
            text_color=C["gold"], font=font_ui(12, "bold"), anchor="w",
        ).pack(fill="x", padx=16, pady=(14, 6))

        for tip in [
            "Weight shifts from foot to foot on each beat.",
            "Arms reach full extension before pulling back.",
            "The torso stays centred — only limbs move expressively.",
        ]:
            row = ctk.CTkFrame(
                p, fg_color=C["tip"], corner_radius=8,
                border_width=1, border_color=C["gold_deep"],
            )
            row.pack(fill="x", padx=16, pady=3)
            ctk.CTkLabel(
                row, text="·  " + tip, text_color=C["ivory"],
                font=font_ui(10), wraplength=310,
                justify="left", anchor="w",
            ).pack(anchor="w", padx=10, pady=8)

        ctk.CTkFrame(p, fg_color="transparent", height=16).pack()

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
        return True

    def _tick(self):
        if not self._playing or self._cap is None:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._cur = 0
            ret, frame = self._cap.read()
            if not ret:
                return
        self._cur += 1

        lb = _letterbox(frame, VIDEO_MAX_W, VIDEO_MAX_H)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        photo = ctk.CTkImage(
            light_image=pil_img, dark_image=pil_img,
            size=(VIDEO_MAX_W, VIDEO_MAX_H),
        )
        self._video_lbl.configure(image=photo, text="")
        self._photo = photo
        self._progress.set(self._cur / max(self._total, 1))
        self._time_lbl.configure(
            text=f"{_fmt(self._cur/self._fps)} / {_fmt(self._total/self._fps)}"
        )
        self._after = self.after(max(1, int(FRAME_MS / self._speed)), self._tick)

    def _toggle_play(self):
        if self._cap is None and not self._open_video():
            return
        self._playing = not self._playing
        self._play_btn.configure(text="⏸  Pause" if self._playing else "▶  Play")
        if self._playing:
            self._tick()
        elif self._after:
            self.after_cancel(self._after)

    def _replay(self):
        if self._cap is None and not self._open_video():
            return
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._cur = 0
        if not self._playing:
            self._toggle_play()

    def _toggle_speed(self):
        self._speed = 0.5 if self._speed == 1.0 else 1.0
        self._speed_btn.configure(
            text=f"⚡ {'0.5×' if self._speed == 0.5 else '1×'}"
        )

    def _stop(self):
        self._playing = False
        if self._after:
            self.after_cancel(self._after)
            self._after = None

    def _on_back(self):
        self._stop()
        self.on_back()

    def _on_start(self):
        self._stop()
        self.on_start_practice()

    def on_show(self):
        if self._cap is None:
            self._open_video()

    def on_hide(self):
        self._stop()

    def destroy(self):
        self._stop()
        if self._cap:
            self._cap.release()
        super().destroy()
