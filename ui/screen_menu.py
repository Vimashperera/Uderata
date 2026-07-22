"""
screen_menu.py — Welcome screen (Udarata / Pa Saramba 01).
Black & gold brand-first home.
"""
import customtkinter as ctk
from tkinter import Canvas
import tkinter as tk
from typing import Callable, Optional

import config
from ui.theme import C, font_display, font_ui


class MenuScreen(ctk.CTkFrame):
    """Screen 1 — select Pa Saramba 01 and begin."""

    def __init__(self, master, on_begin: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color=C["bg"], **kwargs)
        self.on_begin = on_begin
        self._selected_step: Optional[str] = None
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        # Thin gold crown line
        ctk.CTkFrame(self, fg_color=C["gold"], height=3, corner_radius=0).grid(
            row=0, column=0, sticky="ew"
        )

        content = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        content.rowconfigure(1, weight=0)
        content.rowconfigure(2, weight=1)

        center = ctk.CTkFrame(content, fg_color="transparent")
        center.grid(row=1, column=0)
        center.columnconfigure(0, weight=1)

        self._build_header(center)
        ctk.CTkFrame(center, fg_color=C["gold_dim"], height=1, corner_radius=0).grid(
            row=1, column=0, sticky="ew", padx=80, pady=(8, 18)
        )
        self._build_style_card(center)
        self._build_cta(center)

        ctk.CTkFrame(self, fg_color=C["gold_deep"], height=3, corner_radius=0).grid(
            row=2, column=0, sticky="ew"
        )

    def _build_header(self, parent):
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, padx=60, pady=(40, 4))

        ctk.CTkLabel(
            header,
            text="UDARATA",
            text_color=C["gold"],
            font=font_display(42, "bold"),
        ).pack()

        ctk.CTkLabel(
            header,
            text="Pa Saramba Coaching",
            text_color=C["ivory"],
            font=font_display(16, "italic"),
        ).pack(pady=(4, 10))

        ctk.CTkLabel(
            header,
            text=f"{config.DANCE_STYLE}  ·  Live form & timing feedback",
            text_color=C["muted"],
            font=font_ui(11),
        ).pack()

    def _build_style_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=C["surface"],
            corner_radius=14,
            border_width=1,
            border_color=C["border"],
        )
        card.grid(row=2, column=0, padx=60, pady=4, sticky="ew")
        card.columnconfigure(0, weight=1)

        header_row = ctk.CTkFrame(card, fg_color=C["elevated"], corner_radius=10)
        header_row.grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        ctk.CTkLabel(
            header_row,
            text="Choose your step",
            text_color=C["gold"],
            font=font_ui(13, "bold"),
            anchor="w",
        ).pack(side="left", padx=20, pady=12)

        ctk.CTkLabel(
            header_row,
            text="Kandyan tradition",
            text_color=C["muted"],
            font=font_ui(10, "italic"),
        ).pack(side="right", padx=20)

        steps_frame = ctk.CTkFrame(card, fg_color="transparent")
        steps_frame.grid(row=1, column=0, padx=16, pady=16, sticky="ew")
        steps_frame.columnconfigure(0, weight=1)

        self._step_cards = {}
        self._add_step_card(
            steps_frame,
            row=0,
            step_id=config.STEP_ID,
            title=config.STEP_TITLE,
            description=(
                "Practice against a fused expert timeline built from master performances. "
                "Match joint angles, bone lines, and musical timing."
            ),
            tags=["Legs", "Arms", "Torso"],
            difficulty="Focus",
        )
        self._add_coming_soon_card(
            steps_frame, row=1, title="More Udarata steps — coming later"
        )

    def _add_step_card(self, parent, row, step_id, title, description, tags, difficulty):
        frame = ctk.CTkFrame(
            parent,
            fg_color=C["card"],
            corner_radius=12,
            border_width=1,
            border_color=C["divider"],
            cursor="hand2",
        )
        frame.grid(row=row, column=0, sticky="ew", padx=4, pady=6)
        frame.columnconfigure(1, weight=1)

        radio_canvas = Canvas(
            frame, width=22, height=22, bg=C["card"], highlightthickness=0
        )
        radio_canvas.grid(row=0, column=0, rowspan=2, padx=(14, 8), pady=14)
        radio_canvas.create_oval(2, 2, 20, 20, outline=C["gold_dim"], width=2)

        title_lbl = ctk.CTkLabel(
            frame,
            text=title,
            text_color=C["ivory"],
            font=font_ui(14, "bold"),
            anchor="w",
        )
        title_lbl.grid(row=0, column=1, sticky="w", padx=(4, 8), pady=(12, 2))

        desc_lbl = ctk.CTkLabel(
            frame,
            text=description,
            text_color=C["muted"],
            font=font_ui(10),
            anchor="w",
            wraplength=520,
            justify="left",
        )
        desc_lbl.grid(row=1, column=1, sticky="w", padx=(4, 8), pady=(0, 8))

        meta_row = ctk.CTkFrame(frame, fg_color="transparent")
        meta_row.grid(row=2, column=1, sticky="w", padx=(4, 8), pady=(0, 12))

        for tag in tags:
            ctk.CTkLabel(
                meta_row,
                text=f"  {tag}  ",
                text_color=C["gold"],
                fg_color=C["tip"],
                font=font_ui(9, "bold"),
                corner_radius=6,
            ).pack(side="left", padx=3)

        ctk.CTkLabel(
            meta_row,
            text=f"· {difficulty}",
            text_color=C["muted"],
            font=font_ui(9),
        ).pack(side="left", padx=(12, 0))

        def on_enter(_):
            if self._selected_step != step_id:
                frame.configure(border_color=C["gold_dim"])

        def on_leave(_):
            if self._selected_step != step_id:
                frame.configure(border_color=C["divider"])

        def on_click(_):
            self._select_step(step_id, frame, radio_canvas)

        for widget in [frame, title_lbl, desc_lbl, radio_canvas, meta_row]:
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", on_click)

        self._step_cards[step_id] = {"frame": frame, "radio_canvas": radio_canvas}

    def _add_coming_soon_card(self, parent, row, title):
        frame = ctk.CTkFrame(
            parent,
            fg_color=C["bg"],
            corner_radius=12,
            border_width=1,
            border_color=C["divider"],
        )
        frame.grid(row=row, column=0, sticky="ew", padx=4, pady=6)
        ctk.CTkLabel(
            frame,
            text=title,
            text_color=C["muted"],
            font=font_ui(11, "italic"),
        ).pack(padx=20, pady=14)

    def _select_step(self, step_id: str, frame: ctk.CTkFrame, radio_canvas: Canvas):
        for sid, widgets in self._step_cards.items():
            widgets["frame"].configure(
                border_color=C["divider"], fg_color=C["card"]
            )
            widgets["radio_canvas"].delete("fill")

        self._selected_step = step_id
        frame.configure(border_color=C["gold"], fg_color=C["step_active"])
        radio_canvas.delete("fill")
        radio_canvas.create_oval(
            6, 6, 16, 16, fill=C["gold"], outline="", tags="fill"
        )

        if hasattr(self, "_cta_btn"):
            self._cta_btn.configure(
                state="normal",
                fg_color=C["gold"],
                hover_color=C["gold_hover"],
                text_color=C["ink"],
            )

    def _build_cta(self, parent):
        cta_frame = ctk.CTkFrame(parent, fg_color="transparent")
        cta_frame.grid(row=3, column=0, pady=(22, 48))

        self._cta_btn = ctk.CTkButton(
            cta_frame,
            text="Begin Learning  →",
            font=font_ui(14, "bold"),
            width=280,
            height=50,
            corner_radius=8,
            fg_color=C["elevated"],
            hover_color=C["elevated"],
            text_color=C["muted"],
            border_width=1,
            border_color=C["divider"],
            state="disabled",
            command=self._on_begin_clicked,
        )
        self._cta_btn.pack()

        ctk.CTkLabel(
            cta_frame,
            text="Select the step above to continue",
            text_color=C["muted"],
            font=font_ui(10, "italic"),
        ).pack(pady=(10, 0))

    def _on_begin_clicked(self):
        if self._selected_step:
            self.on_begin(self._selected_step)
