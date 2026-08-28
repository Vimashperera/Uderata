"""
screen_menu.py — Step selection for the active dance style.
Black & gold brand-first home.
"""
import customtkinter as ctk
from tkinter import Canvas
from typing import Callable, Optional

import config
from ui.theme import C, font_display, font_ui

# Content column width inside the 1280px window
MENU_CONTENT_W = 960
STEP_DESC_WRAP = 820


class MenuScreen(ctk.CTkFrame):
    """Screen 1 — select a step and begin (filtered by active style)."""

    def __init__(
        self,
        master,
        on_begin: Callable[[str], None],
        on_back: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color=C["bg"], **kwargs)
        self.on_begin = on_begin
        self.on_back = on_back
        self._selected_step: Optional[str] = None
        self._style_id = config.STYLE_ID
        self._step_cards = {}
        self._steps_host = None
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        ctk.CTkFrame(self, fg_color=C["gold"], height=3, corner_radius=0).grid(
            row=0, column=0, sticky="ew"
        )

        content = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        # Fixed-width centered column so the step list never collapses
        center = ctk.CTkFrame(
            content, fg_color="transparent", width=MENU_CONTENT_W, height=620,
        )
        center.grid(row=0, column=0)
        center.grid_propagate(False)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(2, weight=1)

        self.after(50, lambda: self._size_center(center, content))

        self._build_header(center)
        ctk.CTkFrame(center, fg_color=C["gold_dim"], height=1, corner_radius=0).grid(
            row=1, column=0, sticky="ew", pady=(4, 12)
        )
        self._build_style_card(center)
        self._build_cta(center)

        ctk.CTkFrame(self, fg_color=C["gold_deep"], height=3, corner_radius=0).grid(
            row=2, column=0, sticky="ew"
        )

    def _size_center(self, center: ctk.CTkFrame, content: ctk.CTkFrame):
        try:
            content.update_idletasks()
            h = max(content.winfo_height() - 8, 520)
            center.configure(height=h)
        except Exception:
            center.configure(height=620)

    def _build_header(self, parent):
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(20, 2))

        top = ctk.CTkFrame(header, fg_color="transparent")
        top.pack(fill="x")

        if self.on_back:
            ctk.CTkButton(
                top,
                text="← Styles",
                width=100,
                height=28,
                corner_radius=6,
                fg_color=C["elevated"],
                hover_color=C["card"],
                text_color=C["gold"],
                border_width=1,
                border_color=C["divider"],
                font=font_ui(11),
                command=self.on_back,
            ).pack(side="left", padx=(0, 8))

        style = config.get_style(self._style_id)

        self._brand_lbl = ctk.CTkLabel(
            header,
            text=style["short_title"].upper(),
            text_color=C["gold"],
            font=font_display(42, "bold"),
        )
        self._brand_lbl.pack(pady=(8, 0))

        self._heading_lbl = ctk.CTkLabel(
            header,
            text=style.get("menu_heading", f"{style['short_title']} Step Coaching"),
            text_color=C["ivory"],
            font=font_display(16, "italic"),
        )
        self._heading_lbl.pack(pady=(2, 4))

        self._meta_lbl = ctk.CTkLabel(
            header,
            text=f"{style['title']}  ·  Live form & timing feedback",
            text_color=C["muted"],
            font=font_ui(11),
        )
        self._meta_lbl.pack()

    def _build_style_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=C["surface"],
            corner_radius=14,
            border_width=1,
            border_color=C["border"],
        )
        card.grid(row=2, column=0, sticky="nsew", pady=4)
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        header_row = ctk.CTkFrame(card, fg_color=C["elevated"], corner_radius=10)
        header_row.grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        ctk.CTkLabel(
            header_row,
            text="Choose your step",
            text_color=C["gold"],
            font=font_ui(13, "bold"),
            anchor="w",
        ).pack(side="left", padx=20, pady=10)

        style = config.get_style(self._style_id)
        self._tradition_lbl = ctk.CTkLabel(
            header_row,
            text=style["subtitle"],
            text_color=C["muted"],
            font=font_ui(10, "italic"),
        )
        self._tradition_lbl.pack(side="right", padx=20)

        self._steps_host = ctk.CTkScrollableFrame(
            card,
            fg_color="transparent",
            corner_radius=0,
            width=MENU_CONTENT_W - 28,
        )
        self._steps_host.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self._steps_host.columnconfigure(0, weight=1)

        self._rebuild_step_cards()

    def _rebuild_step_cards(self):
        if self._steps_host is None:
            return
        for child in self._steps_host.winfo_children():
            child.destroy()
        self._step_cards = {}
        self._selected_step = None

        for i, step in enumerate(config.list_steps(self._style_id)):
            self._add_step_card(
                self._steps_host,
                row=i,
                step_id=step["id"],
                title=step["title"],
                description=step["description"],
                tags=step.get("tags", []),
                difficulty=step.get("difficulty", ""),
            )

        if hasattr(self, "_cta_btn"):
            self._cta_btn.configure(
                state="disabled",
                fg_color=C["elevated"],
                hover_color=C["elevated"],
                text_color=C["muted"],
            )

    def configure_style(self, style_id: str):
        """Reload the step list for the chosen dance style."""
        self._style_id = style_id
        style = config.get_style(style_id)
        if hasattr(self, "_brand_lbl"):
            self._brand_lbl.configure(text=style["short_title"].upper())
        if hasattr(self, "_heading_lbl"):
            self._heading_lbl.configure(
                text=style.get("menu_heading", f"{style['short_title']} Step Coaching")
            )
        if hasattr(self, "_meta_lbl"):
            self._meta_lbl.configure(
                text=f"{style['title']}  ·  Live form & timing feedback"
            )
        if hasattr(self, "_tradition_lbl"):
            self._tradition_lbl.configure(text=style["subtitle"])
        self._rebuild_step_cards()

    def _add_step_card(self, parent, row, step_id, title, description, tags, difficulty):
        frame = ctk.CTkFrame(
            parent,
            fg_color=C["card"],
            corner_radius=12,
            border_width=1,
            border_color=C["divider"],
            cursor="hand2",
        )
        frame.grid(row=row, column=0, sticky="ew", padx=6, pady=5)
        frame.columnconfigure(1, weight=1)

        radio_canvas = Canvas(
            frame, width=22, height=22, bg=C["card"], highlightthickness=0
        )
        radio_canvas.grid(row=0, column=0, rowspan=2, padx=(14, 8), pady=12)
        radio_canvas.create_oval(2, 2, 20, 20, outline=C["gold_dim"], width=2)

        title_lbl = ctk.CTkLabel(
            frame,
            text=title,
            text_color=C["ivory"],
            font=font_ui(14, "bold"),
            anchor="w",
        )
        title_lbl.grid(row=0, column=1, sticky="ew", padx=(4, 16), pady=(10, 2))

        desc_lbl = ctk.CTkLabel(
            frame,
            text=description,
            text_color=C["muted"],
            font=font_ui(10),
            anchor="w",
            wraplength=STEP_DESC_WRAP,
            justify="left",
        )
        desc_lbl.grid(row=1, column=1, sticky="ew", padx=(4, 16), pady=(0, 4))

        meta_row = ctk.CTkFrame(frame, fg_color="transparent")
        meta_row.grid(row=2, column=1, sticky="w", padx=(4, 16), pady=(0, 10))

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
        cta_frame.grid(row=3, column=0, sticky="ew", pady=(14, 24))

        self._cta_btn = ctk.CTkButton(
            cta_frame,
            text="Begin Learning  →",
            font=font_ui(14, "bold"),
            height=48,
            corner_radius=8,
            fg_color=C["elevated"],
            hover_color=C["elevated"],
            text_color=C["muted"],
            border_width=1,
            border_color=C["divider"],
            state="disabled",
            command=self._on_begin_clicked,
        )
        self._cta_btn.pack(fill="x")

        ctk.CTkLabel(
            cta_frame,
            text="Select the step above to continue",
            text_color=C["muted"],
            font=font_ui(10, "italic"),
        ).pack(pady=(8, 0))

    def _on_begin_clicked(self):
        if self._selected_step:
            self.on_begin(self._selected_step)

    def on_show(self):
        pass

    def on_hide(self):
        pass
