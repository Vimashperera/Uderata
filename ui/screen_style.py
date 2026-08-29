"""
screen_style.py — Choose a dance style before the step menu.
"""
import customtkinter as ctk
from typing import Callable, Optional

import config
from ui.theme import C, font_display, font_ui

MENU_CONTENT_W = 960


class StyleScreen(ctk.CTkFrame):
    """Screen 0 — select a dance style from the STYLES registry."""

    def __init__(self, master, on_style_selected: Callable[[str], None],
                 on_history: Optional[Callable[[], None]] = None, **kwargs):
        super().__init__(master, fg_color=C["bg"], **kwargs)
        self.on_style_selected = on_style_selected
        self.on_history = on_history
        self._selected_style: Optional[str] = None
        self._style_cards = {}
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

        center = ctk.CTkFrame(
            content, fg_color="transparent", width=MENU_CONTENT_W, height=620,
        )
        center.grid(row=0, column=0)
        center.grid_propagate(False)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(2, weight=1)

        self.after(50, lambda: self._size_center(center, content))

        header = ctk.CTkFrame(center, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(48, 2))

        ctk.CTkLabel(
            header,
            text=config.APP_NAME.upper(),
            text_color=C["gold"],
            font=font_display(28, "bold"),
            wraplength=MENU_CONTENT_W - 40,
        ).pack()

        ctk.CTkLabel(
            header,
            text="Choose a dance style to begin",
            text_color=C["ivory"],
            font=font_display(16, "italic"),
        ).pack(pady=(10, 4))

        ctk.CTkLabel(
            header,
            text="Live form & timing feedback",
            text_color=C["muted"],
            font=font_ui(11),
        ).pack()

        if self.on_history:
            ctk.CTkButton(
                header,
                text="My Progress",
                width=130,
                height=30,
                corner_radius=6,
                fg_color=C["elevated"],
                hover_color=C["card"],
                text_color=C["gold"],
                border_width=1,
                border_color=C["divider"],
                font=font_ui(11, "bold"),
                command=self.on_history,
            ).pack(pady=(12, 0))

        ctk.CTkFrame(center, fg_color=C["gold_dim"], height=1, corner_radius=0).grid(
            row=1, column=0, sticky="ew", pady=(16, 20)
        )

        card = ctk.CTkFrame(
            center,
            fg_color=C["surface"],
            corner_radius=14,
            border_width=1,
            border_color=C["border"],
        )
        card.grid(row=2, column=0, sticky="nsew", pady=4)
        card.columnconfigure(0, weight=1)

        header_row = ctk.CTkFrame(card, fg_color=C["elevated"], corner_radius=10)
        header_row.grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        ctk.CTkLabel(
            header_row,
            text="Dance styles",
            text_color=C["gold"],
            font=font_ui(13, "bold"),
            anchor="w",
        ).pack(side="left", padx=20, pady=10)

        card.rowconfigure(1, weight=1)
        list_frame = ctk.CTkScrollableFrame(
            card,
            fg_color="transparent",
            corner_radius=0,
            width=MENU_CONTENT_W - 40,
        )
        list_frame.grid(row=1, column=0, sticky="nsew", padx=14, pady=(8, 16))
        list_frame.columnconfigure(0, weight=1)

        for i, style in enumerate(config.list_styles()):
            self._add_style_card(
                list_frame,
                row=i,
                style_id=style["id"],
                title=style["title"],
                subtitle=style["subtitle"],
                step_count=len(style["step_order"]),
            )

        cta = ctk.CTkFrame(center, fg_color="transparent")
        cta.grid(row=3, column=0, sticky="ew", pady=(18, 28))

        self._cta_btn = ctk.CTkButton(
            cta,
            text="Continue to Steps  →",
            font=font_ui(14, "bold"),
            height=48,
            corner_radius=8,
            fg_color=C["elevated"],
            hover_color=C["elevated"],
            text_color=C["muted"],
            border_width=1,
            border_color=C["divider"],
            state="disabled",
            command=self._on_continue,
        )
        self._cta_btn.pack(fill="x")

        ctk.CTkLabel(
            cta,
            text="Select a style above to continue",
            text_color=C["muted"],
            font=font_ui(10, "italic"),
        ).pack(pady=(8, 0))

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

    def _add_style_card(self, parent, row, style_id, title, subtitle, step_count):
        frame = ctk.CTkFrame(
            parent,
            fg_color=C["card"],
            corner_radius=12,
            border_width=1,
            border_color=C["divider"],
            cursor="hand2",
            height=88,
        )
        frame.grid(row=row, column=0, sticky="ew", pady=6)
        frame.grid_propagate(False)
        frame.columnconfigure(0, weight=1)

        title_lbl = ctk.CTkLabel(
            frame,
            text=title,
            text_color=C["ivory"],
            font=font_ui(18, "bold"),
            anchor="w",
        )
        title_lbl.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 2))

        meta_lbl = ctk.CTkLabel(
            frame,
            text=f"{subtitle}  ·  {step_count} steps",
            text_color=C["muted"],
            font=font_ui(11),
            anchor="w",
        )
        meta_lbl.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 16))

        def on_enter(_):
            if self._selected_style != style_id:
                frame.configure(border_color=C["gold_dim"])

        def on_leave(_):
            if self._selected_style != style_id:
                frame.configure(border_color=C["divider"])

        def on_click(_):
            self._select_style(style_id)

        for widget in (frame, title_lbl, meta_lbl):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", on_click)

        self._style_cards[style_id] = frame

    def _select_style(self, style_id: str):
        for sid, frame in self._style_cards.items():
            if sid == style_id:
                frame.configure(border_color=C["gold"], fg_color=C["step_active"])
            else:
                frame.configure(border_color=C["divider"], fg_color=C["card"])

        self._selected_style = style_id
        self._cta_btn.configure(
            state="normal",
            fg_color=C["gold"],
            hover_color=C["gold_hover"],
            text_color=C["ink"],
        )

    def _on_continue(self):
        if self._selected_style:
            self.on_style_selected(self._selected_style)

    def on_show(self):
        pass

    def on_hide(self):
        pass
