"""
screen_style.py — Choose a dance style before the step menu.
"""
import customtkinter as ctk
from typing import Callable, Optional

import config
from ui.theme import C, font_display, font_ui, palette, activate_style


class StyleScreen(ctk.CTkFrame):
    """Screen 0 — select a dance style from the STYLES registry."""

    def __init__(self, master, on_style_selected: Callable[[str], None],
                 on_history: Optional[Callable[[], None]] = None, **kwargs):
        super().__init__(master, fg_color=C["bg"], **kwargs)
        self.on_style_selected = on_style_selected
        self.on_history = on_history
        self._selected_style: Optional[str] = None
        self._style_cards = {}
        self._cols = 2
        self._build_ui()
        self.bind("<Configure>", self._on_resize)

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._rule = ctk.CTkFrame(self, fg_color=C["divider"], height=2, corner_radius=0)
        self._rule.grid(row=0, column=0, sticky="ew")

        body = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=880)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)
        self._body = body

        header = ctk.CTkFrame(body, fg_color="transparent")
        header.grid(row=1, column=1, sticky="ew", pady=(8, 8))
        header.columnconfigure(0, weight=1)
        self._header = header

        ctk.CTkLabel(
            header,
            text=config.APP_NAME,
            text_color=C["ivory"],
            font=font_display(28, "bold"),
            anchor="w",
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        if self.on_history:
            ctk.CTkButton(
                header,
                text="Progress",
                width=120,
                height=34,
                corner_radius=4,
                fg_color=C["surface"],
                hover_color=C["elevated"],
                text_color=C["ivory"],
                border_width=1,
                border_color=C["divider"],
                font=font_ui(13),
                command=self.on_history,
            ).grid(row=0, column=1, sticky="e", padx=(16, 0))

        ctk.CTkLabel(
            header,
            text="Choose a tradition",
            text_color=C["ivory"],
            font=font_ui(16),
            anchor="w",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(14, 0))

        ctk.CTkLabel(
            header,
            text="Then pick a step, watch the expert, and practice.",
            text_color=C["muted"],
            font=font_ui(13),
            anchor="w",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self._board = ctk.CTkFrame(body, fg_color="transparent")
        self._board.grid(row=2, column=1, sticky="ew", pady=(16, 8))
        self._board.columnconfigure(0, weight=1)
        self._board.columnconfigure(1, weight=1)

        for i, style in enumerate(config.list_styles()):
            self._add_style_card(
                row=i // 2,
                col=i % 2,
                style_id=style["id"],
                title=style["title"],
                subtitle=style["subtitle"],
                step_count=len(style["step_order"]),
            )

        cta = ctk.CTkFrame(body, fg_color="transparent")
        cta.grid(row=3, column=1, sticky="ew", pady=(12, 8))
        cta.columnconfigure(0, weight=1)
        self._cta_host = cta

        self._cta_btn = ctk.CTkButton(
            cta,
            text="Continue",
            font=font_ui(15, "bold"),
            height=46,
            corner_radius=4,
            fg_color=C["elevated"],
            hover_color=C["elevated"],
            text_color=C["muted"],
            state="disabled",
            command=self._on_continue,
        )
        self._cta_btn.grid(row=0, column=0, sticky="ew")

        self._hint = ctk.CTkLabel(
            cta,
            text="Select a tradition to continue",
            text_color=C["muted"],
            font=font_ui(12),
        )
        self._hint.grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _add_style_card(self, row, col, style_id, title, subtitle, step_count):
        pal = palette(style_id)
        frame = ctk.CTkFrame(
            self._board,
            fg_color=pal["tip"],
            corner_radius=4,
            border_width=1,
            border_color=pal["border"],
            cursor="hand2",
        )
        frame.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)
        frame.grid_propagate(False)
        frame.configure(height=176)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        band = ctk.CTkFrame(frame, width=8, fg_color=pal["gold"], corner_radius=0)
        band.grid(row=0, column=0, sticky="ns")
        band.grid_propagate(False)

        text = ctk.CTkFrame(frame, fg_color="transparent")
        text.grid(row=0, column=1, sticky="nsew", padx=(18, 20), pady=22)
        text.columnconfigure(0, weight=1)
        text.rowconfigure(2, weight=1)

        title_lbl = ctk.CTkLabel(
            text,
            text=title,
            text_color=pal["gold"],
            font=font_display(26, "bold"),
            anchor="w",
            justify="left",
        )
        title_lbl.grid(row=0, column=0, sticky="w")

        meta_lbl = ctk.CTkLabel(
            text,
            text=subtitle,
            text_color=C["ivory"],
            font=font_ui(14),
            anchor="w",
        )
        meta_lbl.grid(row=1, column=0, sticky="w", pady=(6, 0))

        count_lbl = ctk.CTkLabel(
            text,
            text=f"{step_count} steps",
            text_color=C["muted"],
            font=font_ui(13),
            anchor="w",
        )
        count_lbl.grid(row=2, column=0, sticky="sw", pady=(16, 0))

        def on_enter(_event, f=frame, p=pal):
            if self._selected_style != style_id:
                f.configure(border_color=p["gold_dim"])

        def on_leave(_event, f=frame, p=pal):
            if self._selected_style != style_id:
                f.configure(border_color=p["border"])

        def on_click(_event, sid=style_id):
            self._select_style(sid)

        for widget in (frame, band, text, title_lbl, meta_lbl, count_lbl):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", on_click)

        self._style_cards[style_id] = {
            "frame": frame,
            "palette": pal,
        }

    def _on_resize(self, event):
        if event.widget is not self:
            return
        span = min(1100, max(640, event.width - 80))
        self._body.columnconfigure(1, minsize=span)
        cols = 1 if event.width < 860 else 2
        self._fit_title(span)
        if cols == self._cols:
            return
        self._cols = cols
        ids = list(self._style_cards.keys())
        for i, sid in enumerate(ids):
            frame = self._style_cards[sid]["frame"]
            if cols == 1:
                r, c = i, 0
            else:
                r, c = divmod(i, 2)
            frame.grid(row=r, column=c, sticky="nsew", padx=8, pady=8)
        self._board.columnconfigure(1, weight=1 if cols == 2 else 0)

    def _fit_title(self, width: int):
        wrap = max(280, width - 160)
        for child in self._header.winfo_children():
            if isinstance(child, ctk.CTkLabel) and child.cget("text") == config.APP_NAME:
                child.configure(wraplength=wrap)
                break

    def _select_style(self, style_id: str):
        for sid, item in self._style_cards.items():
            pal = item["palette"]
            frame = item["frame"]
            if sid == style_id:
                frame.configure(border_color=pal["gold"], border_width=3)
            else:
                frame.configure(border_color=pal["border"], border_width=1)

        self._selected_style = style_id
        pal = self._style_cards[style_id]["palette"]
        self._cta_btn.configure(
            state="normal",
            fg_color=pal["gold"],
            hover_color=pal["gold_hover"],
            text_color=pal["ink"],
        )
        style = config.get_style(style_id)
        self._hint.configure(text=style["subtitle"])

    def _on_continue(self):
        if self._selected_style:
            self.on_style_selected(self._selected_style)

    def on_show(self):
        activate_style(None)
        self.configure(fg_color=C["bg"])

    def on_hide(self):
        pass
