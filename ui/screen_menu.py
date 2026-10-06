"""
screen_menu.py — Step selection for the active dance style.
"""
import customtkinter as ctk
from typing import Callable, Optional

import config
from ui.theme import C, font_display, font_ui


class MenuScreen(ctk.CTkFrame):
    """Screen 1 — select a step and begin (filtered by active style)."""

    def __init__(
        self,
        master,
        on_begin: Callable[[str], None],
        on_back: Optional[Callable[[], None]] = None,
        on_history: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color=C["bg"], **kwargs)
        self.on_begin = on_begin
        self.on_back = on_back
        self.on_history = on_history
        self._selected_step: Optional[str] = None
        self._style_id = config.STYLE_ID
        self._step_cards = {}
        self._desc_labels = []
        self._build_ui()
        self.bind("<Configure>", self._on_resize)

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._rule = ctk.CTkFrame(self, fg_color=C["gold"], height=3, corner_radius=0)
        self._rule.grid(row=0, column=0, sticky="ew")

        body = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        self._body = body

        self._build_header(body)

        self._list_holder = ctk.CTkFrame(body, fg_color="transparent")
        self._list_holder.grid(row=1, column=0, sticky="nsew", pady=(4, 4))
        self._list_holder.rowconfigure(0, weight=1)
        self._list_holder.columnconfigure(0, weight=1)

        self._steps_host = ctk.CTkScrollableFrame(
            self._list_holder,
            fg_color=C["bg"],
            corner_radius=0,
            scrollbar_fg_color=C["surface"],
            scrollbar_button_color=C["gold_dim"],
            scrollbar_button_hover_color=C["gold"],
        )
        self._steps_host.grid(row=0, column=0, sticky="nsew")
        self._steps_host.columnconfigure(0, weight=1)
        self._steps_host.bind("<MouseWheel>", self._on_step_wheel)
        self._steps_host._parent_canvas.bind("<MouseWheel>", self._on_step_wheel)

        self._build_cta(body)
        self._rebuild_step_cards()

    def _build_header(self, parent):
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=40, pady=(22, 6))
        header.columnconfigure(0, weight=1)
        self._header = header

        nav = ctk.CTkFrame(header, fg_color="transparent")
        nav.grid(row=0, column=0, sticky="ew")
        nav.columnconfigure(0, weight=1)

        if self.on_back:
            ctk.CTkButton(
                nav,
                text="Traditions",
                width=120,
                height=34,
                corner_radius=4,
                fg_color=C["surface"],
                hover_color=C["elevated"],
                text_color=C["ivory"],
                border_width=1,
                border_color=C["divider"],
                font=font_ui(13),
                command=self.on_back,
            ).grid(row=0, column=0, sticky="w")

        if self.on_history:
            ctk.CTkButton(
                nav,
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
            ).grid(row=0, column=1, sticky="e")

        style = config.get_style(self._style_id)

        self._brand_lbl = ctk.CTkLabel(
            header,
            text=style["short_title"],
            text_color=C["gold"],
            font=font_display(36, "bold"),
            anchor="w",
        )
        self._brand_lbl.grid(row=1, column=0, sticky="w", pady=(16, 0))

        self._heading_lbl = ctk.CTkLabel(
            header,
            text=style.get("menu_heading", f"{style['short_title']} step coaching"),
            text_color=C["ivory"],
            font=font_ui(15),
            anchor="w",
        )
        self._heading_lbl.grid(row=2, column=0, sticky="w", pady=(2, 0))

        self._meta_lbl = ctk.CTkLabel(
            header,
            text=style["subtitle"],
            text_color=C["muted"],
            font=font_ui(13),
            anchor="w",
        )
        self._meta_lbl.grid(row=3, column=0, sticky="w", pady=(2, 8))

    def _on_resize(self, event):
        if event.widget is not self:
            return
        inset = 48 if event.width < 900 else 72
        wrap = max(280, min(event.width - inset * 2, 760))
        for lbl in self._desc_labels:
            try:
                lbl.configure(wraplength=wrap)
            except Exception:
                pass
        side = max(28, (event.width - 980) // 2)
        self._apply_insets(side, event.width)

    def _rebuild_step_cards(self):
        if self._steps_host is None:
            return
        for child in self._steps_host.winfo_children():
            child.destroy()
        self._step_cards = {}
        self._desc_labels = []
        self._selected_step = None

        intro = ctk.CTkLabel(
            self._steps_host,
            text="Steps",
            text_color=C["ivory"],
            font=font_ui(14, "bold"),
            anchor="w",
        )
        intro.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 6))

        for i, step in enumerate(config.list_steps(self._style_id)):
            self._add_step_card(
                self._steps_host,
                row=i + 1,
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
            self._hint.configure(text="Select a step to continue")

    def configure_style(self, style_id: str):
        """Reload the step list for the chosen dance style."""
        self._style_id = style_id
        style = config.get_style(style_id)
        self.configure(fg_color=C["bg"])
        self._body.configure(fg_color=C["bg"])
        self._rule.configure(fg_color=C["gold"])
        self._steps_host.configure(
            fg_color=C["bg"],
            scrollbar_fg_color=C["surface"],
            scrollbar_button_color=C["gold_dim"],
            scrollbar_button_hover_color=C["gold"],
        )
        if hasattr(self, "_brand_lbl"):
            self._brand_lbl.configure(text=style["short_title"], text_color=C["gold"])
        if hasattr(self, "_heading_lbl"):
            self._heading_lbl.configure(
                text=style.get("menu_heading", f"{style['short_title']} step coaching"),
                text_color=C["ivory"],
            )
        if hasattr(self, "_meta_lbl"):
            self._meta_lbl.configure(text=style["subtitle"], text_color=C["muted"])
        self._rebuild_step_cards()
        self._fit_to_window()

    def _add_step_card(self, parent, row, step_id, title, description, tags, difficulty):
        frame = ctk.CTkFrame(
            parent,
            fg_color=C["card"],
            corner_radius=4,
            border_width=1,
            border_color=C["divider"],
            cursor="hand2",
        )
        frame.grid(row=row, column=0, sticky="ew", padx=8, pady=5)
        frame.columnconfigure(1, weight=1)

        mark = ctk.CTkFrame(frame, width=4, fg_color=C["divider"], corner_radius=0)
        mark.grid(row=0, column=0, rowspan=3, sticky="ns", padx=(0, 4))
        mark.grid_propagate(False)

        title_lbl = ctk.CTkLabel(
            frame,
            text=title,
            text_color=C["ivory"],
            font=font_ui(16, "bold"),
            anchor="w",
        )
        title_lbl.grid(row=0, column=1, sticky="ew", padx=(12, 18), pady=(12, 2))

        desc_lbl = ctk.CTkLabel(
            frame,
            text=description,
            text_color=C["muted"],
            font=font_ui(13),
            anchor="w",
            wraplength=640,
            justify="left",
        )
        desc_lbl.grid(row=1, column=1, sticky="new", padx=(12, 18), pady=(0, 4))
        self._desc_labels.append(desc_lbl)

        meta_row = ctk.CTkFrame(frame, fg_color="transparent")
        meta_row.grid(row=2, column=1, sticky="w", padx=(12, 18), pady=(0, 12))

        for tag in tags:
            ctk.CTkLabel(
                meta_row,
                text=f"  {tag}  ",
                text_color=C["gold"],
                fg_color=C["tip"],
                font=font_ui(11),
                corner_radius=3,
            ).pack(side="left", padx=(0, 6))

        if difficulty:
            ctk.CTkLabel(
                meta_row,
                text=difficulty,
                text_color=C["muted"],
                font=font_ui(12),
            ).pack(side="left", padx=(8, 0))

        def on_enter(_event):
            if self._selected_step != step_id:
                frame.configure(border_color=C["gold_dim"])

        def on_leave(_event):
            if self._selected_step != step_id:
                frame.configure(border_color=C["divider"])

        def on_click(_event):
            self._select_step(step_id, frame, mark)

        self._bind_card(frame, on_enter, on_leave, on_click)
        self._step_cards[step_id] = {"frame": frame, "mark": mark}

    def _bind_card(self, widget, on_enter, on_leave, on_click):
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
        widget.bind("<Button-1>", on_click)
        widget.bind("<MouseWheel>", self._on_step_wheel)
        for child in widget.winfo_children():
            self._bind_card(child, on_enter, on_leave, on_click)

    def _on_step_wheel(self, event):
        canvas = self._steps_host._parent_canvas
        if canvas.yview() == (0.0, 1.0):
            return "break"
        canvas.yview_scroll(-int(event.delta / 6), "units")
        return "break"

    def _equalize_cards(self):
        frames = [item["frame"] for item in self._step_cards.values()]
        if not frames:
            return
        for frame in frames:
            frame.grid_propagate(True)
        self.update_idletasks()
        target = max(frame.winfo_reqheight() for frame in frames)
        target = max(target, 112)
        inner_w = self._steps_host.winfo_width()
        card_w = max(280, inner_w - 28) if inner_w > 80 else 0
        for frame in frames:
            frame.rowconfigure(1, weight=1)
            if card_w:
                frame.configure(height=target, width=card_w)
            else:
                frame.configure(height=target)
            frame.grid_propagate(False)

    def _refresh_scroll(self):
        canvas = self._steps_host._parent_canvas
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all") or (0, 0, 0, 0))
        canvas.yview_moveto(0)

    def _select_step(self, step_id: str, frame: ctk.CTkFrame, mark: ctk.CTkFrame):
        for _sid, widgets in self._step_cards.items():
            widgets["frame"].configure(border_color=C["divider"], fg_color=C["card"], border_width=1)
            widgets["mark"].configure(fg_color=C["divider"])

        self._selected_step = step_id
        frame.configure(border_color=C["gold"], fg_color=C["step_active"], border_width=2)
        mark.configure(fg_color=C["gold"])

        if hasattr(self, "_cta_btn"):
            self._cta_btn.configure(
                state="normal",
                fg_color=C["gold"],
                hover_color=C["gold_hover"],
                text_color=C["ink"],
            )
            self._hint.configure(text="Opens the expert recording for this step")

    def _build_cta(self, parent):
        cta_frame = ctk.CTkFrame(parent, fg_color="transparent")
        cta_frame.grid(row=2, column=0, sticky="ew", padx=40, pady=(8, 22))
        cta_frame.columnconfigure(0, weight=1)
        self._cta_host = cta_frame

        self._cta_btn = ctk.CTkButton(
            cta_frame,
            text="Begin learning",
            font=font_ui(15, "bold"),
            height=46,
            corner_radius=4,
            fg_color=C["elevated"],
            hover_color=C["elevated"],
            text_color=C["muted"],
            state="disabled",
            command=self._on_begin_clicked,
        )
        self._cta_btn.grid(row=0, column=0, sticky="ew")

        self._hint = ctk.CTkLabel(
            cta_frame,
            text="Select a step to continue",
            text_color=C["muted"],
            font=font_ui(12),
            anchor="w",
        )
        self._hint.grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _on_begin_clicked(self):
        if self._selected_step:
            self.on_begin(self._selected_step)

    def on_show(self):
        self.after(40, self._fit_to_window)

    def _fit_to_window(self):
        try:
            width = self.winfo_width()
        except Exception:
            return
        if width < 200:
            return
        side = max(28, (width - 980) // 2)
        self._apply_insets(side, width)

    def _apply_insets(self, side: int, width: int):
        wrap = max(280, min(width - side * 2 - 64, 760))
        for lbl in self._desc_labels:
            try:
                lbl.configure(wraplength=wrap)
            except Exception:
                pass
        try:
            self._header.grid_configure(padx=side + 8)
            self._list_holder.grid_configure(padx=side)
            self._cta_host.grid_configure(padx=side + 8)
            self._brand_lbl.configure(wraplength=max(240, width - side * 2 - 40))
        except Exception:
            pass
        self._equalize_cards()
        self._refresh_scroll()

    def on_hide(self):
        pass
