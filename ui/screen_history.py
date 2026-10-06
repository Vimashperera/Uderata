"""
screen_history.py — My Progress: local session trends per style/step.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

import customtkinter as ctk
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from core.angle_calculator import JOINT_DISPLAY_NAMES
from core import session_history as hist
from ui.theme import C, font_display, font_ui, palette
from ui.screen_report import _fig_to_image


class HistoryScreen(ctk.CTkFrame):
    """Browse past session trends for a selected style and step."""

    def __init__(self, master, on_back: Callable[[], None], **kwargs):
        super().__init__(master, fg_color=C["bg"], **kwargs)
        self.on_back = on_back
        self._style_id = config.STYLE_ORDER[0] if config.STYLE_ORDER else "udarata"
        self._step_id: Optional[str] = None
        self._chart_img = None
        self._style_var = ctk.StringVar(value="")
        self._step_var = ctk.StringVar(value="")
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._rule = ctk.CTkFrame(self, fg_color=C["gold"], height=3, corner_radius=0)
        self._rule.grid(row=0, column=0, sticky="ew")

        top = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0)
        top.grid(row=1, column=0, sticky="nsew")
        top.columnconfigure(0, weight=1)
        top.rowconfigure(2, weight=1)

        header = ctk.CTkFrame(top, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=22, pady=(16, 8))
        header.columnconfigure(1, weight=1)

        ctk.CTkButton(
            header,
            text="Back",
            width=90,
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

        self._title = ctk.CTkLabel(
            header,
            text="Progress",
            text_color=C["gold"],
            font=font_display(26, "bold"),
        )
        self._title.grid(row=0, column=1, sticky="w", padx=16)

        ctk.CTkLabel(
            header,
            text="Saved on this computer",
            text_color=C["muted"],
            font=font_ui(12),
        ).grid(row=0, column=2, sticky="e")

        filters = ctk.CTkFrame(top, fg_color=C["card"], corner_radius=10)
        filters.grid(row=1, column=0, sticky="ew", padx=22, pady=(4, 10))
        filters.columnconfigure(1, weight=1)
        filters.columnconfigure(3, weight=1)

        ctk.CTkLabel(
            filters, text="Style", text_color=C["muted"], font=font_ui(10),
        ).grid(row=0, column=0, padx=(14, 6), pady=12, sticky="w")

        style_names = [s["title"] for s in config.list_styles()]
        self._style_map = {s["title"]: s["id"] for s in config.list_styles()}
        self._style_menu = ctk.CTkOptionMenu(
            filters,
            values=style_names or ["—"],
            variable=self._style_var,
            command=self._on_style_changed,
            fg_color=C["surface"],
            button_color=C["gold"],
            button_hover_color=C["gold_hover"],
            text_color=C["ivory"],
            dropdown_fg_color=C["surface"],
            dropdown_text_color=C["ivory"],
            dropdown_hover_color=C["tip"],
            font=font_ui(13),
            width=200,
        )
        self._style_menu.grid(row=0, column=1, padx=6, pady=12, sticky="ew")

        ctk.CTkLabel(
            filters, text="Step", text_color=C["muted"], font=font_ui(10),
        ).grid(row=0, column=2, padx=(18, 6), pady=12, sticky="w")

        self._step_menu = ctk.CTkOptionMenu(
            filters,
            values=["—"],
            variable=self._step_var,
            command=self._on_step_changed,
            fg_color=C["surface"],
            button_color=C["gold"],
            button_hover_color=C["gold_hover"],
            text_color=C["ivory"],
            dropdown_fg_color=C["surface"],
            dropdown_text_color=C["ivory"],
            dropdown_hover_color=C["tip"],
            font=font_ui(13),
            width=280,
        )
        self._step_menu.grid(row=0, column=3, padx=(6, 14), pady=12, sticky="ew")

        self._body = ctk.CTkScrollableFrame(top, fg_color=C["bg"], corner_radius=0)
        self._body.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self._body.columnconfigure(0, weight=1)

        # Initial selection
        if style_names:
            self._style_var.set(style_names[0])
            self._on_style_changed(style_names[0])

    def on_show(self):
        # Refresh charts when returning after new sessions
        if self._step_id:
            self._render_step(self._step_id)

    def on_hide(self):
        pass

    def _paint_style(self, style_id: str):
        pal = palette(style_id)
        if hasattr(self, "_rule"):
            self._rule.configure(fg_color=pal["gold"])
        if hasattr(self, "_title"):
            self._title.configure(text_color=pal["gold"])
        for menu in (getattr(self, "_style_menu", None), getattr(self, "_step_menu", None)):
            if menu is None:
                continue
            menu.configure(
                button_color=pal["gold"],
                button_hover_color=pal["gold_hover"],
                dropdown_hover_color=pal["tip"],
            )

    def _on_style_changed(self, title: str):
        self._style_id = self._style_map.get(title, self._style_id)
        steps = config.list_steps(self._style_id)
        self._step_title_map = {s["title"]: s["id"] for s in steps}
        titles = [s["title"] for s in steps] or ["—"]
        self._step_menu.configure(values=titles)
        self._step_var.set(titles[0])
        self._paint_style(self._style_id)
        self._on_step_changed(titles[0])

    def _on_step_changed(self, title: str):
        if title == "—" or title not in getattr(self, "_step_title_map", {}):
            self._step_id = None
            self._show_empty("Select a step to view progress.")
            return
        self._step_id = self._step_title_map[title]
        self._render_step(self._step_id)

    def _clear_body(self):
        for w in self._body.winfo_children():
            w.destroy()
        self._chart_img = None

    def _show_empty(self, message: str):
        self._clear_body()
        box = ctk.CTkFrame(
            self._body, fg_color=C["card"], corner_radius=12,
            border_width=1, border_color=C["border"],
        )
        box.grid(row=0, column=0, sticky="ew", padx=8, pady=40)
        ctk.CTkLabel(
            box,
            text=message,
            text_color=C["muted"],
            font=font_ui(13),
            wraplength=900,
            justify="center",
        ).pack(padx=40, pady=48)

    def _render_step(self, step_id: str):
        self._clear_body()
        sessions = hist.list_sessions_for_step(step_id)
        if not sessions:
            self._show_empty(
                "No sessions recorded yet for this step — complete a practice "
                "session to start tracking your progress."
            )
            return

        stats = hist.step_aggregate_stats(step_id)
        flagged = hist.most_flagged_joints(step_id, limit=3)
        pal = palette(self._style_id)

        # Aggregate cards
        stats_row = ctk.CTkFrame(self._body, fg_color="transparent")
        stats_row.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        for i in range(4):
            stats_row.columnconfigure(i, weight=1)

        cards = [
            ("Sessions", str(stats["count"])),
            ("Best form", f"{stats['best_form']:.0f}%"),
            ("Latest form", f"{stats['latest_form']:.0f}%"),
            ("Latest timing", f"{stats['latest_timing']:.0f}%"),
        ]
        for i, (label, value) in enumerate(cards):
            card = ctk.CTkFrame(
                stats_row, fg_color=C["card"], corner_radius=10,
                border_width=1, border_color=C["border"],
            )
            card.grid(row=0, column=i, sticky="ew", padx=5, pady=4)
            ctk.CTkLabel(
                card, text=label, text_color=C["muted"], font=font_ui(10),
            ).pack(anchor="w", padx=14, pady=(10, 0))
            ctk.CTkLabel(
                card, text=value, text_color=pal["gold"], font=font_display(22, "bold"),
            ).pack(anchor="w", padx=14, pady=(2, 12))

        # Trend chart
        ctk.CTkLabel(
            self._body,
            text="Form and timing across sessions",
            text_color=pal["gold"],
            font=font_ui(12, "bold"),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(16, 4))

        forms = [float(s["form_score"]) for s in sessions]
        timings = [float(s["timing_score"]) for s in sessions]
        xs = np.arange(1, len(sessions) + 1)

        body_w = self._body.winfo_width()
        if body_w < 240:
            body_w = max(640, self.winfo_width() - 64)
        img_w = max(520, min(1080, body_w - 20))
        img_h = max(180, int(img_w * 0.32))
        fig, ax = plt.subplots(figsize=(img_w / 100, img_h / 100))
        fig.patch.set_facecolor(C["surface"])
        ax.set_facecolor(C["bg"])
        ax.plot(xs, forms, color=pal["gold"], marker="o", linewidth=1.8, label="Form %")
        ax.plot(
            xs, timings, color=C["good"], marker="s", linewidth=1.4,
            alpha=0.9, label="Timing %",
        )
        ax.set_ylim(0, 105)
        ax.set_xlabel("Session (oldest to newest)", color=C["muted"], fontsize=8)
        ax.set_ylabel("Score %", color=C["muted"], fontsize=8)
        ax.tick_params(colors=C["ivory"], labelsize=7)
        ax.spines[:].set_color(C["divider"])
        ax.legend(facecolor=C["surface"], labelcolor=C["ivory"], fontsize=8)
        fig.tight_layout(pad=0.8)
        img = _fig_to_image(fig, img_w, img_h)
        plt.close(fig)
        self._chart_img = img
        ctk.CTkLabel(self._body, image=img, text="").grid(
            row=2, column=0, sticky="ew", padx=8, pady=4
        )

        # Date range note (factual)
        first_ts = sessions[0]["timestamp"]
        last_ts = sessions[-1]["timestamp"]
        ctk.CTkLabel(
            self._body,
            text=f"Recorded from {self._fmt_ts(first_ts)} to {self._fmt_ts(last_ts)}",
            text_color=C["muted"],
            font=font_ui(9),
            anchor="w",
        ).grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 10))

        # Persistent problem joints
        ctk.CTkLabel(
            self._body,
            text="Most frequently flagged joints",
            text_color=pal["gold"],
            font=font_ui(12, "bold"),
            anchor="w",
        ).grid(row=4, column=0, sticky="ew", padx=14, pady=(8, 4))

        if not flagged:
            ctk.CTkLabel(
                self._body,
                text="No flagged joints stored for this step yet.",
                text_color=C["muted"],
                font=font_ui(10),
                anchor="w",
            ).grid(row=5, column=0, sticky="ew", padx=16, pady=4)
        else:
            joint_box = ctk.CTkFrame(self._body, fg_color="transparent")
            joint_box.grid(row=5, column=0, sticky="ew", padx=8, pady=4)
            for i, row in enumerate(flagged):
                name = JOINT_DISPLAY_NAMES.get(row["joint_name"], row["joint_name"])
                card = ctk.CTkFrame(
                    joint_box, fg_color=C["card"], corner_radius=10,
                    border_width=1, border_color=C["border"],
                )
                card.pack(fill="x", pady=3, padx=6)
                ctk.CTkLabel(
                    card,
                    text=(
                        f"{name}: flagged in {int(row['flag_count'])} "
                        f"session{'s' if int(row['flag_count']) != 1 else ''}, "
                        f"average accuracy {float(row['avg_accuracy']):.0f}%"
                    ),
                    text_color=C["ivory"],
                    font=font_ui(11),
                    anchor="w",
                ).pack(fill="x", padx=14, pady=12)

    @staticmethod
    def _fmt_ts(ts: str) -> str:
        try:
            dt = datetime.fromisoformat(ts)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(ts)
