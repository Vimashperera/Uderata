"""
screen_report.py  —  Screen 4: Performance Report
Black & gold session summary. Consumes session_data from Screen 3.
"""
import io
import math
from datetime import datetime
from typing import Callable, Dict, List, Optional
import customtkinter as ctk
import tkinter as tk
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from core.angle_calculator import (
    JOINT_DISPLAY_NAMES, JOINT_ICONS, CORRECTIVE_INSTRUCTIONS, ALL_JOINT_NAMES,
)
from core import session_history as hist
from ui.theme import C, font_display, font_ui
import config


def _stars(acc):
    return hist.star_rating_from_form(acc)


def _fig_to_image(fig, w, h):
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=100, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    pil = Image.open(buf).convert("RGBA")
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=(w, h))


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=120, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    return buf.read()


def draw_star_rating(canvas: tk.Canvas, filled: int, total: int = 5,
                     x0: int = 8, y0: int = 8, size: float = 11.0,
                     gap: int = 26):
    """Canvas-drawn gold stars (filled / outline)."""
    filled = max(0, min(int(filled), total))

    def _star_points(cx, cy, r_outer, r_inner=None):
        if r_inner is None:
            r_inner = r_outer * 0.45
        pts = []
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            r = r_outer if i % 2 == 0 else r_inner
            pts.extend([cx + r * math.cos(ang), cy + r * math.sin(ang)])
        return pts

    for i in range(total):
        cx = x0 + size + i * gap
        cy = y0 + size
        pts = _star_points(cx, cy, size)
        if i < filled:
            canvas.create_polygon(
                pts, fill=C["gold"], outline=C["gold_bright"], width=1, smooth=False,
            )
        else:
            canvas.create_polygon(
                pts, fill="", outline=C["gold_dim"], width=2, smooth=False,
            )


class ReportScreen(ctk.CTkFrame):
    """Screen 4 — Session Performance Report."""

    def __init__(
        self, master, on_practice_again: Callable,
        on_watch_expert: Callable, on_menu: Callable, **kwargs,
    ):
        super().__init__(master, fg_color=C["bg"], **kwargs)
        self.on_practice_again = on_practice_again
        self.on_watch_expert = on_watch_expert
        self.on_menu = on_menu
        self._session_data: Dict = {}
        self._img1 = self._img2 = None
        self._bar_png: Optional[bytes] = None
        self._line_png: Optional[bytes] = None

    def load_report(self, session_data: Dict):
        self._session_data = session_data
        for w in self.winfo_children():
            w.destroy()
        self._bar_png = self._line_png = None
        self._build_ui()

    def _compute(self):
        sd = self._session_data
        summary = hist.summarize_session(sd)
        joint_acc = summary["joint_acc"]
        joint_dev = summary["joint_dev"]

        sorted_joints = sorted(joint_acc.items(), key=lambda x: x[1])
        top_errors = []
        for jname, acc in sorted_joints[:3]:
            top_errors.append({
                "joint": jname,
                "display_name": JOINT_DISPLAY_NAMES.get(jname, jname),
                "icon": JOINT_ICONS.get(jname, "·"),
                "accuracy": acc,
                "avg_deviation_deg": joint_dev[jname],
                "instruction": CORRECTIVE_INSTRUCTIONS.get(jname, ""),
            })

        accs = sd.get("frame_accuracies", [])
        form_accs = sd.get("form_accuracies") or accs
        timing_accs = sd.get("timing_accuracies") or []

        style_id = sd.get("style_id") or ""
        style_title = ""
        if style_id:
            try:
                style_title = config.get_style(style_id)["title"]
            except KeyError:
                style_title = style_id

        prior = sd.get("prior_session")  # set by app before record, or None
        form_cmp = hist.format_comparison_line(summary["form"], prior)
        timing_cmp = hist.format_timing_comparison_line(summary["timing"], prior)

        ts = sd.get("timestamp")
        if not ts:
            ts = datetime.now().astimezone().isoformat(timespec="seconds")

        return {
            "overall": summary["overall"],
            "form": summary["form"],
            "timing": summary["timing"],
            "avg_lag_ms": summary["avg_lag_ms"],
            "stars": summary["stars"],
            "joint_acc": joint_acc,
            "top_errors": top_errors,
            "history": form_accs if form_accs else accs,
            "timing_history": timing_accs,
            "duration": summary["duration"],
            "ended_by": summary["ended_by"],
            "step_name": summary["step_name"],
            "step_id": summary["step_id"],
            "style_id": style_id,
            "style_title": style_title,
            "timestamp": ts,
            "form_comparison": form_cmp,
            "timing_comparison": timing_cmp,
            "is_first_session": prior is None and bool(summary.get("step_id")),
        }

    def _build_ui(self):
        r = self._compute()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_header(r)
        self._build_body(r)
        self._build_actions()

    def _fmt_ts(self, ts: str) -> str:
        try:
            return datetime.fromisoformat(ts).strftime("%Y-%m-%d  %H:%M")
        except Exception:
            return str(ts)

    def _build_header(self, r):
        wrap = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0)
        wrap.grid(row=0, column=0, sticky="ew")
        wrap.columnconfigure(0, weight=1)

        ctk.CTkFrame(wrap, fg_color=C["gold"], height=3, corner_radius=0).grid(
            row=0, column=0, sticky="ew"
        )

        hdr = ctk.CTkFrame(wrap, fg_color=C["surface"], corner_radius=0)
        hdr.grid(row=1, column=0, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        left = ctk.CTkFrame(hdr, fg_color="transparent")
        left.grid(row=0, column=0, padx=22, pady=12, sticky="w")

        ctk.CTkLabel(
            left, text=f"Session Complete — {r['step_name']}",
            text_color=C["gold"], font=font_display(18, "bold"),
        ).pack(anchor="w")

        style_bit = r.get("style_title") or r.get("style_id") or ""
        meta = f"{style_bit}  ·  {self._fmt_ts(r['timestamp'])}" if style_bit else self._fmt_ts(r["timestamp"])
        ctk.CTkLabel(
            left, text=meta, text_color=C["muted"], font=font_ui(10),
        ).pack(anchor="w", pady=(2, 0))

        star_row = ctk.CTkFrame(left, fg_color="transparent")
        star_row.pack(anchor="w", pady=(4, 0))
        star_canvas = tk.Canvas(
            star_row, width=140, height=28, bg=C["surface"], highlightthickness=0,
        )
        star_canvas.pack(side="left")
        draw_star_rating(star_canvas, r["stars"], total=5, x0=4, y0=2, size=10, gap=24)

        dur = r["duration"]
        m, s = int(dur) // 60, int(dur) % 60
        ctk.CTkLabel(
            left, text=f"Duration: {m} min {s} sec",
            text_color=C["muted"], font=font_ui(10),
        ).pack(anchor="w", pady=(2, 0))

        ctk.CTkLabel(
            left,
            text=(
                f"Form {r['form']:.0f}%  ·  Timing {r['timing']:.0f}%"
                + (f"  ·  avg lag {r['avg_lag_ms']:.0f} ms" if r.get("avg_lag_ms", 0) > 1 else "")
            ),
            text_color=C["ivory"], font=font_ui(11, "bold"),
        ).pack(anchor="w", pady=(4, 0))

        if r.get("form_comparison"):
            ctk.CTkLabel(
                left, text=r["form_comparison"],
                text_color=C["gold"], font=font_ui(10),
            ).pack(anchor="w", pady=(2, 0))
        if r.get("timing_comparison"):
            ctk.CTkLabel(
                left, text=r["timing_comparison"],
                text_color=C["muted"], font=font_ui(10),
            ).pack(anchor="w")
        elif r.get("is_first_session") and not r.get("form_comparison"):
            ctk.CTkLabel(
                left,
                text="First recorded session for this step",
                text_color=C["muted"], font=font_ui(9, "italic"),
            ).pack(anchor="w", pady=(2, 0))

        if r["ended_by"] == "user":
            ctk.CTkLabel(
                left,
                text="Session ended early — scores reflect performance until you stopped.",
                text_color=C["close"], font=font_ui(9, "italic"),
            ).pack(anchor="w")

        gf = ctk.CTkFrame(hdr, fg_color="transparent")
        gf.grid(row=0, column=2, padx=24, pady=8)

        gc = tk.Canvas(gf, width=110, height=110, bg=C["surface"], highlightthickness=0)
        gc.pack()
        self._draw_mini_gauge(gc, r["form"])
        ctk.CTkLabel(
            gf, text=f"{r['form']:.0f}%  Form",
            text_color=C["ivory"], font=font_ui(10, "bold"),
        ).pack()

    def _draw_mini_gauge(self, c, value):
        cx, cy, radius = 55, 55, 44
        c.create_arc(
            cx - radius, cy - radius, cx + radius, cy + radius,
            start=220, extent=-260, outline=C["divider"], width=9, style="arc",
        )
        if value > 0:
            ext = -260 * (value / 100)
            col = C["good"] if value >= 75 else (C["close"] if value >= 50 else C["poor"])
            c.create_arc(
                cx - radius, cy - radius, cx + radius, cy + radius,
                start=220, extent=ext, outline=col, width=9, style="arc",
            )
        c.create_text(
            cx, cy, text=f"{value:.0f}%",
            fill=C["ivory"], font=("Georgia", 13, "bold"),
        )

    def _build_body(self, r):
        scroll = ctk.CTkScrollableFrame(self, fg_color=C["bg"], corner_radius=0)
        scroll.grid(row=1, column=0, sticky="nsew")
        scroll.columnconfigure(0, weight=1)
        scroll.columnconfigure(1, weight=1)

        left = ctk.CTkFrame(scroll, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(14, 6), pady=12)
        left.columnconfigure(0, weight=1)

        right = ctk.CTkFrame(scroll, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 14), pady=12)
        right.columnconfigure(0, weight=1)

        self._build_bar_chart(left, r)
        self._build_top_errors(left, r)
        self._build_line_chart(right, r)

    def _make_bar_fig(self, r):
        ja = r["joint_acc"]
        names = [JOINT_DISPLAY_NAMES.get(n, n) for n in ALL_JOINT_NAMES]
        values = [ja.get(n, 0.0) for n in ALL_JOINT_NAMES]
        order = np.argsort(values)
        values = [values[i] for i in order]
        names = [names[i] for i in order]
        colors = [
            C["good"] if v >= 85 else (C["close"] if v >= 65 else C["poor"])
            for v in values
        ]
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        fig.patch.set_facecolor(C["surface"])
        ax.set_facecolor(C["surface"])
        bars = ax.barh(names, values, color=colors, height=0.6)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Accuracy %", color=C["muted"], fontsize=8)
        ax.tick_params(colors=C["ivory"], labelsize=7)
        ax.spines[:].set_color(C["divider"])
        for bar, val in zip(bars, values):
            ax.text(
                min(val + 1, 101), bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", color=C["ivory"], fontsize=7,
            )
        fig.tight_layout(pad=0.8)
        return fig

    def _make_line_fig(self, r):
        history = r["history"]
        x = np.linspace(0, len(history) / 30, len(history))
        y = np.array(history, dtype=float)
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        fig.patch.set_facecolor(C["surface"])
        ax.set_facecolor(C["bg"])
        ax.plot(x, y, color=C["gold"], linewidth=1.6, alpha=0.95)
        ax.fill_between(x, y, alpha=0.18, color=C["gold"])
        best_i = int(np.argmax(y))
        worst_i = int(np.argmin(y))
        ax.scatter([x[best_i]], [y[best_i]], color=C["good"], s=50, zorder=5)
        ax.scatter([x[worst_i]], [y[worst_i]], color=C["poor"], s=50, zorder=5)
        ax.axhline(75, color=C["good"], linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axhline(50, color=C["close"], linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Time (s)", color=C["muted"], fontsize=8)
        ax.set_ylabel("Accuracy %", color=C["muted"], fontsize=8)
        ax.tick_params(colors=C["ivory"], labelsize=7)
        ax.spines[:].set_color(C["divider"])
        legend = [
            mpatches.Patch(color=C["good"], label="Best"),
            mpatches.Patch(color=C["poor"], label="Worst"),
        ]
        ax.legend(
            handles=legend, facecolor=C["surface"],
            labelcolor=C["ivory"], fontsize=7,
        )
        fig.tight_layout(pad=0.8)
        return fig

    def _build_bar_chart(self, parent, r):
        ctk.CTkLabel(
            parent, text="Joint Accuracy Breakdown",
            text_color=C["gold"], font=font_ui(12, "bold"), anchor="w",
        ).pack(fill="x", pady=(0, 6))

        if not r["joint_acc"]:
            ctk.CTkLabel(parent, text="No data.", text_color=C["muted"]).pack()
            return

        fig = self._make_bar_fig(r)
        self._bar_png = _fig_to_png_bytes(fig)
        img = _fig_to_image(fig, 440, 300)
        plt.close(fig)
        ctk.CTkLabel(parent, image=img, text="").pack()
        self._img1 = img

    def _build_top_errors(self, parent, r):
        ctk.CTkLabel(
            parent, text="Top 3 Corrections Needed",
            text_color=C["gold"], font=font_ui(12, "bold"), anchor="w",
        ).pack(fill="x", pady=(14, 6))

        for err in r["top_errors"]:
            card = ctk.CTkFrame(
                parent, fg_color=C["card"], corner_radius=10,
                border_width=1, border_color=C["border"],
            )
            card.pack(fill="x", pady=3)
            card.columnconfigure(1, weight=1)

            ctk.CTkLabel(
                card, text=err["icon"], font=font_ui(22), width=46,
            ).grid(row=0, column=0, rowspan=3, padx=8, pady=8)

            ctk.CTkLabel(
                card, text=err["display_name"],
                text_color=C["ivory"], font=font_ui(11, "bold"),
                anchor="w",
            ).grid(row=0, column=1, sticky="w", padx=4, pady=(8, 0))

            ctk.CTkLabel(
                card,
                text=(
                    f"Avg deviation: {err['avg_deviation_deg']:.1f}°  |  "
                    f"Accuracy: {err['accuracy']:.0f}%"
                ),
                text_color=C["muted"], font=font_ui(9), anchor="w",
            ).grid(row=1, column=1, sticky="w", padx=4)

            ctk.CTkLabel(
                card, text=err["instruction"],
                text_color=C["ivory"], font=font_ui(9),
                wraplength=340, justify="left", anchor="w",
            ).grid(row=2, column=1, sticky="w", padx=4, pady=(0, 8))

    def _build_line_chart(self, parent, r):
        ctk.CTkLabel(
            parent, text="Form Over Time",
            text_color=C["gold"], font=font_ui(12, "bold"), anchor="w",
        ).pack(fill="x", pady=(0, 6))

        history = r["history"]
        if len(history) < 2:
            ctk.CTkLabel(
                parent, text="Not enough data for timeline.",
                text_color=C["muted"],
            ).pack()
            return

        fig = self._make_line_fig(r)
        self._line_png = _fig_to_png_bytes(fig)
        img = _fig_to_image(fig, 460, 340)
        plt.close(fig)
        ctk.CTkLabel(parent, image=img, text="").pack()
        self._img2 = img

    def _build_actions(self):
        bar = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")

        ctk.CTkFrame(bar, fg_color=C["gold_dim"], height=1, corner_radius=0).pack(
            fill="x"
        )

        row = ctk.CTkFrame(bar, fg_color=C["surface"], corner_radius=0)
        row.pack(fill="x")

        for text, fg, cmd in [
            ("↩  Practice Again", C["gold"], self.on_practice_again),
            ("Watch Expert Again", C["elevated"], self.on_watch_expert),
            ("Return to Menu", C["elevated"], self.on_menu),
        ]:
            kwargs = {
                "text": text, "width": 185, "height": 40,
                "font": font_ui(11, "bold"), "fg_color": fg,
                "hover_color": C["gold_hover"], "corner_radius": 8, "command": cmd,
            }
            if fg == C["gold"]:
                kwargs["text_color"] = C["ink"]
            else:
                kwargs["text_color"] = C["ivory"]
                kwargs["border_width"] = 1
                kwargs["border_color"] = C["divider"]
            ctk.CTkButton(row, **kwargs).pack(side="left", padx=8, pady=12)

        ctk.CTkButton(
            row, text="Export PDF", width=145, height=40,
            font=font_ui(11, "bold"), fg_color=C["card"],
            hover_color=C["gold_deep"], text_color=C["muted"],
            border_width=1, border_color=C["divider"],
            corner_radius=8, command=self._export_pdf,
        ).pack(side="right", padx=14, pady=12)

    def _export_pdf(self):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.utils import ImageReader
            import tkinter.filedialog as fd
            r = self._compute()
            step_slug = str(r.get("step_name", "session")).lower().replace(" ", "_")
            path = fd.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf")],
                initialfile=f"{step_slug}_report.pdf",
            )
            if not path:
                return

            # Rebuild chart PNGs if missing (e.g. sparse data skipped a chart)
            bar_png = self._bar_png
            line_png = self._line_png
            if bar_png is None and r.get("joint_acc"):
                fig = self._make_bar_fig(r)
                bar_png = _fig_to_png_bytes(fig)
                plt.close(fig)
            if line_png is None and len(r.get("history") or []) >= 2:
                fig = self._make_line_fig(r)
                line_png = _fig_to_png_bytes(fig)
                plt.close(fig)

            c = rl_canvas.Canvas(path, pagesize=A4)
            pw, ph = A4
            y = ph - 40

            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, f"{config.APP_NAME}")
            y -= 18
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y, f"{r['step_name']}")
            y -= 16
            c.setFont("Helvetica", 10)
            style_bit = r.get("style_title") or r.get("style_id") or ""
            c.drawString(
                40, y,
                f"{style_bit}  |  {self._fmt_ts(r['timestamp'])}",
            )
            y -= 16
            dur = r["duration"]
            m, s = int(dur) // 60, int(dur) % 60
            c.drawString(
                40, y,
                f"Duration: {m}m {s}s  |  Form: {r['form']:.1f}%  |  "
                f"Timing: {r['timing']:.1f}%  |  Stars: {r['stars']}/5  |  "
                f"Avg lag: {r.get('avg_lag_ms', 0):.0f} ms",
            )
            y -= 14
            if r.get("form_comparison"):
                c.setFont("Helvetica", 10)
                c.drawString(40, y, r["form_comparison"])
                y -= 12
            if r.get("timing_comparison"):
                c.drawString(40, y, r["timing_comparison"])
                y -= 12
            elif r.get("is_first_session"):
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(40, y, "First recorded session for this step")
                y -= 12
            if r["ended_by"] == "user":
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(40, y, "Session ended early.")
                y -= 14

            y -= 6
            if bar_png:
                c.setFont("Helvetica-Bold", 11)
                c.drawString(40, y, "Joint Accuracy")
                y -= 8
                img = ImageReader(io.BytesIO(bar_png))
                img_w, img_h = 500, 280
                y -= img_h
                if y < 60:
                    c.showPage()
                    y = ph - 40 - img_h
                c.drawImage(img, 50, y, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")
                y -= 16

            if line_png:
                if y < 320:
                    c.showPage()
                    y = ph - 40
                c.setFont("Helvetica-Bold", 11)
                c.drawString(40, y, "Form Over Time")
                y -= 8
                img = ImageReader(io.BytesIO(line_png))
                img_w, img_h = 500, 300
                y -= img_h
                c.drawImage(img, 50, y, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")
                y -= 18

            if y < 120:
                c.showPage()
                y = ph - 40
            c.setFont("Helvetica-Bold", 11)
            c.drawString(40, y, "Top Corrections:")
            y -= 16
            c.setFont("Helvetica", 9)
            for err in r["top_errors"]:
                c.drawString(
                    50, y,
                    f"• {err['display_name']}: {err['avg_deviation_deg']}deg avg deviation "
                    f"({err['accuracy']:.0f}% accuracy)",
                )
                y -= 12
                c.drawString(60, y, f"  {err['instruction'][:110]}")
                y -= 16
                if y < 50:
                    c.showPage()
                    y = ph - 40

            c.save()
            import tkinter.messagebox as mb
            mb.showinfo("Exported", f"PDF saved to:\n{path}")
        except ImportError:
            import tkinter.messagebox as mb
            mb.showwarning("Missing package", "Install reportlab:\n  pip install reportlab")
        except Exception as e:
            import tkinter.messagebox as mb
            mb.showerror("Export Failed", str(e))
