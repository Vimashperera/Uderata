"""
screen_report.py  —  Screen 4: Performance Report
Black & gold session summary. Consumes session_data from Screen 3.
"""
import io
from typing import Callable, Dict, List
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from core.angle_calculator import (
    JOINT_DISPLAY_NAMES, JOINT_ICONS, CORRECTIVE_INSTRUCTIONS, ALL_JOINT_NAMES,
)
from ui.theme import C, font_display, font_ui


def _stars(acc):
    if acc >= 90:
        return 5
    if acc >= 75:
        return 4
    if acc >= 60:
        return 3
    if acc >= 45:
        return 2
    return 1


def _fig_to_image(fig, w, h):
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=100, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    pil = Image.open(buf).convert("RGBA")
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=(w, h))


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

    def load_report(self, session_data: Dict):
        self._session_data = session_data
        for w in self.winfo_children():
            w.destroy()
        self._build_ui()

    def _compute(self):
        sd = self._session_data
        accs = sd.get("frame_accuracies", [])
        form_accs = sd.get("form_accuracies") or accs
        timing_accs = sd.get("timing_accuracies") or []
        lag_hist = sd.get("lag_ms_history") or []

        overall = float(np.mean(accs)) if accs else 0.0
        form_overall = float(np.mean(form_accs)) if form_accs else overall
        timing_overall = float(np.mean(timing_accs)) if timing_accs else 100.0
        avg_lag = float(np.mean(np.abs(lag_hist))) if lag_hist else 0.0

        jh = sd.get("joint_histories", {})
        jd = sd.get("joint_deviations", {})
        joint_acc: Dict[str, float] = {}
        joint_dev: Dict[str, float] = {}
        for jname in ALL_JOINT_NAMES:
            vals = jh.get(jname, [])
            joint_acc[jname] = float(np.mean(vals)) if vals else 0.0
            dev_vals = jd.get(jname, [])
            if dev_vals:
                joint_dev[jname] = round(float(np.mean(dev_vals)), 1)
            elif vals:
                joint_dev[jname] = round((100 - joint_acc[jname]) / 100 * 15, 1)
            else:
                joint_dev[jname] = 0.0

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

        return {
            "overall": overall,
            "form": form_overall,
            "timing": timing_overall,
            "avg_lag_ms": avg_lag,
            "stars": _stars(form_overall),
            "joint_acc": joint_acc,
            "top_errors": top_errors,
            "history": form_accs if form_accs else accs,
            "timing_history": timing_accs,
            "duration": sd.get("duration_seconds", 0.0),
            "ended_by": sd.get("ended_by", "user"),
            "step_name": sd.get("step_name", "Pa Saramba 01"),
        }

    def _build_ui(self):
        r = self._compute()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_header(r)
        self._build_body(r)
        self._build_actions()

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
        left.grid(row=0, column=0, padx=22, pady=14, sticky="w")

        ctk.CTkLabel(
            left, text=f"Session Complete — {r['step_name']}",
            text_color=C["gold"], font=font_display(18, "bold"),
        ).pack(anchor="w")

        stars_text = "★" * r["stars"] + "☆" * (5 - r["stars"])
        ctk.CTkLabel(
            left, text=stars_text, text_color=C["gold"], font=font_ui(20),
        ).pack(anchor="w", pady=(2, 0))

        dur = r["duration"]
        m, s = int(dur) // 60, int(dur) % 60
        ctk.CTkLabel(
            left, text=f"You practiced for {m} min {s} sec",
            text_color=C["muted"], font=font_ui(10),
        ).pack(anchor="w")

        ctk.CTkLabel(
            left,
            text=(
                f"Form {r['form']:.0f}%  ·  Timing {r['timing']:.0f}%"
                + (f"  ·  avg lag {r['avg_lag_ms']:.0f} ms" if r.get("avg_lag_ms", 0) > 1 else "")
            ),
            text_color=C["ivory"], font=font_ui(11, "bold"),
        ).pack(anchor="w", pady=(4, 0))

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

    def _build_bar_chart(self, parent, r):
        ctk.CTkLabel(
            parent, text="Joint Accuracy Breakdown",
            text_color=C["gold"], font=font_ui(12, "bold"), anchor="w",
        ).pack(fill="x", pady=(0, 6))

        ja = r["joint_acc"]
        if not ja:
            ctk.CTkLabel(parent, text="No data.", text_color=C["muted"]).pack()
            return

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
            import tkinter.filedialog as fd
            path = fd.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf")],
                initialfile="pa_saramba_01_report.pdf",
            )
            if not path:
                return
            r = self._compute()
            c = rl_canvas.Canvas(path, pagesize=A4)
            pw, ph = A4
            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, ph - 50, "Udarata Dance — Pa Saramba 01")
            c.setFont("Helvetica", 12)
            c.drawString(40, ph - 72, f"Report: {r['step_name']}")
            dur = r["duration"]
            m, s = int(dur) // 60, int(dur) % 60
            c.drawString(
                40, ph - 92,
                f"Duration: {m}m {s}s  |  Form: {r['form']:.1f}%  |  Timing: {r['timing']:.1f}%",
            )
            c.drawString(
                40, ph - 112,
                f"Star Rating: {'*'*r['stars']}{'.'*(5-r['stars'])}  |  "
                f"Avg lag: {r.get('avg_lag_ms', 0):.0f} ms",
            )
            if r["ended_by"] == "user":
                c.setFont("Helvetica-Oblique", 10)
                c.drawString(40, ph - 132, "Session ended early.")
            c.setFont("Helvetica-Bold", 11)
            c.drawString(40, ph - 160, "Top Corrections:")
            c.setFont("Helvetica", 10)
            y = ph - 178
            for err in r["top_errors"]:
                c.drawString(
                    50, y,
                    f"• {err['display_name']}: {err['avg_deviation_deg']}deg avg deviation",
                )
                y -= 16
                c.drawString(60, y, f"  {err['instruction']}")
                y -= 20
            c.save()
            import tkinter.messagebox as mb
            mb.showinfo("Exported", f"PDF saved to:\n{path}")
        except ImportError:
            import tkinter.messagebox as mb
            mb.showwarning("Missing package", "Install reportlab:\n  pip install reportlab")
        except Exception as e:
            import tkinter.messagebox as mb
            mb.showerror("Export Failed", str(e))
