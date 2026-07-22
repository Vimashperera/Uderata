"""
app.py — Screen manager for Udarata Pa Saramba 01 (1280×720).
"""
import os
import customtkinter as ctk
import tkinter.messagebox as mb

import config
from ui.theme import C, apply_app_chrome
from ui.screen_menu import MenuScreen
from ui.screen_preview import PreviewScreen
from ui.screen_practice import PracticeScreen
from ui.screen_report import ReportScreen


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        apply_app_chrome(self)

        self.title("Udarata Dance — Pa Saramba 01")
        self.geometry("1280x720")
        self.resizable(False, False)

        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"1280x720+{(sw-1280)//2}+{(sh-720)//2}")

        self.configure(fg_color=C["bg"])

        self._container = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        self._container.pack(fill="both", expand=True)
        self._container.rowconfigure(0, weight=1)
        self._container.columnconfigure(0, weight=1)

        self._screens: dict = {}
        self._current = None

        self._init_screens()
        self.show_screen("menu")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_screens(self):
        menu = MenuScreen(self._container, on_begin=self._on_step_selected)
        menu.grid(row=0, column=0, sticky="nsew")
        self._screens["menu"] = menu

        preview = PreviewScreen(
            self._container,
            video_path=config.VIDEO_PATH,
            on_start_practice=self._on_ready_start_practice,
            on_back=lambda: self.show_screen("menu"),
        )
        preview.grid(row=0, column=0, sticky="nsew")
        self._screens["preview"] = preview

        practice = PracticeScreen(
            self._container,
            video_path=config.VIDEO_PATH,
            json_path=config.JSON_PATH,
            step_title=config.STEP_TITLE,
            on_session_end=self._on_session_end,
            on_back=lambda: self.show_screen("preview"),
        )
        practice.grid(row=0, column=0, sticky="nsew")
        self._screens["practice"] = practice

        report = ReportScreen(
            self._container,
            on_practice_again=self._on_practice_again,
            on_watch_expert=lambda: self.show_screen("preview"),
            on_menu=lambda: self.show_screen("menu"),
        )
        report.grid(row=0, column=0, sticky="nsew")
        self._screens["report"] = report

    def show_screen(self, name: str):
        if name not in self._screens:
            return
        if self._current and hasattr(self._current, "on_hide"):
            self._current.on_hide()
        screen = self._screens[name]
        if hasattr(screen, "on_show"):
            screen.on_show()
        screen.tkraise()
        self._current = screen

    def _on_step_selected(self, step_id: str):
        assets = config.ensure_runtime_assets()

        if not assets["json_ok"]:
            mb.showerror(
                "Data Not Found",
                f"Fused expert data not found:\n{config.JSON_PATH}\n\n"
                "Run first:\n  python preprocess_multi_expert.py",
            )
            return

        if not assets["video_ok"]:
            mb.showerror(
                "Expert Video Missing",
                assets["message"]
                or (
                    f"Reference video not found under:\n{config.ASSETS_DIR}\n\n"
                    "Expected: Uderata pasaramba expert.mp4"
                ),
            )
            return

        # Keep screens pointed at the resolved clip (name may vary)
        video_path = assets.get("video_path") or config.VIDEO_PATH
        self._screens["preview"].video_path = video_path
        self._screens["practice"].video_path = video_path
        self._screens["practice"]._audio_wav_path = os.path.join(
            os.path.dirname(os.path.abspath(video_path)), "expert_display.wav"
        )

        if assets.get("placeholder"):
            mb.showwarning("Using Placeholder Video", assets["message"])

        self.show_screen("preview")

    def _on_ready_start_practice(self):
        """Preview CTA: open practice screen and start the session."""
        try:
            self.show_screen("practice")
            # Let on_show finish resetting UI, then start (same pattern as Practice Again)
            self.after(300, lambda: self._screens["practice"]._start_session())
        except Exception:
            import traceback
            mb.showerror(
                "Start Practice Error",
                f"Could not start practice:\n{traceback.format_exc()}",
            )

    def _on_session_end(self, session_data: dict):
        report_screen: ReportScreen = self._screens["report"]
        report_screen.load_report(session_data)
        self.show_screen("report")

    def _on_practice_again(self):
        try:
            self.show_screen("practice")
            # Use lambda to ensure delayed execution works safely
            self.after(700, lambda: self._screens["practice"]._start_session())
        except Exception as e:
            import traceback
            import tkinter.messagebox as mb
            mb.showerror("Practice Again Error", f"An error occurred:\n{traceback.format_exc()}")

    def _on_close(self):
        if self._current and hasattr(self._current, "on_hide"):
            self._current.on_hide()
        self.destroy()
