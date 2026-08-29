"""
app.py — Screen manager for multi-style dance learner (1280×720).
"""
from datetime import datetime

import customtkinter as ctk
import tkinter.messagebox as mb

import config
from core import session_history as hist
from ui.theme import C, apply_app_chrome
from ui.screen_style import StyleScreen
from ui.screen_menu import MenuScreen
from ui.screen_preview import PreviewScreen
from ui.screen_practice import PracticeScreen
from ui.screen_report import ReportScreen
from ui.screen_history import HistoryScreen


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        apply_app_chrome(self)
        hist.init_db()

        self.title(config.APP_NAME)
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
        self._active_style_id = config.STYLE_ID
        self._active_step_id = config.STEP_ID
        self._history_return = "style"

        self._init_screens()
        self.show_screen("style")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_screens(self):
        style = StyleScreen(
            self._container,
            on_style_selected=self._on_style_selected,
            on_history=lambda: self._open_history("style"),
        )
        style.grid(row=0, column=0, sticky="nsew")
        self._screens["style"] = style

        menu = MenuScreen(
            self._container,
            on_begin=self._on_step_selected,
            on_back=lambda: self.show_screen("style"),
            on_history=lambda: self._open_history("menu"),
        )
        menu.grid(row=0, column=0, sticky="nsew")
        self._screens["menu"] = menu

        history = HistoryScreen(
            self._container,
            on_back=self._on_history_back,
        )
        history.grid(row=0, column=0, sticky="nsew")
        self._screens["history"] = history

        preview = PreviewScreen(
            self._container,
            video_path=config.VIDEO_PATH,
            on_start_practice=self._on_ready_start_practice,
            on_back=lambda: self.show_screen("menu"),
            step_title=config.STEP_TITLE,
            preview_blurb=config.get_step(config.STEP_ID).get("preview_blurb", ""),
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
            reference_loops=config.get_step(config.STEP_ID).get(
                "reference_loops", config.DEFAULT_REFERENCE_LOOPS
            ),
            step_id=config.STEP_ID,
            style_id=config.STYLE_ID,
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

    def _open_history(self, return_to: str):
        self._history_return = return_to if return_to in self._screens else "style"
        self.show_screen("history")

    def _on_history_back(self):
        self.show_screen(self._history_return)

    def _on_style_selected(self, style_id: str):
        try:
            config.set_active_style(style_id)
        except KeyError:
            mb.showerror("Unknown Style", f"Style id not recognised:\n{style_id}")
            return

        self._active_style_id = style_id
        self._active_step_id = config.STEP_ID
        style = config.get_style(style_id)
        self.title(f"{config.APP_NAME} — {style['title']}")
        self._screens["menu"].configure_style(style_id)
        self.show_screen("menu")

    def _on_step_selected(self, step_id: str):
        try:
            step = config.get_step(step_id)
        except KeyError:
            mb.showerror("Unknown Step", f"Step id not recognised:\n{step_id}")
            return

        assets = config.ensure_runtime_assets(step_id)
        self._active_step_id = step_id
        self._active_style_id = step["style_id"]
        style = config.get_style(step["style_id"])
        self.title(f"{config.APP_NAME} — {style['short_title']} — {step['title']}")

        if not assets["json_ok"]:
            mb.showerror(
                "Data Not Found",
                f"Fused expert data not found for {step['title']}:\n"
                f"{assets.get('json_path') or config.step_json_path(step_id)}\n\n"
                f"Run first:\n  python preprocess_multi_expert.py --step {step_id}",
            )
            return

        if not assets["video_ok"]:
            mb.showerror(
                "Expert Video Missing",
                assets["message"]
                or (
                    f"Reference video not found under:\n"
                    f"{config.step_assets_dir(step_id)}\n\n"
                    f"Expected: {step['video_candidates'][0]}"
                ),
            )
            return

        source_path = assets.get("video_path") or config.VIDEO_PATH
        video_path = assets.get("playback_path") or source_path
        json_path = assets.get("json_path") or config.step_json_path(step_id)
        loops = int(
            assets.get("reference_loops")
            or step.get("reference_loops", config.DEFAULT_REFERENCE_LOOPS)
        )
        audio_path = config.resolve_expert_audio_path(source_path)

        self._screens["preview"].configure_step(
            video_path=video_path,
            step_title=step["title"],
            preview_blurb=step.get("preview_blurb", ""),
            audio_path=audio_path,
        )
        self._screens["practice"].configure_step(
            video_path=video_path,
            json_path=json_path,
            step_title=step["title"],
            reference_loops=loops,
            audio_path=audio_path,
            step_id=step_id,
            style_id=step["style_id"],
        )

        if assets.get("placeholder"):
            mb.showwarning("Using Placeholder Video", assets["message"])

        self.show_screen("preview")

    def _on_ready_start_practice(self):
        """Preview CTA: open practice screen and start the session."""
        try:
            self.show_screen("practice")
            self.after(300, lambda: self._screens["practice"]._start_session())
        except Exception:
            import traceback
            mb.showerror(
                "Start Practice Error",
                f"Could not start practice:\n{traceback.format_exc()}",
            )

    def _on_session_end(self, session_data: dict):
        # Ensure style/step ids (practice should already set these)
        if not session_data.get("step_id"):
            session_data["step_id"] = self._active_step_id
        if not session_data.get("style_id"):
            session_data["style_id"] = self._active_style_id
        if not session_data.get("timestamp"):
            session_data["timestamp"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )

        # Prior session for comparison — read BEFORE writing this one
        prior = None
        step_id = session_data.get("step_id") or ""
        if step_id:
            prior = hist.get_previous_session(step_id)
        session_data["prior_session"] = prior

        # Persist after scoring complete (does not affect live practice loop)
        try:
            summary = hist.summarize_session(session_data)
            hist.record_session(session_data, summary=summary)
        except Exception as e:
            print(f"[history] Failed to record session: {e}")

        report_screen: ReportScreen = self._screens["report"]
        report_screen.load_report(session_data)
        self.show_screen("report")

    def _on_practice_again(self):
        try:
            self.show_screen("practice")
            self.after(700, lambda: self._screens["practice"]._start_session())
        except Exception as e:
            import traceback
            import tkinter.messagebox as mb
            mb.showerror("Practice Again Error", f"An error occurred:\n{traceback.format_exc()}")

    def _on_close(self):
        if self._current and hasattr(self._current, "on_hide"):
            self._current.on_hide()
        self.destroy()
