"""
app.py — Screen manager for multi-style dance learner (1280×720).
"""
import customtkinter as ctk
import tkinter.messagebox as mb

import config
from ui.theme import C, apply_app_chrome
from ui.screen_style import StyleScreen
from ui.screen_menu import MenuScreen
from ui.screen_preview import PreviewScreen
from ui.screen_practice import PracticeScreen
from ui.screen_report import ReportScreen


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        apply_app_chrome(self)

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

        self._init_screens()
        self.show_screen("style")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_screens(self):
        style = StyleScreen(self._container, on_style_selected=self._on_style_selected)
        style.grid(row=0, column=0, sticky="nsew")
        self._screens["style"] = style

        menu = MenuScreen(
            self._container,
            on_begin=self._on_step_selected,
            on_back=lambda: self.show_screen("style"),
        )
        menu.grid(row=0, column=0, sticky="nsew")
        self._screens["menu"] = menu

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
        # Prefer lightweight proxy for UI playback (4K sources are too heavy)
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
