"""
theme.py — Visual language for Sri Lankan Traditional Dance Coaching.

The room is a dark rehearsal studio. Each tradition brings one costume color.
Score colors stay green, amber, and red so their meaning does not change.
"""

# Surfaces and type that do not change between traditions.
BASE = {
    "bg": "#14171C",
    "surface": "#1C2028",
    "panel": "#181C24",
    "card": "#232833",
    "elevated": "#2C3340",
    "ivory": "#E6E2DA",
    "offwhite": "#E6E2DA",
    "muted": "#9AA1AD",
    "divider": "#323844",
    "stage": "#07080A",
    "stage_text": "#C5CCD6",
    "good": "#3CB87A",
    "close": "#E0A84A",
    "poor": "#E06A62",
}

# Quiet chrome used before a tradition is chosen.
NEUTRAL_ACCENT = {
    "gold": "#C5CBD6",
    "gold_bright": "#E6E2DA",
    "gold_dim": "#6E7684",
    "gold_deep": "#8E96A3",
    "gold_hover": "#D5DAE2",
    "ink": "#14171C",
    "border": "#3A4150",
    "tip": "#232833",
    "step_active": "#2A3140",
    "step_border": "#C5CBD6",
    "accent": "#C5CBD6",
    "accent_hover": "#D5DAE2",
    "burgundy": "#8E96A3",
}

# Costume color for each tradition, lifted so it reads on charcoal.
# Keys match the older screen code (gold = the tradition accent).
STYLE_THEMES = {
    # Kandyan ves: lacquer jacket, brass headdress.
    "udarata": {
        "gold": "#E07A84",
        "gold_bright": "#E0C48A",
        "gold_dim": "#8A4A52",
        "gold_deep": "#A84854",
        "gold_hover": "#EBA4AB",
        "ink": "#1A1012",
        "border": "#5A3036",
        "tip": "#2A1C20",
        "step_active": "#321F24",
        "step_border": "#E07A84",
        "accent": "#E07A84",
        "accent_hover": "#EBA4AB",
        "burgundy": "#A84854",
    },
    # Sabaragamuwa: white cloth, silver, river slate.
    "sabaragamuwa": {
        "gold": "#7EB4C6",
        "gold_bright": "#C5D5DE",
        "gold_dim": "#3E6270",
        "gold_deep": "#4E7A8C",
        "gold_hover": "#A5CEDB",
        "ink": "#0E181C",
        "border": "#2E4A56",
        "tip": "#1A2830",
        "step_active": "#1E3038",
        "step_border": "#7EB4C6",
        "accent": "#7EB4C6",
        "accent_hover": "#A5CEDB",
        "burgundy": "#4E7A8C",
    },
    # Low-country masks: vermilion and turmeric.
    "pahatharata": {
        "gold": "#E08A45",
        "gold_bright": "#E6C27A",
        "gold_dim": "#8A5530",
        "gold_deep": "#B5682E",
        "gold_hover": "#F0A66A",
        "ink": "#1C1008",
        "border": "#5C3A22",
        "tip": "#2A2018",
        "step_active": "#322418",
        "step_border": "#E08A45",
        "accent": "#E08A45",
        "accent_hover": "#F0A66A",
        "burgundy": "#B5682E",
    },
    # Folk cloth: indigo, with a leaf secondary.
    "jana_natum": {
        "gold": "#8B97D6",
        "gold_bright": "#9BB58A",
        "gold_dim": "#4A5278",
        "gold_deep": "#5C68A8",
        "gold_hover": "#AEB6E4",
        "ink": "#12141F",
        "border": "#343C5C",
        "tip": "#1C2030",
        "step_active": "#22263A",
        "step_border": "#8B97D6",
        "accent": "#8B97D6",
        "accent_hover": "#AEB6E4",
        "burgundy": "#5C68A8",
    },
}

FONT_DISPLAY = "Constantia"
FONT_UI = "Candara"


def palette(style_id: str | None = None) -> dict:
    """Full color dict for a tradition, or the neutral studio if none."""
    colors = dict(BASE)
    accent = STYLE_THEMES.get(style_id or "", NEUTRAL_ACCENT)
    colors.update(accent)
    return colors


# Shared dict. Screens import this object; activate_style() updates it in place.
C = palette(None)


def activate_style(style_id: str | None) -> dict:
    """Point the shared palette at one tradition. Widgets already drawn need restyle()."""
    C.update(palette(style_id))
    return C


def font_display(size: int, weight: str = "bold"):
    if weight == "italic":
        return (FONT_DISPLAY, size, "italic")
    if weight == "normal":
        return (FONT_DISPLAY, size)
    return (FONT_DISPLAY, size, "bold")


def font_ui(size: int, weight: str = "normal"):
    if weight == "bold":
        return (FONT_UI, size, "bold")
    if weight == "italic":
        return (FONT_UI, size, "italic")
    return (FONT_UI, size)


def apply_app_chrome(root) -> None:
    """Dark window chrome. Tradition color is applied later, per screen."""
    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root.configure(fg_color=C["bg"])


def fit_window(root) -> None:
    """Size the window to the display, and allow the learner to resize it."""
    root.update_idletasks()
    sw = max(root.winfo_screenwidth(), 800)
    sh = max(root.winfo_screenheight(), 600)
    w = max(800, sw - 48)
    h = max(560, sh - 96)
    root.minsize(min(960, w), min(620, h))
    root.resizable(True, True)
    root.geometry(f"{min(w, sw)}x{min(h, sh - 40)}+8+8")
    try:
        root.state("zoomed")
    except Exception:
        pass
