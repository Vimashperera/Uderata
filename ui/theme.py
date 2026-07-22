"""
theme.py — Shared black & gold visual language for Udarata Pa Saramba.
"""

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    # Surfaces (deep black → charcoal)
    "bg": "#050505",
    "surface": "#0C0C0C",
    "panel": "#101010",
    "card": "#171717",
    "elevated": "#1F1F1F",
    "tip": "#14110A",

    # Gold family
    "gold": "#D4AF37",
    "gold_bright": "#F0D060",
    "gold_dim": "#8B7340",
    "gold_deep": "#5C4A1E",
    "gold_hover": "#B8962E",

    # Type
    "ivory": "#F7F1E3",
    "offwhite": "#F7F1E3",
    "muted": "#A89F8C",
    "ink": "#0A0A0A",

    # Lines / chrome
    "divider": "#2A2A2A",
    "border": "#3D3420",

    # Semantic (kept for scoring UI)
    "good": "#3DDC97",
    "close": "#E8A838",
    "poor": "#E85D5D",

    # Compatibility aliases used by older screen code
    "burgundy": "#5C4A1E",
    "accent": "#D4AF37",
    "accent_hover": "#B8962E",
    "step_active": "#1A1608",
    "step_border": "#D4AF37",
}

# ── Typography (Windows-friendly expressive faces) ────────────────────────────
FONT_DISPLAY = "Georgia"
FONT_UI = "Bahnschrift"
FONT_UI_FALLBACK = "Segoe UI"


def font_display(size: int, weight: str = "bold"):
    return (FONT_DISPLAY, size, weight)


def font_ui(size: int, weight: str = "normal"):
    """Primary UI face; Tk falls back if Bahnschrift is missing on rare setups."""
    if weight == "bold":
        return (FONT_UI, size, "bold")
    if weight == "italic":
        return (FONT_UI, size, "italic")
    return (FONT_UI, size)


def apply_app_chrome(root) -> None:
    """Dark window chrome for the main CustomTkinter app."""
    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root.configure(fg_color=C["bg"])
