"""Generate a short PDF summary of Udarata technology and process."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, HRFlowable,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Udarata_Technology_Process_Summary.pdf")

GOLD = HexColor("#D4AF37")
INK = HexColor("#1A1A1A")
MUTED = HexColor("#555555")
BG_ROW = HexColor("#F7F1E3")


def main():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleGold",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=INK,
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=11,
        textColor=MUTED,
        spaceAfter=14,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=INK,
        spaceBefore=12,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=INK,
        spaceAfter=6,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=9,
        leading=12,
        textColor=MUTED,
    )
    mono = ParagraphStyle(
        "Mono",
        parent=body,
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        leftIndent=6,
        textColor=INK,
    )

    doc = SimpleDocTemplate(
        OUT,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Udarata Dance Coaching — Technology & Process Summary",
        author="Udarata Pa Saramba",
    )

    story = []
    story.append(Paragraph("Udarata Dance Coaching", title))
    story.append(Paragraph("Technology and Process Summary", subtitle))
    story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD, spaceAfter=10))

    story.append(Paragraph("1. What the system is", h1))
    story.append(Paragraph(
        "A Windows desktop coaching application for Kandyan (Udarata) dance. "
        "Learners select a step, watch an expert reference video, then practice with a "
        "webcam while the system scores <b>form</b> and <b>timing</b> against a fused "
        "expert pose timeline.",
        body,
    ))

    story.append(Paragraph("2. Technology stack", h1))
    tech_data = [
        [Paragraph("<b>Layer</b>", body), Paragraph("<b>Technology</b>", body)],
        ["Language / runtime", "Python 3"],
        ["User interface", "CustomTkinter (black & gold theme)"],
        ["Video / camera", "OpenCV"],
        ["Pose estimation", "MediaPipe Pose Landmarker (full model)"],
        ["Numerics", "NumPy"],
        ["Audio playback", "Pygame mixer + WAV (imageio-ffmpeg)"],
        ["Charts / PDF report", "Matplotlib, ReportLab"],
        ["Packaging", "PyInstaller (Udarata.exe onedir build)"],
    ]
    tech_table = Table(tech_data, colWidths=[45 * mm, 125 * mm])
    tech_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GOLD),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), BG_ROW),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BG_ROW, HexColor("#FFFFFF")]),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Core software modules", body))
    modules = [
        "<b>core/pose_extractor.py</b> — live and offline pose extraction",
        "<b>core/angle_calculator.py</b> — joint angles and deviations",
        "<b>core/motion_features.py</b> — bone directions, velocity, hybrid score",
        "<b>core/soft_dtw.py</b> — Soft-DTW form alignment and lag-based timing",
        "<b>core/expert_fusion.py</b> — multi-expert DTW align, median fuse, variance bands",
        "<b>core/dtw_engine.py</b> — expert JSON loader and matching",
        "<b>config.py</b> — step registry, paths, asset checks",
        "<b>preprocess_multi_expert.py</b> — offline expert rebuild pipeline",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(m, small), leftIndent=8, bulletColor=GOLD) for m in modules],
        bulletType="bullet",
        start="•",
    ))

    story.append(Paragraph("3. Dance steps currently supported", h1))
    story.append(Paragraph(
        "Namaskaraya; Pa Saramba 01 / 02 / 03; Goda Saramba 01 / 02 / 03; "
        "Goda Saramba Kasthirama 01 / 02 / 03.",
        body,
    ))
    story.append(Paragraph(
        "Each step has an expert video (and optional 720p proxy) in <b>assets/</b>, "
        "a beat WAV beside the video, a fused pose JSON in <b>data/</b>, and a "
        "<b>reference_loops</b> setting (for example Kasthirama = 2 loops, many other "
        "steps = 1, Pa Saramba 01 = 3).",
        body,
    ))

    story.append(Paragraph("4. Offline process — expert preparation", h1))
    story.append(Paragraph(
        "Before live coaching, expert performances are converted into a fused reference:",
        body,
    ))
    story.append(Paragraph(
        "Expert video(s)<br/>"
        "&nbsp;&nbsp;→ MediaPipe Pose (VIDEO mode)<br/>"
        "&nbsp;&nbsp;→ Joint angles + torso-frame bones<br/>"
        "&nbsp;&nbsp;→ DTW-align every expert onto the canonical teaching clip<br/>"
        "&nbsp;&nbsp;→ Median fuse + per-joint variance / adaptive tolerance scales<br/>"
        "&nbsp;&nbsp;→ Save <b>data/&lt;step&gt;.json</b><br/>"
        "&nbsp;&nbsp;→ Extract WAV audio for synchronized playback",
        mono,
    ))

    story.append(Paragraph("5. Live process — learner session", h1))
    live_steps = [
        "<b>Menu</b> — select a dance step",
        "<b>Preview</b> — watch expert video with audio",
        "<b>Practice</b> — countdown, then webcam + expert play together",
        "<b>Scoring (each frame)</b> — learner pose → angles / bones / velocity; "
        "Soft-DTW alignment for Form; timing/lag vs music–video clock; adaptive "
        "tolerances from expert variance",
        "<b>End</b> — after N reference loops (or user stop)",
        "<b>Report</b> — Form / Timing summary, joint breakdown, optional PDF export",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(s, small), leftIndent=8, bulletColor=GOLD) for s in live_steps],
        bulletType="bullet",
        start="•",
    ))
    story.append(Paragraph(
        "Hybrid form score (approximate blend): <b>~55% angles + ~35% bones + ~10% velocity</b>.",
        body,
    ))

    story.append(Paragraph("6. End-to-end flow", h1))
    story.append(Paragraph(
        "Expert videos → Preprocess / Fusion → Expert JSON + WAV<br/>"
        "Learner webcam ──────────────→ Live Practice UI ←── Expert JSON + WAV<br/>"
        "Live Practice UI → Form + Timing scores → Session Report",
        mono,
    ))

    story.append(Paragraph("7. Distribution for testing", h1))
    story.append(Paragraph(
        "A packaged Windows folder can be built with <b>build_exe.ps1</b> "
        "(PyInstaller). Testers receive <b>dist/Udarata/</b> containing "
        "<b>Udarata.exe</b> plus <b>assets/</b>, <b>data/</b>, and <b>models/</b>. "
        "Zip the whole folder and keep that structure when sharing.",
        body,
    ))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.8, color=GOLD, spaceAfter=6))
    story.append(Paragraph(
        "Udarata Pa Saramba — desktop coaching system summary for documentation and testing.",
        small,
    ))

    doc.build(story)
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
