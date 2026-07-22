# Udarata Pa Saramba 01

Desktop coaching app for Kandyan (Udarata) **Pa Saramba 01** — live webcam pose comparison against a fused expert timeline.

## Features

- MediaPipe Pose (full model) with joint angles + bone directions
- Soft-DTW form alignment and timing / lag feedback
- Multi-expert canonical DTW fusion with adaptive tolerances
- Black & gold CustomTkinter UI (preview → practice → report)

## Setup

```powershell
cd "E:\SLIIT\Uderata system\UdarataPaSaramba"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the teaching clip at:

`assets/Udarata pasaramba expert.mp4`

MediaPipe `.task` models download automatically into `models/` on first run.

## Preprocess (rebuild expert JSON)

```powershell
$env:UDARATA_EXPERT_VIDEOS = "E:\SLIIT\FINAL RESEARCH\Expert data"
python preprocess_multi_expert.py
```

## Run

```powershell
python main.py
```

## Project layout

- `main.py` — entry point
- `ui/` — screens (menu, preview, practice, report)
- `core/` — pose, angles, Soft-DTW, expert fusion
- `data/pa_saramba_01.json` — fused expert timeline
- `preprocess_multi_expert.py` — Phase-4 expert rebuild
