# TRIBE Analyzer

A [Gradio](https://www.gradio.app/) web app that analyzes content using Meta’s [**TRIBE v2**](https://github.com/facebookresearch/tribev2) brain encoding model. It predicts cortical responses from **video**, **audio**, or **text scripts**, visualizes them on brain surfaces, aggregates activity into functional networks, and sends structured data plus heatmap images to **Claude** (via [OpenRouter](https://openrouter.ai/)) to produce a layperson-friendly report on cognitive engagement — helping content creators optimize their Instagram Reels and YouTube Shorts.

**Important:** Outputs are *model-based estimates*, not measurements from real brain scans.

---

## What it does

1. **Accepts three input types** via tabbed UI: **Script** (text), **Voiceover** (audio), or **Video**.
2. **Splits** media into **20-second** segments (via ffmpeg). Scripts are processed as a single segment via TRIBE’s internal gTTS conversion.
3. **Runs [TRIBE v2](https://github.com/facebookresearch/tribev2)** on each segment: event extraction, predictions (~20,484 vertices per timestep).
4. **Aggregates** vertex data through the **Schaefer 400** atlas into **seven Yeo-style networks** (visual, somatomotor, dorsal/ventral attention, limbic, frontoparietal, default mode) — see `networks.py`.
5. **Renders** 5-second interval heatmaps, peak/drop snapshots, per-segment **MP4** videos (with audio), and **GIF** animations.
6. **Builds** a timestamped **JSON** summary (`analysis.py`).
7. **Calls** OpenRouter (OpenAI-compatible API) with **Claude Opus 4.6** and multimodal content (JSON + images) to generate a markdown report with actionable recommendations.
8. **Packages** downloadable **HTML**, **PDF**, and **ZIP** reports.

**Persistence:** All analysis results are saved to `./results/` with per-segment checkpointing. If the app crashes mid-analysis, it resumes from the last checkpoint — no lost GPU compute or LLM tokens. Re-uploading the same file loads cached results instantly.

Maximum media length enforced in the UI: **120 seconds** (allows upload up to 2:30, trims to 2:00).

---

## Requirements

| Requirement | Notes |
|-------------|--------|
| **GPU** | Strongly recommended; [TRIBE v2](https://github.com/facebookresearch/tribev2) is GPU-heavy. The project was tested with **~48 GB VRAM** (e.g. RTX 6000 Pro on RunPod). |
| **Python** | **3.12+** recommended. |
| **CUDA** | CUDA 12.x typical for PyTorch stacks. |
| **ffmpeg** | Required for splitting video; often preinstalled on ML images. |
| **Hugging Face** | Access to [`facebook/tribev2`](https://huggingface.co/facebook/tribev2) (token + license acceptance if required). |
| **OpenRouter** | API key for Claude; user enters it in the UI (not stored in the repo). |

---

## Installation

From the `tribev2-video-analyzer` directory:

```bash
pip install -r requirements.txt
```

`requirements.txt` installs [`tribev2[plotting]`](https://github.com/facebookresearch/tribev2) from the official repo, Gradio, the OpenAI client (for OpenRouter), NumPy/Pandas/Matplotlib, and `imageio[ffmpeg]` for GIFs.

Log in to Hugging Face so weights can download:

```bash
huggingface-cli login
```

---

## Run the app

```bash
python app.py
```

By default the server listens on **0.0.0.0:7860** and Gradio **`share=True`** is enabled (public `gradio.live` link). Open the printed URL in a browser, paste your **OpenRouter** API key, and use one of three tabs:

- **Script** — paste text, analyze language-driven brain responses
- **Voiceover** — upload audio, analyze voice/tone/pacing effects
- **Video** — upload video, full visual + audio brain analysis
- **History** — browse past analyses, reload results instantly

---

## Project layout

| File | Role |
|------|------|
| `app.py` | Gradio UI (4 tabs: Script, Voiceover, Video, History), orchestrates the pipeline with checkpointing. |
| `brain.py` | [TRIBE v2](https://github.com/facebookresearch/tribev2) loading, per-segment prediction (video/audio/text), atlas aggregation, peaks/drops. |
| `persistence.py` | Save/load analysis results, per-segment checkpointing, content-hash deduplication, history index. |
| `networks.py` | Schaefer region labels → seven functional networks. |
| `video.py` | ffmpeg-based splitting (video + audio), trimming, text-to-file utilities. |
| `visuals.py` | Interval heatmaps, peak/drop images, segment GIFs and MP4s. |
| `analysis.py` | JSON summary construction and file output. |
| `report.py` | OpenRouter multimodal call, markdown report, HTML/PDF export, ZIP packaging. |
| `setup.sh` | One-command RunPod setup (tribev2, deps, PyTorch, HuggingFace login). |
| `docs/PLAN.md` | Design and implementation plan. |
| `docs/RUNPOD_SETUP.md` | Step-by-step RunPod GPU setup, ports, costs. |
| `docs/TROUBLESHOOTING.md` | Solutions for common errors (PyTorch/Blackwell, hf_transfer, ffmpeg, etc.). |

---

## Further documentation

- **[facebookresearch/tribev2](https://github.com/facebookresearch/tribev2)** — upstream TRIBE v2 code, paper link, and Colab demo.
- **[docs/RUNPOD_SETUP.md](docs/RUNPOD_SETUP.md)** — GPU pod creation, disk size, port **7860**, HF login, and costs.
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — Solutions for PyTorch + Blackwell GPU issues, hf_transfer, CUDA NVRTC, version conflicts, and more.
- **[docs/PLAN.md](docs/PLAN.md)** — Architecture, data flow, and UI design notes.

---

## License and third-party models

[**TRIBE v2**](https://github.com/facebookresearch/tribev2) is provided by Meta under its own terms; use of `facebook/tribev2` on Hugging Face is subject to that model’s license. This repository does not redistribute model weights.
