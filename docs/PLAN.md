# TRIBE Analyzer — Implementation Plan

## Overview

A Gradio web app that analyzes short-form video (Instagram Reels / YouTube Shorts) using Meta's TRIBE v2 brain encoding model. It predicts fMRI brain responses, visualizes them, and sends the data + heatmap images to Claude Opus 4.6 (via OpenRouter) to generate a layman-friendly report on cognitive engagement — helping content creators optimize their videos.

## Target Environment

- RunPod with RTX 6000 Pro (48 GB VRAM)
- CUDA 12.x, Python 3.12+
- Gradio with `share=True` or exposed port

---

## Architecture

```
[Video Upload] → [Split into 20s segments] → [TRIBE v2 per segment]
                                                      ↓
                                              [Per-second predictions]
                                              [20,484 vertices/timestep]
                                                      ↓
                                          [Aggregate by brain region]
                                          [Schaefer 400 atlas → functional networks]
                                                      ↓
                              [Generate heatmaps (5s intervals) + GIFs + peak snapshots]
                                                      ↓
                                        [Build timestamped JSON summary]
                                                      ↓
                                [Send JSON + heatmap images to Claude Opus 4.6]
                                [via OpenRouter (multimodal)]
                                                      ↓
                                    [Layman-friendly report in Gradio]
                                    [+ downloadable HTML/PDF report]
```

---

## Step-by-Step Plan

### Step 1: Project Setup

**File:** `app.py` (main entry point)
**File:** `requirements.txt`

Dependencies:
- `tribev2[plotting]` (from GitHub)
- `gradio`
- `openai` (OpenRouter uses OpenAI-compatible API)
- `numpy`, `pandas`, `matplotlib`
- `imageio` (for GIF generation)

Model loading at startup:
```python
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
plotter = PlotBrain(mesh="fsaverage5")
```

### Step 2: Video Splitting

**File:** `video.py`

Split uploaded video into 20-second segments using ffmpeg (available on RunPod):
- Input: video file path, segment duration (default 20s)
- Output: list of segment file paths in a temp directory
- Use `ffmpeg -i input.mp4 -f segment -segment_time 20 -c copy segment_%03d.mp4`
- Track original timestamps for each segment (0:00-0:20, 0:20-0:40, 0:40-1:00)

Why split the video file (not just predictions):
- `get_events_dataframe(video_path=...)` handles audio extraction + transcription internally
- Running per-segment gives us clean event boundaries at 20s marks
- Avoids potential memory issues on longer videos
- Each segment gets its own events DataFrame and predictions

### Step 3: TRIBE v2 Processing

**File:** `brain.py`

For each 20s segment:
1. `df = model.get_events_dataframe(video_path=segment_path)`
2. `preds, segments = model.predict(events=df)`
3. Store: predictions array, segments list, events DataFrame

Output per segment:
- `preds`: shape (n_timesteps, 20484) — ~20 timesteps for 20s
- `segments`: list of segment objects with `.start`, `.duration`, `.events`

### Step 4: Brain Region Aggregation

**File:** `brain.py`

Convert raw vertex data into meaningful region-level activations:

1. Use PlotBrain with Schaefer 400 atlas: `PlotBrain(mesh="fsaverage5", atlas_name="schaefer_2018", atlas_dim=400)`
2. Map 20,484 vertices → 400 named regions → ~7 functional networks:
   - **Visual** (occipital cortex)
   - **Auditory** (superior temporal)
   - **Language** (Broca's, Wernicke's areas)
   - **Attention / Salience** (dorsal/ventral attention networks)
   - **Default Mode** (medial prefrontal, posterior cingulate — mind wandering, self-referential)
   - **Frontoparietal / Executive** (decision-making, working memory)
   - **Somatomotor** (motor cortex, sensory)

3. For each timestep, compute mean activation per network
4. Identify peak moments (top 3-5 timesteps with highest overall activation)
5. Identify engagement drops (lowest activation periods)

Output: structured dict with per-second, per-network activation values + peak/drop annotations.

### Step 5: Heatmap Image Generation

**File:** `visuals.py`

Generate three types of visualizations:

**A. 5-second interval heatmaps (12 images for 60s video):**
```python
# One brain surface plot every 5 seconds
for t in range(0, total_timesteps, 5):
    fig = plotter.plot_timesteps(
        preds[t:t+1], segments=segments[t:t+1],
        cmap="fire", norm_percentile=99,
        vmin=0.5, alpha_cmap=(0, 0.2), show_stimuli=True,
    )
```
- 4 images per 20s segment, 12 total for 60s
- Each image is a single-timestep brain plot — large, readable
- Saved as PNG at 150 DPI
- Displayed in Gradio gallery
- Sent to Claude as multimodal input
- Embedded in downloadable report

**B. Peak moment snapshots (3-5 images):**
- Single-timestep brain plots at highest activation moments
- Also at biggest second-to-second change and deepest drops
- These are the "money shots" for the report
- Also sent to Claude as multimodal input

**C. Animated brain GIF/video (1 per segment, 3 total):**
- Per-second brain surface plots rendered as frames
- Stitched into GIF or MP4 using matplotlib animation or imageio
- Shows brain activation evolving over time in real-time
- Displayed in Gradio only (NOT sent to Claude — too large, not supported)
- One animation per 20s segment

```python
import imageio
frames = []
for t in range(start, end):
    fig = plotter.plot_timesteps(
        preds[t:t+1], segments=segments[t:t+1],
        cmap="fire", norm_percentile=99,
        vmin=0.5, alpha_cmap=(0, 0.2), show_stimuli=True,
    )
    # Save frame to buffer, append to frames list
frames_to_gif(frames, f"segment_{i}_animation.gif", fps=2)
```
- fps=2 (one brain state every 0.5s real time) for easy viewing
- Alternative: fps=1 for real-time playback

All static images saved to temp directory for:
- Gradio display
- Multimodal input to Claude (5-sec intervals + peaks only)
- Embedding in downloadable report

### Step 6: Build JSON Summary

**File:** `analysis.py`

Structure:
```json
{
  "video_duration_seconds": 60,
  "segment_duration_seconds": 20,
  "segments": [
    {
      "segment_index": 0,
      "time_range": "0:00 - 0:20",
      "timesteps": [
        {
          "time": "0:01",
          "networks": {
            "visual": 0.82,
            "auditory": 0.45,
            "language": 0.31,
            "attention": 0.73,
            "default_mode": 0.12,
            "executive": 0.55,
            "somatomotor": 0.28
          },
          "dominant_network": "visual",
          "transcript_words": ["the", "quick", "brown"],
          "is_peak": false,
          "is_drop": false
        }
      ],
      "segment_summary": {
        "avg_activation_by_network": {},
        "peak_moments": [],
        "drop_moments": [],
        "dominant_network": "visual"
      }
    }
  ],
  "overall": {
    "global_peaks": [{"time": "0:12", "reason": "visual + attention spike"}],
    "global_drops": [{"time": "0:35", "reason": "low across all networks"}],
    "network_timeline": {}
  }
}
```

Save to temp file for reference + send to LLM.

### Step 7: OpenRouter LLM Report Generation

**File:** `report.py`

API call to OpenRouter (OpenAI-compatible):
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=user_provided_key,
)

response = client.chat.completions.create(
    model="anthropic/claude-opus-4-6",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": json_summary},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{seg1_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{seg2_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{seg3_b64}"}},
            # + 5-sec interval images + peak moment images (~15-17 images total)
        ]}
    ],
)
```

**System prompt** will instruct Claude to:
- Write for a content creator, not a neuroscientist
- Structure the report as:
  1. **Executive Summary** — one paragraph, overall cognitive impact
  2. **Timeline Analysis** — second-by-second breakdown grouped by 20s segments
     - What cognitive processes fire at each moment
     - Why (scene cuts, speech, music, motion)
  3. **Peak Engagement Moments** — the best moments and what made them work
  4. **Engagement Drops** — where attention/engagement falls and why
  5. **Brain Heatmap Interpretation** — reference the attached images
  6. **Actionable Recommendations** — specific suggestions for future content
- Use plain language (no jargon, explain any brain terms used)
- Reference specific timestamps (e.g., "at 0:12...")
- Tie findings to content creation strategies

### Step 8: Gradio UI

**File:** `app.py`

Layout:
```
┌──────────────────────────────────────────────────────┐
│  TRIBE Analyzer — Brain Response Video Analysis       │
├──────────────────────────────────────────────────────┤
│  OpenRouter API Key: [__________] (password box)      │
├──────────────────────────────────────────────────────┤
│  Upload Video: [drag & drop]                          │
│  [Analyze] button                                     │
├──────────────────────────────────────────────────────┤
│  Progress bar / status updates                        │
├──────────────────────────────────────────────────────┤
│  Brain Activity Animations:                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ Seg 1    │ │ Seg 2    │ │ Seg 3    │              │
│  │ GIF/MP4  │ │ GIF/MP4  │ │ GIF/MP4  │              │
│  │ 0:00-20  │ │ 0:20-40  │ │ 0:40-60  │              │
│  └──────────┘ └──────────┘ └──────────┘              │
├──────────────────────────────────────────────────────┤
│  5-Second Interval Heatmaps (gallery, 12 images):    │
│  [0:00] [0:05] [0:10] [0:15] [0:20] [0:25] ...      │
├──────────────────────────────────────────────────────┤
│  Peak Moments (gallery, 3-5 images):                 │
│  [Peak 1 + timestamp] [Peak 2] [Drop 1] ...         │
├──────────────────────────────────────────────────────┤
│  Report (markdown rendered):                          │
│  ┌────────────────────────────────────────────┐      │
│  │ Executive Summary...                       │      │
│  │ Timeline Analysis...                       │      │
│  │ Recommendations...                         │      │
│  └────────────────────────────────────────────┘      │
├──────────────────────────────────────────────────────┤
│  [Download Report as HTML] [Download JSON Data]       │
└──────────────────────────────────────────────────────┘
```

Gradio components:
- `gr.Textbox(type="password")` for API key
- `gr.Video()` for upload
- `gr.Video()` or `gr.Image()` x3 for animated brain GIFs (one per segment)
- `gr.Gallery()` for 5-second interval heatmaps (12 images)
- `gr.Gallery()` for peak/drop moment snapshots (3-5 images)
- `gr.Markdown()` for rendered report
- `gr.File()` for downloads (HTML report + raw JSON)
- Progress updates via `gr.Progress` or status textbox

### Step 9: Downloadable Report

**File:** `report.py`

Generate an HTML file that includes:
- The full LLM-generated report (markdown → HTML)
- Embedded heatmap images (base64 inline)
- Basic styling for readability
- Print-friendly CSS

---

## File Structure

```
tribev2-video-analyzer/
├── docs/
│   ├── PLAN.md               # This file
│   ├── RUNPOD_SETUP.md       # RunPod deployment guide
│   └── TROUBLESHOOTING.md    # Common issues and fixes
├── app.py                    # Gradio UI (4 tabs) + pipeline with checkpointing
├── video.py                  # Video/audio splitting, trimming, text-to-file (ffmpeg)
├── brain.py                  # TRIBE v2 processing (video/audio/text) + region aggregation
├── persistence.py            # Save/load results, checkpointing, dedup, history index
├── visuals.py                # Heatmaps, peak images, GIFs, MP4s
├── analysis.py               # JSON summary builder
├── report.py                 # OpenRouter LLM call + HTML/PDF/ZIP export
├── networks.py               # Brain network definitions (Schaefer → functional mapping)
├── setup.sh                  # One-command RunPod setup
├── results/                  # Persisted analysis outputs (gitignored)
└── requirements.txt          # Dependencies
```

---

## Completed Features (beyond original plan)

- **Multi-input support**: Script (text → gTTS), Voiceover (audio), and Video tabs
- **MP4 brain videos**: 30fps MP4s with original audio muxed in via ffmpeg
- **PDF reports**: Generated via weasyprint from the HTML report
- **ZIP packaging**: One-click download of all outputs (HTML, PDF, JSON, MP4s, GIFs, heatmaps)
- **Persistence & checkpointing**: All results saved to `./results/` with per-segment checkpoints. Crash-safe resume. Content-hash deduplication.
- **History tab**: Browse, reload, and delete past analyses
- **Environment auto-config**: `app.py` sets HF_HUB_ENABLE_HF_TRANSFER, PYVISTA_OFF_SCREEN, etc. at startup
- **Actionable report format**: "What to Change" section with Fix Weak Spots, Double Down on What Works, and General Virality Tips

---

## Open Risks / Considerations

1. **Schaefer atlas mapping**: Uses `PlotBrain(atlas_name="schaefer_2018", atlas_dim=400)` with a fallback approximate mapping if atlas labels aren't accessible.

2. **Segment boundary artifacts**: Splitting media at 20s marks means TRIBE processes each chunk independently — speech/context crossing boundaries may lose some accuracy. Acceptable tradeoff for cleaner analysis.

3. **OpenRouter costs**: Claude Opus 4.6 with multimodal input (JSON + ~15 images) costs ~$0.50-1.00 per analysis. Persistence ensures no duplicate charges on re-runs.

4. **Media duration limit**: Capped at 120s (allows upload up to 150s, trims to 120s).

5. **Pod persistence**: Results in `./results/` survive app restarts but NOT pod termination. Users should download ZIPs as backup.
