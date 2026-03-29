# RunPod Setup Guide — TRIBE Analyzer

## 1. Create a Pod

1. Go to [RunPod](https://www.runpod.io/) and create a new GPU Pod
2. **GPU**: Select **RTX 6000 Pro** (48 GB VRAM)
3. **Template**: Pick **RunPod PyTorch 2.x** (comes with CUDA 12.x, Python 3.12, ffmpeg)
4. **Disk**: At least **30 GB** container disk (model weights ~1 GB, plus cache and temp files)
5. **Expose ports**: Add **7860** under HTTP ports (for Gradio UI). Alternatively, use `share=True` for a public Gradio link (enabled by default).
6. Click **Deploy**

## 2. Connect to the Pod

Once the pod is running:
- Click **Connect** → **Start Web Terminal** (or use SSH if you prefer)

## 3. Clone and Install

```bash
# Clone the repo
git clone https://github.com/sarfarazh/tribev2-video-analyzer.git
cd tribev2-video-analyzer

# Run the setup script (handles everything)
bash setup.sh
```

The setup script:
1. Installs `tribev2` without letting it downgrade PyTorch
2. Installs all other dependencies (Gradio, OpenAI, imageio, etc.)
3. Ensures PyTorch 2.8.0+cu128 with Blackwell (sm_120) support
4. Installs OSMesa for headless brain surface rendering
5. Verifies everything works

**Why not just `pip install -r requirements.txt`?** Because tribev2 pins `torch<2.7`, which downgrades PyTorch and breaks Blackwell GPU support. The setup script avoids this.

## 4. Launch the App

The setup script already handles HuggingFace login (prompts for your token during install) and persists all required environment variables to `~/.bashrc`.

**Note**: Check if `facebook/tribev2` requires you to accept a license agreement on the model page first: https://huggingface.co/facebook/tribev2

```bash
python app.py
```

On first launch:
- The TRIBE v2 model downloads (~1 GB) and loads into GPU memory (~20 GB VRAM)
- Two PlotBrain instances initialize (standard + Schaefer atlas)
- This takes 1–3 minutes on first run, ~30 seconds on subsequent runs (cached)

You'll see:
```
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxxx.gradio.live
```

### Access the UI

**Option A** — Public link (works immediately):
Use the `https://xxxxx.gradio.live` URL printed in the terminal.

**Option B** — RunPod proxy:
If you exposed port 7860, go to your pod's **Connect** tab and click the **7860** port link.

## 6. Using the App

1. Paste your **OpenRouter API key** (get one at https://openrouter.ai/keys)
2. Upload a video (max 120 seconds — optimized for Reels/Shorts)
3. Click **Analyze**
4. Wait for processing (~3–5 minutes for a 60s video):
   - Video splits into 20s segments
   - TRIBE v2 processes each segment
   - Brain heatmaps and GIFs render
   - Claude Opus 4.6 generates the report
5. View results: animated GIFs, heatmap gallery, peak moments, full report
6. Download the HTML report and raw JSON data

## 7. Costs

| Item | Estimate |
|------|----------|
| RunPod RTX 6000 Pro | ~$0.74/hr (community cloud) |
| OpenRouter Claude Opus 4.6 | ~$0.50–1.00 per analysis (JSON + ~15 images) |

**Tip**: Stop the pod when not in use. The model re-downloads quickly from cache on restart.

## 8. Troubleshooting

See the full [Troubleshooting Guide](TROUBLESHOOTING.md) for solutions to common issues including:

- PyTorch + Blackwell GPU incompatibility (sm_120)
- hf_transfer not found (whisperx / uv environment)
- CUDA NVRTC errors
- torchvision / torchaudio version conflicts
- huggingface-cli not found
- OOM, ffmpeg, OpenRouter, and Gradio connection issues
