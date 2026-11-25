# Image Enhancing Lab

Make a small image larger, then measure how well common methods do it.

**Live demo:** TODO add URL after deploying. **Portfolio:** TODO add URL.

## What it does
- Upscale a sample or your own image 2x to 4x with nearest, bilinear, bicubic or Lanczos
- Optional median-filter noise reduction and unsharp-mask sharpening
- Side-by-side zoom on any region, and PNG download
- **Test tab:** shrink the image, enlarge it back with each method, and score the result against the
  original with PSNR and SSIM, with zoomed patches to judge by eye

Samples are public-domain/CC0 images bundled with scikit-image, including an IHC histology slide.

## Honest notes
- Classical methods enlarge; they cannot recover detail that was never captured.
- Deep-learning super-resolution (ESRGAN, used by the original sample app) can synthesise plausible
  detail but may invent texture that was not in the scene. Fine for game textures; risky for medical,
  legal or forensic images.
- PSNR and SSIM favour smooth results over sharp, realistic ones. Use them with visual checks.
- Large inputs are shrunk to 1600 px on the long side and outputs capped at 3200 px to suit small hosts.
- Uploaded images are processed in memory and not stored.

## Run locally
```
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:8050
```

## Deploy on Render
Push this folder to GitHub, create a Blueprint from `render.yaml` (free tier sleeps after 15 minutes
idle; Starter is $7/month). Set `PORTFOLIO_URL` to show a back link.

## The original ESRGAN version
`legacy/` holds the original Plotly sample (TensorFlow + ESRGAN from TF Hub, plus a Colab notebook).
It was written for 2020 libraries and I have not run it; the model URL it uses (`tfhub.dev`) has moved,
and it needs a GPU to be practical. It is kept for reference only.

## Licence and attribution
MIT. Concept adapted from Plotly's
[dash-image-enhancing](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-image-enhancing)
sample app. Bootstrap 5.3.3 (MIT) is bundled in `assets/`. See `LICENSE`.
