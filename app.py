# -*- coding: utf-8 -*-
"""Image Enhancing Lab.

Concept adapted from Plotly's open-source `dash-image-enhancing` sample app (MIT licence, see LICENSE),
which upscaled images with ESRGAN on TensorFlow. This version is rebuilt by Pawel Sogra to run on
ordinary hosting (no TensorFlow, no GPU) and to measure what it does:

* upscaling with transparent methods (nearest, bilinear, bicubic, Lanczos) plus optional denoise/sharpen;
* a "degrade, restore, measure" test: shrink an image, enlarge it again, compare with the original
  using PSNR and SSIM, so the quality claims are checked rather than assumed.
"""
import base64
import os
import pathlib
from io import BytesIO

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

HERE = pathlib.Path(__file__).parent.resolve()
SAMPLES = {
    "Histology (IHC stain)": "histology-ihc.png",
    "Cell (phase image)": "cell-phase.png",
    "Coffee cup (photo)": "coffee.png",
    "Brick wall (texture)": "brick.png",
}
PORTFOLIO_URL = os.environ.get("PORTFOLIO_URL", "")

Image.MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard
MAX_PIXELS_IN = 25_000_000
MAX_SIDE_IN = 1600   # larger inputs are shrunk first so the demo stays responsive on small hosts
MAX_SIDE_OUT = 3200  # largest output side
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

METHODS = {
    "Nearest neighbour": Image.NEAREST,
    "Bilinear": Image.BILINEAR,
    "Bicubic": Image.BICUBIC,
    "Lanczos": Image.LANCZOS,
}
METHOD_NOTES = {
    "Nearest neighbour": "Copies pixels. Blocky, but adds nothing that is not there.",
    "Bilinear": "Averages neighbouring pixels. Smooth but soft.",
    "Bicubic": "Fits cubic curves. The usual default; a good balance.",
    "Lanczos": "Windowed sinc. Sharpest classical option; can ring near strong edges.",
}


# ----------------------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------------------
def decode_upload(contents):
    header, data = contents.split(",", 1)
    img = Image.open(BytesIO(base64.b64decode(data)))
    img = ImageOps.exif_transpose(img)
    if img.width * img.height > MAX_PIXELS_IN:
        raise ValueError("Image is larger than 25 megapixels.")
    return img.convert("RGB")


def load_sample(label):
    return Image.open(HERE / "assets" / "samples" / SAMPLES[label]).convert("RGB")


def limit_input(img):
    """Shrink very large inputs. Returns (image, note or None)."""
    if max(img.size) > MAX_SIDE_IN:
        out = img.copy()
        out.thumbnail((MAX_SIDE_IN, MAX_SIDE_IN), Image.LANCZOS)
        return out, f"Input was {img.width} x {img.height}; shrunk to {out.width} x {out.height} first."
    return img, None


def upscale(img, scale, method, sharpen, denoise):
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(3))
    out = img.resize((img.width * scale, img.height * scale), METHODS[method])
    if sharpen > 0:
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=int(sharpen), threshold=2))
    return out


def effective_scale(img, scale):
    allowed = max(1, MAX_SIDE_OUT // max(img.size))
    return min(scale, allowed)


def to_data_url(img, fmt="JPEG", quality=90):
    buf = BytesIO()
    img.save(buf, format=fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def crop_box(size, region, frac=0.25):
    w, h = size
    cw, ch = max(8, int(w * frac)), max(8, int(h * frac))
    xs = {"Centre": (w - cw) // 2, "Top left": 0, "Top right": w - cw, "Bottom left": 0, "Bottom right": w - cw}
    ys = {"Centre": (h - ch) // 2, "Top left": 0, "Top right": 0, "Bottom left": h - ch, "Bottom right": h - ch}
    x, y = xs[region], ys[region]
    return (x, y, x + cw, y + ch)


def scale_box(box, s):
    return tuple(int(v * s) for v in box)


def get_image(sample, contents):
    """Sample takes priority while selected; uploading an image clears the sample choice."""
    if sample:
        return load_sample(sample)
    if contents:
        return decode_upload(contents)
    return load_sample(next(iter(SAMPLES)))


# ----------------------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------------------
def kpi(label, value, sub=""):
    return dbc.Col(
        html.Div(
            [html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"), html.Div(sub, className="kpi-sub")],
            className="kpi-tile",
        ),
        md=3, xs=6,
    )


def image_card(src, header, sub=None):
    body = [html.Img(src=src, style={"width": "100%", "imageRendering": "auto"})]
    if sub:
        body.append(html.Div(sub, className="patch-label mt-1"))
    return dbc.Card([dbc.CardHeader(header), dbc.CardBody(body)], className="mb-3")


NAVBAR = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("Image Enhancing Lab", className="navbar-brand mb-0 h1"),
                    html.Span("Pawel Sogra | image analytics live demo", className="nav-sub"),
                ]
            ),
            html.A(
                "Back to portfolio", href=PORTFOLIO_URL or "#", className="btn btn-outline-light btn-sm",
                style={} if PORTFOLIO_URL else {"display": "none"},
            ),
        ],
        fluid=True, className="d-flex justify-content-between align-items-center",
    ),
    color="dark", dark=True, sticky="top",
)

INTRO = dbc.Alert(
    [
        html.Strong("What am I looking at? "),
        "Make a small image bigger and see how well common methods do. Pick a sample or upload your own, "
        "choose a method, then use the second tab to check each method against the original using PSNR and SSIM. "
        "Images are processed in memory and not stored.",
    ],
    color="light", className="mt-3 mb-0",
)

CONTROLS = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Image and settings"),
            html.Hr(className="my-2"),
            html.Label("Sample image", className="lead"),
            dcc.Dropdown(
                id="sample", options=list(SAMPLES), value=list(SAMPLES)[0], clearable=True,
                placeholder="Choose a sample...", style={"fontSize": 13},
            ),
            html.Label("Or upload your own", className="lead mt-3"),
            dcc.Upload(
                html.Div("Drag and drop, or click (JPG/PNG, up to 8 MB)", className="upload-box"),
                id="upload", multiple=False, max_size=MAX_UPLOAD_BYTES,
            ),
            html.Div(id="upload-msg", className="small mt-1"),
            html.Hr(),
            html.Label("Enlargement factor", className="lead"),
            dbc.RadioItems(
                id="scale", options=[{"label": f"{s}x", "value": s} for s in (2, 3, 4)], value=4, inline=True,
            ),
            html.Label("Method", className="lead mt-3"),
            dcc.Dropdown(id="method", options=list(METHODS), value="Bicubic", clearable=False, style={"fontSize": 13}),
            html.Div(id="method-note", className="small text-muted mt-1"),
            html.Label("Sharpening", className="lead mt-3"),
            dcc.Slider(id="sharpen", min=0, max=300, step=25, value=0, marks={0: "off", 150: "150%", 300: "300%"}),
            dbc.Checkbox(id="denoise", label="Reduce noise first (median filter)", value=False, className="mt-3"),
        ]
    ),
    className="mb-4",
)

ENHANCE_TAB = [
    html.Div(id="kpi-row", className="mb-3"),
    dbc.Alert(id="enh-note", color="info", is_open=False, className="py-2"),
    dbc.Row(
        [
            dbc.Col(html.Div("Zoom region:", className="pt-1"), width="auto"),
            dbc.Col(
                dcc.Dropdown(
                    id="region", options=["Centre", "Top left", "Top right", "Bottom left", "Bottom right"],
                    value="Centre", clearable=False, style={"minWidth": 150, "fontSize": 13},
                ),
                width="auto",
            ),
            dbc.Col(dbc.Button("Download enhanced PNG", id="dl-btn", color="primary", size="sm"), width="auto"),
        ],
        className="mb-3 g-2 align-items-center",
    ),
    dcc.Download(id="dl"),
    dcc.Loading(html.Div(id="enh-out")),
]

TEST_TAB = [
    html.P(
        "Test: the image is shrunk by the chosen factor (averaging pixels, like a low-resolution sensor), then "
        "enlarged back with each method using your sharpen/denoise settings, and compared with the original. "
        "Higher PSNR (dB) and SSIM (0 to 1) mean closer to the original.",
        className="small text-muted",
    ),
    dcc.Loading(html.Div(id="test-out")),
]

BODY = dbc.Container(
    [
        INTRO,
        dbc.Row(
            [
                dbc.Col(CONTROLS, md=3),
                dbc.Col(
                    dbc.Tabs(
                        [
                            dbc.Tab(ENHANCE_TAB, label="Enhance", tab_id="t1", className="pt-3"),
                            dbc.Tab(TEST_TAB, label="How good is it? (test)", tab_id="t2", className="pt-3"),
                        ],
                        id="tabs", active_tab="t1",
                    ),
                    md=9,
                ),
            ],
            className="mt-3",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("A note on limits"),
                    html.P(
                        "Classical methods make an image larger; they cannot recover detail that was never captured. "
                        "Deep-learning super-resolution (such as ESRGAN, used in the original version of this demo) can "
                        "synthesise plausible detail, but it may invent texture that was not in the scene. That is "
                        "acceptable for game textures and risky for medical, legal or forensic images. PSNR and SSIM also "
                        "tend to favour smooth results over sharp, realistic ones, so use them alongside visual checks.",
                        className="small mb-0",
                    ),
                ]
            ),
            className="mt-2 mb-4",
        ),
    ],
    fluid=True,
)

FOOTER = html.Div(
    [
        "Sample images are public-domain/CC0 files bundled with scikit-image. Concept adapted from Plotly's open-source ",
        html.A("dash-image-enhancing", href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-image-enhancing", target="_blank"),
        " (MIT licence). Rebuilt by Pawel Sogra to run without TensorFlow or a GPU, with measured quality tests.",
    ],
    id="footer",
)

app = dash.Dash(__name__)  # Bootstrap 5.3.3 (MIT) is bundled in assets/
app.title = "Image Enhancing Lab | Pawel Sogra"
server = app.server
app.layout = html.Div([NAVBAR, BODY, FOOTER])


# ----------------------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------------------
@app.callback(
    [Output("sample", "value"), Output("upload-msg", "children")],
    [Input("upload", "contents"), Input("upload", "last_modified")],
    prevent_initial_call=True,
)
def on_upload(contents, _lm):
    if not contents:
        return dash.no_update, dash.no_update
    return None, html.Span("Using your uploaded image.", className="text-success")


@app.callback(Output("method-note", "children"), Input("method", "value"))
def method_note(method):
    return METHOD_NOTES[method]


def safe_image(sample, contents):
    try:
        return get_image(sample, contents), None
    except Exception as exc:  # unreadable or too large
        return None, f"Could not read that image: {exc}"


@app.callback(
    [
        Output("kpi-row", "children"), Output("enh-out", "children"),
        Output("enh-note", "children"), Output("enh-note", "is_open"),
    ],
    [
        Input("sample", "value"), Input("upload", "contents"), Input("scale", "value"), Input("method", "value"),
        Input("sharpen", "value"), Input("denoise", "value"), Input("region", "value"),
    ],
)
def enhance(sample, contents, scale, method, sharpen, denoise, region):
    img, err = safe_image(sample, contents)
    if err:
        return [], [], err, True
    img, note = limit_input(img)
    s = effective_scale(img, scale)
    notes = [note] if note else []
    if s < scale:
        notes.append(f"Output capped at {MAX_SIDE_OUT}px on the long side, so {s}x was used.")
    if s == 1:
        return [], [], " ".join(notes + ["This image is already large; nothing to enlarge."]), True

    out = upscale(img, s, method, sharpen, denoise)
    box = crop_box(img.size, region)
    lo_patch = img.crop(box).resize((box[2] - box[0], box[3] - box[1]), Image.NEAREST)
    lo_patch = lo_patch.resize(((box[2] - box[0]) * s, (box[3] - box[1]) * s), Image.NEAREST)
    hi_patch = out.crop(scale_box(box, s))

    kpis = dbc.Row(
        [
            kpi("Input", f"{img.width} x {img.height}", f"{img.width * img.height / 1e6:.2f} megapixels"),
            kpi("Output", f"{out.width} x {out.height}", f"{s}x enlargement"),
            kpi("Method", method, "sharpened" if sharpen else "no sharpening"),
            kpi("Noise reduction", "On" if denoise else "Off", "median 3x3 before enlarging"),
        ],
        className="g-3",
    )
    body = [
        dbc.Row(
            [
                dbc.Col(image_card(to_data_url(img), "Original", f"{img.width} x {img.height}"), md=6),
                dbc.Col(image_card(to_data_url(out), f"Enhanced ({method}, {s}x)", f"{out.width} x {out.height}"), md=6),
            ]
        ),
        html.H6("Zoom: the same region, side by side", className="mt-2"),
        dbc.Row(
            [
                dbc.Col(image_card(to_data_url(lo_patch, "PNG"), "Original, pixels enlarged", "Each original pixel becomes a block"), md=6),
                dbc.Col(image_card(to_data_url(hi_patch, "PNG"), "Enhanced", f"{method}"), md=6),
            ]
        ),
    ]
    return kpis, body, " ".join(notes), bool(notes)


@app.callback(
    Output("dl", "data"),
    Input("dl-btn", "n_clicks"),
    [
        State("sample", "value"), State("upload", "contents"), State("scale", "value"), State("method", "value"),
        State("sharpen", "value"), State("denoise", "value"),
    ],
    prevent_initial_call=True,
)
def download(_n, sample, contents, scale, method, sharpen, denoise):
    img, err = safe_image(sample, contents)
    if err:
        return dash.no_update
    img, _ = limit_input(img)
    s = effective_scale(img, scale)
    if s == 1:
        return dash.no_update
    out = upscale(img, s, method, sharpen, denoise)
    buf = BytesIO()
    out.save(buf, format="PNG")
    return dcc.send_bytes(buf.getvalue(), f"enhanced_{s}x_{method.split()[0].lower()}.png")


@app.callback(
    Output("test-out", "children"),
    [
        Input("tabs", "active_tab"), Input("sample", "value"), Input("upload", "contents"), Input("scale", "value"),
        Input("sharpen", "value"), Input("denoise", "value"), Input("region", "value"),
    ],
)
def run_test(tab, sample, contents, scale, sharpen, denoise, region):
    if tab != "t2":
        return dash.no_update
    img, err = safe_image(sample, contents)
    if err:
        return dbc.Alert(err, color="warning")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    w, h = (img.width // scale) * scale, (img.height // scale) * scale
    if min(w, h) < scale * 8:
        return dbc.Alert("Image is too small to test at this factor.", color="warning")
    truth = img.crop((0, 0, w, h))
    low = truth.reduce(scale)  # box average, simulates a low-resolution capture

    t_arr = np.asarray(truth)
    rows, results = [], {}
    for name in METHODS:
        rec = upscale(low, scale, name, sharpen, denoise)
        r_arr = np.asarray(rec)
        psnr = peak_signal_noise_ratio(t_arr, r_arr, data_range=255)
        ssim = structural_similarity(t_arr, r_arr, channel_axis=2, data_range=255)
        results[name] = (rec, psnr, ssim)
        rows.append((name, psnr, ssim))

    names = [r[0] for r in rows]
    best = max(rows, key=lambda r: r[1])[0]
    colors = ["#2c82ff" if n == best else "#b8c7e0" for n in names]

    def bar(idx, title, fmt, rng=None):
        fig = go.Figure(
            go.Bar(x=names, y=[r[idx] for r in rows], marker_color=colors, text=[fmt.format(r[idx]) for r in rows], textposition="outside")
        )
        fig.update_layout(
            template="plotly_white", title=dict(text=title, font=dict(size=14)), height=300,
            margin=dict(t=45, b=30, l=40, r=10), yaxis=dict(range=rng), showlegend=False,
        )
        return fig

    psnr_vals = [r[1] for r in rows]
    ssim_vals = [r[2] for r in rows]
    psnr_fig = bar(1, "PSNR (dB, higher is better)", "{:.2f}", [min(psnr_vals) - 3, max(psnr_vals) + 2])
    ssim_fig = bar(2, "SSIM (higher is better)", "{:.3f}", [max(0, min(ssim_vals) - 0.08), min(1.0, max(ssim_vals) + 0.04)])

    box = crop_box(truth.size, region)
    patch = lambda im: im.crop(box).resize((200 * 2, 200 * 2 * (box[3] - box[1]) // (box[2] - box[0])), Image.NEAREST)
    patch_cols = [dbc.Col(image_card(to_data_url(patch(truth), "PNG"), "Original (ground truth)"), md=4, lg=True)]
    for name, (rec, psnr, ssim) in results.items():
        patch_cols.append(
            dbc.Col(image_card(to_data_url(patch(rec), "PNG"), name, f"PSNR {psnr:.2f} dB | SSIM {ssim:.3f}"), md=4, lg=True)
        )

    table = dbc.Table(
        [html.Thead(html.Tr([html.Th("Method"), html.Th("PSNR (dB)"), html.Th("SSIM")]))]
        + [
            html.Tbody(
                [
                    html.Tr(
                        [html.Td(n + ("  (best PSNR)" if n == best else "")), html.Td(f"{p:.2f}"), html.Td(f"{s_:.3f}")],
                        style={"fontWeight": 600} if n == best else {},
                    )
                    for n, p, s_ in rows
                ]
            )
        ],
        size="sm", striped=True, className="mb-0",
    )
    summary = dbc.Alert(
        f"Test image {truth.width} x {truth.height}, shrunk {scale}x to {low.width} x {low.height}, then enlarged back. "
        f"Best PSNR: {best}. Results depend on the image: textures and fine detail score lower than smooth areas.",
        color="info", className="py-2",
    )
    return [
        summary,
        dbc.Row([dbc.Col(dcc.Graph(figure=psnr_fig, config={"displayModeBar": False}), md=6),
                 dbc.Col(dcc.Graph(figure=ssim_fig, config={"displayModeBar": False}), md=6)]),
        table,
        html.H6("Zoom: same region from each result", className="mt-3"),
        dbc.Row(patch_cols, className="g-2"),
    ]


if __name__ == "__main__":
    app.run(debug=os.environ.get("DASH_DEBUG", "0") == "1", port=int(os.environ.get("PORT", 8050)))
