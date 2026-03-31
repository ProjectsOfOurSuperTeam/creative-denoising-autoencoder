import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_drawable_canvas import st_canvas

from src.degradation import NOTEBOOK_DEGRADATION_MODES, apply_degradation_pipeline_uint8

st.set_page_config(page_title="DAE Denoise Demo", page_icon="🖼️", layout="wide")

ZOOM = 12  # upscale factor for 28×28 previews (nearest-neighbor look)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }
    [data-testid="stSidebar"] { background: #f6f7fb; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _pixel_zoom(img: np.ndarray, z: int = ZOOM) -> np.ndarray:
    """(H,W) uint8 → (H*z, W*z) nearest-neighbor."""
    return np.repeat(np.repeat(img, z, axis=0), z, axis=1)


def _read_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to read image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _to_gray_28(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        gray = rgb
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)


def _on_reload_degrade():
    st.session_state["force_redegrade"] = True


@st.cache_resource
def _load_dae():
    from src.dae_inference import load_dae_from_checkpoint

    ckpt = PROJECT_ROOT / "dae_best.pt"
    model, meta = load_dae_from_checkpoint(ckpt, map_location=torch.device("cpu"))
    return model, meta


# ── session defaults ───────────────────────────────────────────────────────────
for k, v in [
    ("original_28", None),
    ("degraded_28", None),
    ("denoised_28", None),
    ("degraded_sig", None),
    ("force_redegrade", False),
    ("upload_fingerprint", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

st.title("Denoising autoencoder — 28×28 demo")
st.caption("Upload or draw a grayscale 28×28 input, build a degradation pipeline, then denoise.")

# ── Model ───────────────────────────────────────────────────────────────────
try:
    dae_model, ckpt_meta = _load_dae()
    with st.sidebar:
        st.subheader("Checkpoint")
        st.success(f"Loaded `{PROJECT_ROOT / 'dae_best.pt'}`")
        if isinstance(ckpt_meta, dict) and ckpt_meta.get("val_psnr") is not None:
            st.caption(
                f"Saved val PSNR: {ckpt_meta['val_psnr']:.2f} dB"
                if isinstance(ckpt_meta.get("val_psnr"), (int, float))
                else ""
            )
except FileNotFoundError as e:
    dae_model = None
    ckpt_meta = {}
    st.sidebar.error(str(e))
    st.sidebar.caption("Place `dae_best.pt` in the project root.")

# ── Input source ─────────────────────────────────────────────────────────────
st.subheader("1. Input")
src = st.radio("Source", ["Upload image", "Draw 28×28 (scaled canvas)"], horizontal=True)

if src == "Upload image":
    uploaded = st.file_uploader("Image (converted to grayscale, resized to 28×28)", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        fp = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
        if st.session_state["upload_fingerprint"] != fp:
            st.session_state["upload_fingerprint"] = fp
            rgb = _read_uploaded_image(uploaded)
            st.session_state["original_28"] = _to_gray_28(rgb)
            st.session_state["force_redegrade"] = True
            st.session_state["degraded_sig"] = None
            st.session_state.pop("denoised_28", None)
else:
    st.caption(
        "Полотно 280×280 → даунскейл до 28×28. Для прямокутників/кіл контур білий; заливка напівпрозора (можна накладати фігури)."
    )
    with st.expander("Інструменти канвасу", expanded=True):
        _tool_labels = {
            "freedraw": "Пензель (вільне малювання)",
            "line": "Лінія",
            "rect": "Прямокутник",
            "circle": "Коло / еліпс",
            "polygon": "Багатокутник",
        }
        drawing_mode = st.selectbox(
            "Інструмент",
            options=list(_tool_labels.keys()),
            format_func=lambda k: _tool_labels[k],
            key="canvas_tool",
        )
        stroke_w = st.slider("Товщина лінії / контуру", min_value=1, max_value=32, value=8, key="canvas_stroke")
        stroke_hex = st.color_picker("Колір контуру", value="#FFFFFF", key="canvas_stroke_color")
        # Semi-transparent fill for rect/circle/polygon (visible on black background)
        fill_hex = st.color_picker("Колір заливки (фігури)", value="#808080", key="canvas_fill_color")
        fill_a = st.slider("Прозорість заливки", 0, 100, 40, key="canvas_fill_alpha", help="0 = лише контур")

    def _hex_to_rgba(h: str, alpha_0_255: int) -> str:
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha_0_255 / 255.0:.3f})"

    fill_rgba = _hex_to_rgba(fill_hex, fill_a)

    canvas_result = st_canvas(
        fill_color=fill_rgba,
        stroke_width=int(stroke_w),
        stroke_color=stroke_hex,
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode=drawing_mode,
        key="dae_canvas",
    )
    if st.button("Use drawing as input"):
        if canvas_result.image_data is None:
            st.warning("Nothing drawn yet.")
        else:
            rgba = canvas_result.image_data.astype(np.uint8)
            rgb = rgba[..., :3]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            st.session_state["original_28"] = small
            st.session_state["force_redegrade"] = True
            st.session_state["degraded_sig"] = None
            st.session_state.pop("denoised_28", None)
            st.success("Input updated from canvas.")

orig = st.session_state["original_28"]
if orig is None:
    st.session_state["degraded_28"] = None
    st.session_state["degraded_sig"] = None
    st.session_state.pop("denoised_28", None)

# ── Degradation + preview ─────────────────────────────────────────────────────
st.subheader("2. Degradation pipeline")
st.caption("Order matters: degradations apply top-to-bottom in the list. Empty selection = no degradation.")

col_a, col_b, col_c = st.columns([1.1, 1.0, 1.1])

with col_a:
    st.markdown("**Original (28×28)**")
    if orig is not None:
        st.image(_pixel_zoom(orig), clamp=True, width=ZOOM * 28 + 2)
    else:
        st.info("Load or draw an image.")

with col_b:
    modes = st.multiselect(
        "Modes",
        list(NOTEBOOK_DEGRADATION_MODES),
        help="Same modes as in `main.ipynb` (parameters resampled when the pipeline changes or you click reload).",
        key="pipeline_modes_widget",
    )
    st.button(
        "Reroll degradations ↻",
        on_click=_on_reload_degrade,
        help="Resample random parameters for the same pipeline",
    )

# Recompute degraded when signature changes or reload requested
if orig is not None:
    sig = (orig.tobytes(), tuple(modes))
    if st.session_state["force_redegrade"] or st.session_state["degraded_sig"] != sig:
        if not modes:
            st.session_state["degraded_28"] = orig.copy()
        else:
            st.session_state["degraded_28"] = apply_degradation_pipeline_uint8(orig, modes)
        st.session_state["degraded_sig"] = sig
        st.session_state["force_redegrade"] = False
        st.session_state.pop("denoised_28", None)

deg = st.session_state.get("degraded_28")

with col_c:
    st.markdown("**Degraded preview**")
    if deg is not None:
        st.image(_pixel_zoom(deg), clamp=True, width=ZOOM * 28 + 2)
    elif orig is None:
        st.empty()

# ── Denoise ──────────────────────────────────────────────────────────────────
st.subheader("3. Denoise")
run = st.button("Denoise", type="primary", disabled=dae_model is None or deg is None)

if run and dae_model is not None and deg is not None:
    from src.dae_inference import denoise_gray_28

    try:
        st.session_state["denoised_28"] = denoise_gray_28(dae_model, deg, device=torch.device("cpu"))
    except Exception as e:
        st.error(f"Inference failed: {e}")

den = st.session_state.get("denoised_28")

if orig is not None and deg is not None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Original**")
        st.image(_pixel_zoom(orig), clamp=True, width=min(400, ZOOM * 28 + 2))
    with c2:
        st.markdown("**Degraded**")
        st.image(_pixel_zoom(deg), clamp=True, width=min(400, ZOOM * 28 + 2))
    with c3:
        st.markdown("**Denoised**")
        if den is not None:
            st.image(_pixel_zoom(den), clamp=True, width=min(400, ZOOM * 28 + 2))
        else:
            st.caption("Click **Denoise** to run the model.")
