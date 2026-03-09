import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st
import torch

st.set_page_config(page_title="Creative CV", page_icon="🖼️", layout="wide")

_rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&family=Unbounded:wght@500;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    .stApp {
        font-size: 1.08rem;
    }
    p, label, li, .stMarkdown, .stCaption, .stSelectbox label, .stFileUploader label {
        font-size: 1.08rem !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Unbounded', sans-serif !important;
    }
    h1 {
        font-size: 2rem !important;
        line-height: 1.15 !important;
    }
    h2 {
        font-size: 1.8rem !important;
        line-height: 1.2 !important;
    }
    h3 {
        font-size: 1.45rem !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #ececf3;
        background: linear-gradient(180deg, #f6f7ff 0%, #f9f9ff 100%);
    }
    [data-testid="stSidebar"] * {
        color: #1e1e2e !important;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        background: #ffffff;
        border: 1px solid #e6e8f2;
        border-radius: 14px;
        padding: 10px;
        box-shadow: 0 8px 24px rgba(40, 44, 82, 0.06);
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        padding: 10px 12px;
        border-radius: 10px;
        margin-bottom: 4px;
        transition: background-color 0.2s ease;
        font-size: 1.08rem;
        color: #1e1e2e !important;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background-color: #f3f4ff;
    }
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        font-weight: 600;
        color: #1e1e2e !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.2rem;
    }
    [data-testid="stSidebar"] label {
        color: #1e1e2e !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Creative CV")

if "nav_section" not in st.session_state:
    st.session_state["nav_section"] = "Upload"
if "redirect_to" in st.session_state:
    st.session_state["nav_section"] = st.session_state.pop("redirect_to")

section = st.sidebar.radio(
    "Navigation",
    ["Upload", "Compare", "Degradation", "About"],
    key="nav_section",
)

st.sidebar.markdown("---")
color_mode = st.sidebar.radio(
    "Color mode",
    ["Auto", "Force grayscale", "Force RGB"],
    index=0,
    help="Auto: detect from image. Force grayscale: convert to gray. Force RGB: use RGB models.",
)


def _read_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to read image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _is_grayscale(rgb_image: np.ndarray) -> bool:
    """Check if image is essentially grayscale (all channels equal)."""
    if rgb_image.ndim == 2:
        return True
    return np.allclose(rgb_image[:, :, 0], rgb_image[:, :, 1]) and np.allclose(
        rgb_image[:, :, 0], rgb_image[:, :, 2]
    )


def _opencv_fallback(rgb_image: np.ndarray) -> np.ndarray:
    """Fallback via OpenCV when models are unavailable."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    restored = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)
    return cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)


@st.cache_resource
def _load_models(in_channels: int):
    """Load PyTorch models (grayscale or RGB)."""
    try:
        from src.models import load_best_model

        checkpoint_root = PROJECT_ROOT / "checkpoints"
        models = {}
        for name in ["cdae", "dncnn", "unet"]:
            try:
                model, _ = load_best_model(
                    name,
                    in_channels=in_channels,
                    checkpoint_root=checkpoint_root,
                    map_location=torch.device("cpu"),
                )
                models[name] = model
            except FileNotFoundError:
                pass
        return models if models else None
    except Exception as e:
        st.warning(f"Failed to load models: {e}")
        return None


def _restore_with_model(img: np.ndarray, model_name: str, models: dict) -> np.ndarray | None:
    """Restore image using selected model. img: (H,W) grayscale or (H,W,3) RGB."""
    key = model_name.lower().replace("-", "")
    if key not in models:
        return None
    try:
        from src.models import denoise_image

        model = models[key]
        if img.ndim == 2:
            img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, H, W)
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        out = denoise_image(model, img_t, patch_size=256, device=torch.device("cpu"))
        out_np = np.clip(out.numpy(), 0, 1)
        if out_np.shape[0] == 1:
            out_np = out_np.squeeze(0)  # (H, W)
        else:
            out_np = out_np.transpose(1, 2, 0)  # (H, W, 3)
        return (out_np * 255).astype(np.uint8)
    except Exception as e:
        st.error(f"Inference error: {e}")
        return None


def _restore_image(rgb_image: np.ndarray, model_name: str, use_rgb: bool) -> tuple[np.ndarray, str]:
    """
    Returns (restored_image, status_message).
    status: 'model' if PyTorch used, 'opencv' if fallback.
    """
    in_channels = 3 if use_rgb else 1
    models = _load_models(in_channels)

    if models and model_name:
        # Convert to grayscale for model if needed
        if not use_rgb and rgb_image.ndim == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            img_in = gray
        else:
            img_in = rgb_image

        restored = _restore_with_model(img_in, model_name, models)
        if restored is not None:
            if not use_rgb and rgb_image.ndim == 3:
                restored = cv2.cvtColor(restored, cv2.COLOR_GRAY2RGB)
            return restored, "model"
    return _opencv_fallback(rgb_image), "opencv"


if section == "Upload":
    st.header("Upload Photo")

    model_name = st.selectbox("Model", ["CDAE", "DnCNN", "U-Net"])
    image = None
    uploaded = None
    if st.session_state.get("degraded_for_upload") is not None:
        image = st.session_state["degraded_for_upload"]
        if st.button("Clear and upload new"):
            del st.session_state["degraded_for_upload"]
            _rerun()
    if image is None:
        uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded is not None:
            image = _read_uploaded_image(uploaded)

    if image is not None:
        is_gray = _is_grayscale(image)
        use_rgb = (
            color_mode == "Force RGB"
            or (color_mode == "Auto" and not is_gray)
        )
        restored, status = _restore_image(image, model_name or "CDAE", use_rgb)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Image")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("Restored Result")
            st.image(restored, use_column_width=True)

        restored_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        success, encoded_image = cv2.imencode(".png", restored_bgr)
        if success:
            original_stem = "degraded" if st.session_state.get("degraded_for_upload") is not None else (Path(uploaded.name).stem if uploaded is not None else "image")
            st.download_button(
                label="Save Result",
                data=encoded_image.tobytes(),
                file_name=f"{original_stem}_restored.png",
                mime="image/png",
            )

        if status == "model":
            st.success(f"Selected model: {model_name}. Using PyTorch.")
        else:
            st.info(
                f"Selected model: {model_name}. Checkpoints not found — showing OpenCV demo."
            )
    else:
        st.caption("Upload a photo to see the preview.")

elif section == "Compare":
    st.header("Model Comparison")
    st.write("Upload one image to compare results from all models.")

    image = None
    if st.session_state.get("degraded_for_compare") is not None:
        image = st.session_state["degraded_for_compare"]
        if st.button("Clear and upload new", key="clear_compare"):
            del st.session_state["degraded_for_compare"]
            _rerun()
    if image is None:
        uploaded = st.file_uploader("Choose image for comparison", type=["png", "jpg", "jpeg", "webp"])
        if uploaded is not None:
            image = _read_uploaded_image(uploaded)

    if image is not None:
        is_gray = _is_grayscale(image)
        use_rgb = (
            color_mode == "Force RGB"
            or (color_mode == "Auto" and not is_gray)
        )

        models = _load_models(3 if use_rgb else 1)
        results = {}
        for mn in ["CDAE", "DnCNN", "U-Net"]:
            if models:
                res, _ = _restore_image(image, mn, use_rgb)
            else:
                res = _opencv_fallback(image)
            results[mn] = res

        display_input = image
        if not use_rgb and image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            display_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.subheader("Input")
            st.image(display_input, use_column_width=True)
        with c2:
            st.subheader("CDAE")
            st.image(results["CDAE"], use_column_width=True)
        with c3:
            st.subheader("DnCNN")
            st.image(results["DnCNN"], use_column_width=True)
        with c4:
            st.subheader("U-Net")
            st.image(results["U-Net"], use_column_width=True)

        if not models:
            st.info("Checkpoints not found — all results via OpenCV.")
    else:
        st.caption("No image selected yet. Or add a degraded image from the Degradation page.")

elif section == "Degradation":
    st.header("Degradation")
    st.write("Upload an image, apply synthetic degradation, then use it on Upload or Compare.")

    deg_uploaded = st.file_uploader("Choose image to degrade", type=["png", "jpg", "jpeg", "webp"], key="deg_upload")
    if deg_uploaded is not None:
        orig_img = _read_uploaded_image(deg_uploaded)
        deg_file_key = getattr(deg_uploaded, "file_id", deg_uploaded.name)
        if st.session_state.get("deg_file_key") != deg_file_key:
            st.session_state.pop("degraded_img", None)
            st.session_state["deg_file_key"] = deg_file_key
        def _do_degrade():
            from src.degradation import apply_degradation
            bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            degraded_bgr = apply_degradation(bgr)
            st.session_state["degraded_img"] = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)

        if "degraded_img" not in st.session_state:
            _do_degrade()

        degraded = st.session_state["degraded_img"]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(orig_img, use_column_width=True)
        with col2:
            st.subheader("Degraded")
            st.image(degraded, use_column_width=True)
            if st.button("Regenerate degradation", key="regen_deg", help="Apply new random degradation without re-uploading"):
                _do_degrade()
                _rerun()

        st.caption("Use the degraded image on another page:")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Use on Upload", key="use_upload"):
                st.session_state["degraded_for_upload"] = degraded.copy()
                st.session_state["redirect_to"] = "Upload"
                _rerun()
        with c2:
            if st.button("Use on Compare", key="use_compare"):
                st.session_state["degraded_for_compare"] = degraded.copy()
                st.session_state["redirect_to"] = "Compare"
                _rerun()
    else:
        st.caption("Upload an image to apply degradation.")

else:
    st.header("About")
    st.markdown(
        """
        Project for restoring old and noisy photos using computer vision models:
        **CDAE**, **DnCNN**, and **U-Net**.

        - Synthetic degradation: Gaussian noise, Poisson noise, impulse noise, JPEG artifacts.
        - Grayscale and RGB support.
        - Metrics: PSNR, SSIM, LPIPS.
        - Goal: improve quality without losing important details.
        """
    )
