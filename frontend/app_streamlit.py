import cv2
import numpy as np
import streamlit as st


st.set_page_config(page_title="Creative CV", page_icon="🖼️", layout="wide")

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

    section[data-testid="stSidebar"] {
        border-right: 1px solid #ececf3;
        background: linear-gradient(180deg, #f6f7ff 0%, #f9f9ff 100%);
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        background: #ffffff;
        border: 1px solid #e6e8f2;
        border-radius: 14px;
        padding: 10px;
        box-shadow: 0 8px 24px rgba(40, 44, 82, 0.06);
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        padding: 10px 12px;
        border-radius: 10px;
        margin-bottom: 4px;
        transition: background-color 0.2s ease;
        font-size: 1.08rem;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background-color: #f3f4ff;
    }
    section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        font-weight: 600;
        color: #23263a;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Creative CV")

section = st.sidebar.radio(
    "Навігація",
    ["Завантаження", "Порівняння", "Про проєкт"],
)


def _read_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Не вдалося прочитати зображення.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _demo_restore(rgb_image: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    restored = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)
    return cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)


if section == "Завантаження":
    st.header("Завантаження фото")

    model_name = st.selectbox("Модель", ["CDAE", "DnCNN", "U-Net"])
    uploaded = st.file_uploader("Оберіть зображення", type=["png", "jpg", "jpeg", "webp"])

    if uploaded is not None:
        image = _read_uploaded_image(uploaded)
        restored = _demo_restore(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Вхідне зображення")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Відновлений результат")
            st.image(restored, use_container_width=True)

        st.info(f"Обрана модель: {model_name}. Зараз показано демо-відновлення через OpenCV.")
    else:
        st.caption("Завантажте фото, щоб побачити попередній результат.")

elif section == "Порівняння":
    st.header("Порівняння моделей")
    st.write("Завантажте одне зображення, щоб переглянути макет порівняння моделей.")

    uploaded = st.file_uploader("Оберіть зображення для порівняння", type=["png", "jpg", "jpeg", "webp"])

    if uploaded is not None:
        image = _read_uploaded_image(uploaded)
        demo = _demo_restore(image)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.subheader("Вхід")
            st.image(image, use_container_width=True)
        with c2:
            st.subheader("CDAE")
            st.image(demo, use_container_width=True)
        with c3:
            st.subheader("DnCNN")
            st.image(demo, use_container_width=True)
        with c4:
            st.subheader("U-Net")
            st.image(demo, use_container_width=True)
    else:
        st.caption("Поки зображення не вибрано.")

else:
    st.header("Про проєкт")
    st.markdown(
        """
        Проєкт присвячений очищенню старих і зашумлених фото за допомогою моделей
        комп'ютерного зору: **CDAE**, **DnCNN** та **U-Net**.

        - Синтетична деградація: гаусівський шум, пуассонів шум, імпульсний шум, артефакти JPEG.
        - Оцінка: PSNR, SSIM, LPIPS.
        - Мета: покращити якість без втрати важливих деталей.
        """
    )
