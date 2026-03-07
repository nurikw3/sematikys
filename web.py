import streamlit as st
from PIL import Image
import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["background", "hair", "skin"]

INFER_WIDTH = 256
INFER_HEIGHT = 256

MODEL_PATH = "models/best_model_new.onnx"
IMAGE_DISPLAY_WIDTH = 600

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


@st.cache_resource
def load_model():
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(MODEL_PATH, providers=providers)


def preprocess(image):
    h, w = image.shape[:2]

    scale = min(INFER_WIDTH / w, INFER_HEIGHT / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    pad_w = INFER_WIDTH - new_w
    pad_h = INFER_HEIGHT - new_h

    padded = cv2.copyMakeBorder(
        resized,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_w // 2,
        pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    img = padded.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.astype(np.float32)

    img = img.transpose(2, 0, 1)
    img = img[np.newaxis]

    return img, h, w


@st.cache_data(show_spinner="Running model")
def infer_image(_session, image):

    x, original_h, original_w = preprocess(image)

    input_name = _session.get_inputs()[0].name
    output_name = _session.get_outputs()[0].name

    pr_mask = _session.run([output_name], {input_name: x})[0]

    mask = np.argmax(pr_mask.squeeze(0), axis=0)

    mask = cv2.resize(
        mask.astype(np.uint8),
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask


def create_overlay(image, mask):

    colors = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
        ],
        dtype=np.uint8,
    )

    color_mask = colors[mask]

    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

    return overlay


def adjust_hsv(image, mask, h_adjust, s_adjust, v_adjust, index):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    h, s, v = cv2.split(hsv)

    roi = mask == index

    h[roi] = np.clip(h[roi] + h_adjust, 0, 179)
    s[roi] = np.clip(s[roi] + s_adjust, 0, 255)
    v[roi] = np.clip(v[roi] + v_adjust, 0, 255)

    hsv = cv2.merge([h, s, v]).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def main():

    st.set_page_config(
        page_title="Image Adjustment Tool",
        layout="wide",
    )

    st.title("Image Adjustment Tool")

    session = load_model()

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if uploaded_file is None:
        st.info("Upload an image to start")
        return

    image = np.array(Image.open(uploaded_file).convert("RGB"))

    st.sidebar.header("Adjustment settings")

    h_adjust = st.sidebar.slider("Hue", -179, 179, 0)
    s_adjust = st.sidebar.slider("Saturation", -255, 255, 0)
    v_adjust = st.sidebar.slider("Value", -255, 255, 0)

    region = st.sidebar.selectbox("Region", CLASSES)
    index = CLASSES.index(region)

    mask = infer_image(session, image)

    overlay = create_overlay(image, mask)

    adjusted_image = adjust_hsv(image, mask, h_adjust, s_adjust, v_adjust, index)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Original")
        st.image(image, width=IMAGE_DISPLAY_WIDTH)

    with col2:
        st.caption("Segmentation")
        st.image(overlay, width=IMAGE_DISPLAY_WIDTH)

    with col3:
        st.caption("Adjusted")
        st.image(adjusted_image, width=IMAGE_DISPLAY_WIDTH)


if __name__ == "__main__":
    main()
