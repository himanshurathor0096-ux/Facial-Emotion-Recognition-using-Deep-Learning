import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Facial Expression Classifier", page_icon="🙂")
st.title("Facial Expression Recognition")
st.write("Educational CNN demo. Predictions are dataset labels, not reliable measurements of a person's internal emotions.")

MODEL_PATH = "models/emotion_model.keras"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(image):
    image = image.convert("L").resize((48, 48))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(1, 48, 48, 1)

st.warning("Predictions can be inaccurate. Do not use this app for hiring, grading, policing, healthcare, access control, surveillance, or other high-stakes decisions. Use images only with permission.")
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Predict expression"):
        try:
            model = load_model()
            probs = model.predict(preprocess(image), verbose=0)[0]
            names = CLASS_NAMES
            best = int(np.argmax(probs))
            st.subheader("Predicted dataset label: " + names[best].title())
            st.caption(f"Model score: {probs[best] * 100:.1f}% — not a calibrated probability of a person's true emotion.")
            st.bar_chart({name: float(prob) for name, prob in zip(names, probs)})
        except (OSError, ValueError) as exc:
            st.error("Model not found or incompatible. Follow README.md to train the model first.")
            st.code(str(exc))
else:
    st.info("The trained model must exist at models/emotion_model.keras.")
