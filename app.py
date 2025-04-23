import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = load_model('Image_Forgery_Model.h5')
input_shape = model.input_shape[1:]  # (height, width, channels)

# Image preprocessing
def process_image(image):
    img = ImageOps.fit(image, (input_shape[0], input_shape[1]), Image.Resampling.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = np.array(img) / 255.0
    img = img.reshape((1, *input_shape))
    return img

# Streamlit UI
st.title("Image Forgery Detection")
st.write("Upload an image, and the model will predict whether it is **forged** or **authentic**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = process_image(image)
    prediction = model.predict(processed_image)

    # Class names - adjust if your training had different order
    class_names = ['Authentic', 'Forged']

    predicted_class = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))
    class_label = class_names[predicted_class]

    st.markdown(f"### 🧐 Prediction: The image is likely **{class_label}**.")
    st.markdown(f"### 🔍 Confidence Level: **{confidence * 100:.2f}%**")
