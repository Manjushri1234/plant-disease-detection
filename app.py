# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model('plant_disease_model.h5')
class_names = sorted(os.listdir('PlantVillage'))  # match folder names

# Title
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect the disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions)

    st.success(f"Prediction: **{class_names[class_idx]}** ({confidence*100:.2f}%)")
