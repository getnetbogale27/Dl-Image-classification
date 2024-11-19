import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('image_classification_model.h5')

# Define class labels (same as in your dataset)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Streamlit interface
st.title('Intel Image Classification App')
st.write("Upload an image and the model will classify it.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image (resize and normalize)
    image = np.array(image)
    image = cv2.resize(image, (150, 150))  # Resize to match model input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values to [0, 1]

    # Make prediction
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Prediction: {predicted_class}")
    
    # Display prediction confidence
    st.write(f"Confidence: {100 * np.max(prediction):.2f}%")

