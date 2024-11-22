import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

st.title("Image Classification with TensorFlow")

# Dataset paths
dataset_path = "C:/Users/TadeleBizuye/OneDrive - esxethiopia/Desktop/Laabza/Baacumen/M-7 Deep Learning(1)/Data"
dataset_path_train = os.path.join(dataset_path, "seg_train")
dataset_path_test = os.path.join(dataset_path, "seg_test")
dataset_path_pred = os.path.join(dataset_path, "seg_pred")


def load_images(folder_path, img_size=(150, 150)):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path))  # Sorted for consistent label ordering
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):  # Skip if not a directory
            continue
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)


with st.expander("Load and Preprocess Data"):
    train_images, train_labels = load_images(dataset_path_train)
    val_images, val_labels = load_images(dataset_path_test)

    st.write(f"Training Data: {train_images.shape}, Training Labels: {train_labels.shape}")
    st.write(f"Validation Data: {val_images.shape}, Validation Labels: {val_labels.shape}")

    train_images = train_images / 255.0
    val_images = val_images / 255.0

    st.write("Data normalization completed.")

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(train_images)
    st.write("Data augmentation applied.")

with st.expander("Define and Train the Model"):
    model = Sequential([
        Input(shape=(150, 150, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    st.write("Model compiled.")

    history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                        validation_data=(val_images, val_labels),
                        epochs=20)

    st.write("Model training completed.")
    st.write("Training and validation accuracy:")
    st.line_chart({"Training": history.history['accuracy'], "Validation": history.history['val_accuracy']})

with st.expander("Fine-Tune with Learning Rate Scheduler"):
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                        validation_data=(val_images, val_labels),
                        epochs=20,
                        callbacks=[lr_scheduler])

    st.write("Fine-tuning completed.")
    st.write("Updated training and validation accuracy:")
    st.line_chart({"Training": history.history['accuracy'], "Validation": history.history['val_accuracy']})

    model.save("image_classifier.keras")
    st.write("Model saved as `image_classifier.keras`.")

with st.expander("Upload Image for Prediction"):
    model = load_model("image_classifier.keras")

    def predict_image(image):
        img = cv2.resize(image, (150, 150)) / 255.0
        img = img[np.newaxis, ...]
        predictions = model.predict(img)
        return predictions.argmax()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file:
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        class_idx = predict_image(image)
        st.write(f"Predicted Class: {class_idx}")
