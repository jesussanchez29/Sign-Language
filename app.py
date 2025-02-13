import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Clases de signos (ajústalas según tu modelo)
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

st.title("Reconocimiento de Lenguaje de Señas")
st.write("Sube una imagen de un número en lenguaje de señas y el modelo lo identificará.")

uploaded_file = st.file_uploader("Subir imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)
    
    # Preprocesamiento de la imagen (ajústalo a las dimensiones de tu modelo)
    image = image.resize((64, 64))  # Cambia al tamaño correcto
    image_array = np.array(image) / 255.0  # Normalización
    image_array = np.expand_dims(image_array, axis=0)  # Agregar batch dimension
    
    # Hacer la predicción
    prediction = model.predict(image_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    
    st.write(f"### Número predicho: {predicted_class}")
