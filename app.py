import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Descargar el modelo desde Google Drive
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1-5wwsrLwl58SWvXpvZh6tFZEh0KWHLjs"
    output = "model.h5"

    # Descargar el modelo
    gdown.download(url, output, quiet=False)

    # Verificar si el archivo se descargó correctamente
    if not os.path.exists(output) or os.path.getsize(output) < 1000000:  # Ajusta el tamaño mínimo según tu modelo
        st.error("Error al descargar el modelo. Verifica el ID de Google Drive y que el archivo sea público.")
        return None

    try:
        return tf.keras.models.load_model(output)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

model = load_model()

if model is None:
    st.stop()  # Detiene la ejecución si el modelo no se carga correctamente

# Clases de signos (ajústalas según tu modelo)
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

st.title("Reconocimiento de Lenguaje de Señas")
st.write("Sube una imagen de un número en lenguaje de señas y el modelo lo identificará.")

uploaded_file = st.file_uploader("Subir imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)
    
    # Asegurar que tiene 3 canales (RGB)
    image = image.resize((64, 64)).convert("RGB")

    # Convertir a array de numpy y verificar la forma
    image_array = np.array(image) / 255.0  # Normalización entre 0 y 1
    st.write(f"Forma de la imagen antes de expand_dims: {image_array.shape}")

    # Expandir para batch (asegurar que es (1, height, width, channels))
    image_array = np.expand_dims(image_array, axis=0)
    st.write(f"Forma de la imagen final: {image_array.shape}")

    
    # Hacer la predicción
    prediction = model.predict(image_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    
    st.write(f"### Número predicho: {predicted_class}")
