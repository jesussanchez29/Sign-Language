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
    # Abrir y mostrar la imagen subida
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Redimensionar la imagen al tamaño esperado por el modelo
    image = image.resize((64, 64))  # Ajusta esto según el tamaño que espera tu modelo

    # Convertir imagen a numpy array y normalizar
    image_array = np.array(image) / 255.0  # Normalización
    image_array = np.expand_dims(image_array, axis=0)  # Agregar batch dimension
    image_array = image_array.astype(np.float32)  # Asegurar tipo correcto

    # Verificar forma esperada por el modelo
    st.write(f"Forma esperada por el modelo: {model.input_shape}")
    st.write(f"Forma de la imagen final antes de predecir: {image_array.shape}")

    # Hacer la predicción
    prediction = model.predict(image_array)
    predicted_class = CLASSES[np.argmax(prediction)]  # Obtener la clase predicha

    st.write(f"### Número predicho: {predicted_class}")
