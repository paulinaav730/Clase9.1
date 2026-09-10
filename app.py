import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import platform


# ==============================
# INFORMACIÓN DE PYTHON
# ==============================

st.write("Versión de Python:", platform.python_version())


# ==============================
# CARGAR MODELO
# ==============================

model = load_model("keras_model.h5")

data = np.ndarray(
    shape=(1, 224, 224, 3),
    dtype=np.float32
)


# ==============================
# TÍTULO
# ==============================

st.title("🖼️ Reconocimiento de Imágenes")

st.write(
    "Usando un modelo entrenado en Teachable Machine "
    "puedes usarlo en esta aplicación para identificar imágenes."
)


# ==============================
# IMAGEN DE EJEMPLO
# ==============================

try:
    image = Image.open("OIG5.jpg")
    st.image(image, width=350)
except Exception:
    st.info("No se pudo cargar la imagen de ejemplo.")


# ==============================
# CÁMARA
# ==============================

img_file_buffer = st.camera_input("📷 Toma una foto")


# ==============================
# PROCESAR IMAGEN
# ==============================

if img_file_buffer is not None:

    # Abrir imagen tomada con la cámara
    img = Image.open(img_file_buffer).convert("RGB")

    # Mostrar imagen
    st.image(
        img,
        caption="Imagen capturada",
        use_container_width=True
    )

    # Redimensionar a 224x224
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convertir a numpy
    img_array = np.asarray(img)

    # Normalizar imagen
    normalized_image_array = (
        img_array.astype(np.float32) / 127.0
    ) - 1

    # Cargar imagen en el arreglo
    data[0] = normalized_image_array

    # ==============================
    # PREDICCIÓN
    # ==============================

    prediction = model.predict(data, verbose=0)

    st.subheader("🤖 Resultado")

    st.write("Predicciones:", prediction)


    # ==============================
    # CLASE 1
    # ==============================

    if prediction[0][0] > 0.5:

        st.success(
            "⬅️ Izquierda"
        )

        st.write(
            "Probabilidad:",
            f"{prediction[0][0] * 100:.2f}%"
        )


    # ==============================
    # CLASE 2
    # ==============================

    if prediction[0][1] > 0.5:

        st.success(
            "⬆️ Arriba"
        )

        st.write(
            "Probabilidad:",
            f"{prediction[0][1] * 100:.2f}%"
        )


    # ==============================
    # CLASE 3
    # ==============================

    if len(prediction[0]) > 2:

        if prediction[0][2] > 0.5:

            st.success(
                "➡️ Derecha"
            )

            st.write(
                "Probabilidad:",
                f"{prediction[0][2] * 100:.2f}%"
            )


    # ==============================
    # SI NO HAY UNA PREDICCIÓN CLARA
    # ==============================

    if np.max(prediction[0]) <= 0.5:

        st.warning(
            "🤔 No se pudo identificar un gesto con suficiente confianza."
        )
