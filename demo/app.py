import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("flood_model.h5")

IMG_SIZE = (128, 128)

def predict_flood(image):
    # Resize and normalize the image
    image = image.resize(IMG_SIZE).convert("RGB")
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict using the model
    prediction = model.predict(image_array)[0][0]
    label = "Flood Detected" if prediction >= 0.5 else "No Flood"
    confidence = float(prediction) if prediction >= 0.5 else 1 - float(prediction)
    
    return f"The photo shows: {label}"

app = gr.Interface(
    fn=predict_flood,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Flood Detection from Image",
    description="Upload an image and the model will predict whether there is flooding or not."
)

app.launch()
