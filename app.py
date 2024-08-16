import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress all logs except errors

# Define labels  (adjust according to your model's classes)
labels = ['fresh apple', 'fresh banana', 'fresh orange', 'rotten apple', 'rotten banana', 'rotten orange']

# Load model
@st.cache_resource # Cache the model loading
def load_model():
    model = tf.keras.models.load_model('./pretrained_fruit_classification.h5', compile=False)
    return model

with st.spinner('Model is being loaded...'):
    model = load_model()
print("Loading model completed")

# Preprocess the image
def preprocess_image(image):
    """Preprocess the image to the required input shape for the model"""
    image = image.resize((224, 224)) # Adjust size according to model's input size
    image = np.array(image) # convert to numpy array
    image = image / 255.0 # Normalizing pixel values
    image = np.expand_dims(image, axis=0) # Adding a dimension to convert it into 3D array
    return image

# Predict the label
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

st.markdown(
 """
 <style>
 .prediction {
     font-size: 20px;
     color: #008080;
 }
 </style>
 """,
 unsafe_allow_html=True
)

# Streamlit app
st.title(f"🍍 Fruit Classification 🍍")
st.markdown("---")  # Add a horizontal rule for separation

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Predict image
    prediction = predict(image)
    predicted_label = labels[np.argmax(prediction)]
    prediction_prob = np.max(prediction)
    # Print the result
    st.markdown("---")  # Add a horizontal rule for separation
    st.markdown(f"<p class='prediction'>Prediction: {predicted_label} - confidence score: {prediction_prob * 100:.2f} %</p>", unsafe_allow_html=True)