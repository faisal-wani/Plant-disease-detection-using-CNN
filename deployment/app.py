import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import time

# Clear the session
tf.keras.backend.clear_session()

# Load the model
model = tf.saved_model.load('my_full_model_saved')
#model signature
infer = model.signatures["serving_default"]

# Define the class labels based on your dataset
class_names = {
    'Apple___Apple_scab': 0,
    'Apple___Black_rot': 1,
    'Apple___Cedar_apple_rust': 2,
    'Apple___healthy': 3,
    'Blueberry___healthy': 4,
    'Cherry_(including_sour)___Powdery_mildew': 5,
    'Cherry_(including_sour)___healthy': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
    'Corn_(maize)___Common_rust_': 8,
    'Corn_(maize)___Northern_Leaf_Blight': 9,
    'Corn_(maize)___healthy': 10,
    'Grape___Black_rot': 11,
    'Grape___Esca_(Black_Measles)': 12,
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
    'Grape___healthy': 14,
    'Orange___Haunglongbing_(Citrus_greening)': 15,
    'Peach___Bacterial_spot': 16,
    'Peach___healthy': 17,
    'Pepper,_bell___Bacterial_spot': 18,
    'Pepper,_bell___healthy': 19,
    'Potato___Early_blight': 20,
    'Potato___Late_blight': 21,
    'Potato___healthy': 22,
    'Raspberry___healthy': 23,
    'Soybean___healthy': 24,
    'Squash___Powdery_mildew': 25,
    'Strawberry___Leaf_scorch': 26,
    'Strawberry___healthy': 27,
    'Tomato___Bacterial_spot': 28,
    'Tomato___Early_blight': 29,
    'Tomato___Late_blight': 30,
    'Tomato___Leaf_Mold': 31,
    'Tomato___Septoria_leaf_spot': 32,
    'Tomato___Spider_mites Two-spotted_spider_mite': 33,
    'Tomato___Target_Spot': 34,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
    'Tomato___Tomato_mosaic_virus': 36,
    'Tomato___healthy': 37
}

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Match the model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.astype('float32')  # Ensure it's float32
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app interface
st.title("🌱 Plant Disease Detection 🌿")

# Custom CSS for the layout
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            color: #4CAF50;
            text-align: center;
            font-weight: bold;
        }
        .description {
            font-size: 18px;
            text-align: center;
            color: #555;
            margin-bottom: 30px;
        }
        .upload-button {
            font-size: 18px;
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
        }
        .upload-button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🌱 Plant Disease Detection 🌿</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Upload an image of a plant leaf to detect the disease.</p>', unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image with animation
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Show loading progress bar
    with st.spinner("Processing image..."):
        time.sleep(2)  # Simulate some delay while processing the image

        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Show progress bar while model is predicting
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.05)  # Simulate processing
            progress_bar.progress(i + 1)
        
        # Get predictions
        predictions = infer(tf.constant(processed_image, dtype=tf.float32))
        
        # Get the predicted class index (assuming the model outputs an array of probabilities)
        predictions_tensor = predictions[next(iter(predictions))]  # Get the first element (output tensor)
        class_idx = np.argmax(predictions_tensor, axis=1)[0]

        # Get the corresponding class label
        predicted_class = list(class_names.keys())[class_idx]

        # Display prediction and confidence with animations
        st.write(f"### 🎉 Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{np.max(predictions_tensor) * 100:.2f}%**")

        # Animated congratulations message
        st.balloons()  # This adds a balloon animation to celebrate the result
