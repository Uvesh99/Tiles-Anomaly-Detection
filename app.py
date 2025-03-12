# import streamlit as st
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# from PIL import Image

# # Class names for predictions
# class_names = ['Non Crack', 'Crack', 'Spot']

# # Function to load the selected model
# @st.cache_resource  # Cache models to avoid reloading
# def load_model_by_name(model_name):
#     if model_name == "CNN":
#         return load_model('tile_anomaly_detection_best_model.h5')
#     elif model_name == "ResNet50":
#         return load_model('tile_anomaly_detection_best_model (1).keras')
#     elif model_name == "EfficientNetB3":
#         return load_model('tile_anomaly_detection_model_efficientnet_l2.keras')
#     else:
#         raise ValueError("Invalid model selected")

# # Function to preprocess the uploaded image
# def preprocess_image(img, model_name):
#     if model_name in ["ResNet50", "EfficientNetB3"]:
#         # Convert PIL image to OpenCV format
#         img = np.array(img)
#         # Resize to 224x224 for ResNet50 and EfficientNet
#         img_resized = cv2.resize(img, (224, 224))
#         # Normalize and reshape for model input
#         img_preprocessed = img_resized.reshape(1, 224, 224, 3) / 255.0
#     else:  # Default preprocessing for CNN
#         img = img.resize((128, 128))  # Resize to match CNN input
#         img_preprocessed = np.array(img) / 255.0
#         img_preprocessed = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension

#     return img_preprocessed

# # Streamlit app layout
# st.title("Tile Anomaly Detection")
# st.write("Upload an image of tiles to predict their class using one of the available models.")

# # File uploader for images
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     img = Image.open(uploaded_file)
#     st.image(img, caption='Uploaded Image', use_column_width=True)

#     # Buttons for model selection
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         cnn_button = st.button("Predict with CNN")
#     with col2:
#         resnet_button = st.button("Predict with ResNet50")
#     with col3:
#         efficientnet_button = st.button("Predict with EfficientNetB3")

#     # Prediction logic based on button clicks
#     if cnn_button:
#         model_name = "CNN"
#     elif resnet_button:
#         model_name = "ResNet50"
#     elif efficientnet_button:
#         model_name = "EfficientNetB3"
#     else:
#         model_name = None

#     if model_name:
#         try:
#             st.write(f"Loading the {model_name} model...")
#             model = load_model_by_name(model_name)
#             st.success(f"{model_name} model loaded successfully!")

#             # Preprocess the image
#             img_array = preprocess_image(img, model_name)

#             # Make predictions
#             with st.spinner("Making predictions..."):
#                 predictions = model.predict(img_array)
#                 predicted_class_index = np.argmax(predictions[0])

#             # Display prediction result
#             st.write(f"Predicted Class using {model_name}: {class_names[predicted_class_index]}")

#             # Display prediction probabilities
#             st.write("Prediction Probabilities:")
#             for i, class_name in enumerate(class_names):
#                 st.write(f"{class_name}: {predictions[0][i]:.2%}")

#         except Exception as e:
#             st.error(f"An error occurred: {e}")


import streamlit as st
from tensorflow.keras.models import load_model
import gdown
import cv2
import numpy as np
from PIL import Image
import os

MODEL_URLS = {
    "CNN": "https://drive.google.com/uc?id=1xQwtcfo_A_SpGF4wuJR8GOLIwUVUB3QU",  # Example CNN Model
    "ResNet50": "https://drive.google.com/uc?id=1yaqOUqpz_JVBQcWTpQxsdKUyUvmPjiK7",  # Example ResNet50 Model
    "EfficientNetB3": "https://drive.google.com/uc?id=1iKEibCCvm2A_76zbtCk_zrD2JUDrVT0C"  # Example EfficientNet Model
}

# Class names for predictions
class_names = ['Non Crack', 'Crack', 'Spot']

# Function to download and load the selected model
@st.cache_resource
def load_model_by_name(model_name):
    model_path = f"{model_name}.h5" if model_name == "CNN" else f"{model_name}.keras"

    # Download model if not already downloaded
    if not os.path.exists(model_path):
        st.info(f"Downloading {model_name} model...")
        gdown.download(MODEL_URLS[model_name], model_path, quiet=False)
    
    return load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(img, model_name):
    if model_name in ["ResNet50", "EfficientNetB3"]:
        img = np.array(img)
        img_resized = cv2.resize(img, (224, 224))
        img_preprocessed = img_resized.reshape(1, 224, 224, 3) / 255.0
    else:
        img = img.resize((128, 128))
        img_preprocessed = np.array(img) / 255.0
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

    return img_preprocessed

# Streamlit app layout
st.title("Tile Anomaly Detection")
st.write("Upload an image of tiles to predict their class using one of the available models.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        cnn_button = st.button("Predict with CNN")
    with col2:
        resnet_button = st.button("Predict with ResNet50")
    with col3:
        efficientnet_button = st.button("Predict with EfficientNetB3")

    if cnn_button:
        model_name = "CNN"
    elif resnet_button:
        model_name = "ResNet50"
    elif efficientnet_button:
        model_name = "EfficientNetB3"
    else:
        model_name = None

    if model_name:
        try:
            st.write(f"Loading the {model_name} model...")
            model = load_model_by_name(model_name)
            st.success(f"{model_name} model loaded successfully!")

            img_array = preprocess_image(img, model_name)

            with st.spinner("Making predictions..."):
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])

            st.write(f"Predicted Class using {model_name}: {class_names[predicted_class_index]}")

            st.write("Prediction Probabilities:")
            for i, class_name in enumerate(class_names):
                st.write(f"{class_name}: {predictions[0][i]:.2%}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
