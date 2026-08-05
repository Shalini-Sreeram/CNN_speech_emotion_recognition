import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model(r"C:\Users\Shalini Sreeram\CNN_speech_emotion_recognition\cnn_audio_model.h5")

# Define the class names and their corresponding indices
class_names = {
    0: '😢sadness',
    1: '😨fear',
    2: '🤢disgust',
    3: '😄joy',
    4: '😲surprise',
    5: '😐neutral',
    6: '😡anger'
}

# Load the label encoder classes
try:
    label_encoder_classes = np.load(r"C:\Users\Shalini Sreeram\CNN_speech_emotion_recognition\label_encoder_classes.npy")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
except FileNotFoundError:
    st.error("Label encoder file not found. Make sure 'label_encoder_classes.npy' is present.")
    st.stop()

# Function to extract audio features using Librosa
def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)  # Taking the mean across time
        return mfccs_processed
    except Exception as e:
        st.error(f"Error extracting features from {audio_path}: {str(e)}")
        return None

# Streamlit app
import base64

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Use your full path
img_base64 = get_base64_image(r"C:\Users\Shalini Sreeram\CNN_speech_emotion_recognition\static\img.jpeg")

# Inject CSS with base64 image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; color: #bcdfeb; text-shadow: 0 0 10px #00ffff;'>Speech Emotion Recognition</h1>",
    unsafe_allow_html=True
)
# File uploader
st.markdown("""
<style>
/* Uploader box styling */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #daa520;
    border-radius: 15px;
    background-color: #1974d2;
    padding: 20px;
    text-align: center;
    color: #399cbd;
}

/* Inner text styling (Drag & Drop, Limit info) */
[data-testid="stFileUploaderDropzone"] div {
    color: #dbceeb;                 /* text color */
    font-weight: bold;              /* bold text */
    font-size: 13px;                /* smaller size */
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Layout: move uploader left
col1, col2 = st.columns([2, 1])
with col1:
    # Custom label above uploader
    st.markdown(
        "<h4 style='color:#add8e6; font-size:14px;'>Upload an audio file</h4>",
        unsafe_allow_html=True
    )
    # File uploader widget
    uploaded_file = st.file_uploader("", type=["wav", "mp3"])

if uploaded_file is not None:
    # Extract audio features
    features = extract_audio_features(uploaded_file)
    
    if features is not None:
        # Prepare the features for prediction
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, axis=-1)  # Add channel dimension if needed
        
        # Predict emotion
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_emotion = class_names.get(predicted_class, "Unknown")
        
        st.success(f"Predicted Emotion: {predicted_emotion}")
    else:
        st.write("Could not extract features from the audio file.")
