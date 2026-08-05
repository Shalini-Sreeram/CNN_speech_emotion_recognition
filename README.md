# 🎙️ CNN Speech Emotion Recognition

## 📖 Overview
This project implements a **Convolutional Neural Network (CNN)** to classify emotions from speech audio files (`.wav` or `.mp3`, up to 200 MB). It analyzes the input and predicts one of **7 distinct emotions**:
- Disgust  
- Anger  
- Fear  
- Joy  
- Sadness  
- Neutral  
- Surprise  

The model achieves **81% accuracy** and is designed for integration into **chatbots and voice assistants**.

---

## 📊 Dataset
- **Source:** MELD dataset (Multimodal EmotionLines Dataset)  
- Extracted audio from TV show recordings  
- Preprocessing:  
  - **MFCCs** (Mel-frequency cepstral coefficients)  
  - **Spectral features**  

---

## 🧠 Model Details
- CNN layers for feature extraction  
- Training with multiple **epochs**  
- Optimizers for convergence  
- **SMOTE** applied to handle class imbalance  
- Evaluation metrics:  
  - Confusion Matrix  
  - Accuracy, Precision, Recall, F1-score  

---
## 🚀 Usage
- Be in the project directory.
- Run the Streamlit app:
    - streamlit run app.py
- Upload a .wav or .mp3 file (≤200 MB).
- Get the predicted emotion instantly.

 - **tip:**💡 If errors occur, check that file paths in the code are correct.

---
## 📈 Results
- Accuracy: 81%
- Balanced performance across 7 emotion classes
- Confusion matrix and classification report included

---
## 🔮 Future Work
- Extend to RNN/LSTM architectures
- Improve accuracy with advanced augmentation
- Deploy as a real‑time web/voice assistant module
