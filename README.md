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
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/59bc6a2c-11ea-4009-b6bf-40071e9461cd" />

   
- Extracted audio from TV show recordings  
- Preprocessing:  
  - **MFCCs** (Mel-frequency cepstral coefficients)  
  - **Spectral features**  

---

## 🧠 Model Details
- CNN layers for feature extraction
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/07261f06-ab5a-43a7-b869-2c720b1aaa44" />


- Training with multiple **epochs**

  
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/f7e46b58-2c47-4895-804e-5127f8a1e408" />

- Optimizers for convergence  
- **SMOTE** applied to handle class imbalance

<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/e7b564c0-e460-4399-89db-935ec826583f" />
 
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

<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/5178b346-c0a4-4ffe-b474-d5f10c3bee56" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/f5999feb-8447-4de4-9db9-242cb327492f" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/d6a789d4-4dda-4ac0-b9c6-0461cd3ad1b9" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/1b0c67bd-a636-4f1d-adbc-1a5ebb699c77" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/14abc426-a46a-48dd-9ffa-7a0ff5e3a0f2" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/ca8ccb24-95f5-4a28-9b93-f935fec8ca9e" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/5c9a59f1-f42a-4b25-bdf5-b39341150347" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/2b8204f0-504b-44d2-9803-060f14537157" />

- Confusion matrix and classification report included

<img width="562" height="622" alt="image" src="https://github.com/user-attachments/assets/ba5d039f-100b-4246-86de-8168937dd0ea" />

---
## 🔮 Future Work
- Extend to RNN/LSTM architectures
- Improve accuracy with advanced augmentation
- Deploy as a real‑time web/voice assistant module
