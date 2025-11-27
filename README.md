# Facial Emotion Recognition Lab

## Description
This interface compares three models for real-time emotion recognition:
1. **LBP + KNN** (Baseline)
2. **HOG + Linear SVM**
3. **mini-Xception** (Deep Learning CNN)

## How to Run
1. Install dependencies:
   ```bash
   pip install opencv-python-headless scikit-learn scikit-image tensorflow keras gradio numpy joblib requests
2. Run the script:
    ```bash
    python face.py
3.The script will automatically download the required face detection XML and CNN model weights, and generate placeholder files for the traditional models if they are missing.
