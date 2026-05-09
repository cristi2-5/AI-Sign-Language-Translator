# AI Sign Language Translator

## 🌟 About the Project
This project started as a curiosity-driven experiment, built with the assistance of Large Language Models (LLMs). The goal was to create a real-time American Sign Language (ASL) translator using a standard webcam. The process of building, troubleshooting, and refining this application was so fascinating that it eventually sparked the idea for my bachelor's thesis!

## ❌ The First Attempt (And Why It Failed)
In the `first_try` directory, you'll find the initial approach: a Convolutional Neural Network (CNN) built with PyTorch. 
The idea was to feed raw, resized images (64x64 pixels) directly into the neural network to classify the signs. However, this approach had significant drawbacks:
- **Environmental Sensitivity:** The model was easily confused by different lighting conditions, complex backgrounds, and varying skin tones.
- **Computational Overhead:** Processing raw image pixels in real-time is computationally heavy.
- **Inaccuracy in Real-Time:** While it might perform decently on static dataset images, it struggled to provide stable and reliable predictions on a live, noisy webcam feed.

## 🧠 The Successful Approach (MediaPipe + Keras)
To solve the issues of the first attempt, the strategy shifted from processing *pixels* to processing *geometry*. 

### 1. Data Processing
Instead of feeding raw images to the AI, I utilized **Google's MediaPipe** (`second_try/procesare_kaggle.py`). MediaPipe is a highly optimized framework that detects hands and extracts **21 3D landmarks** (x, y, z coordinates) representing the joints and fingertips.
This step transformed the dataset from heavy image files into a lightweight, highly structured CSV file (`date_kaggle_mediapipe.csv`). The AI no longer had to learn what a hand or background looks like; it only needed to learn the spatial relationships between these 21 specific points for each letter.

### 2. Making it Ambidextrous (Data Augmentation)
To ensure the model could understand signs from both the left and right hand, I applied a clever mathematical mirroring technique (`second_try/model_right_hand.py`). By simply inverting the X-coordinates (`x = 1.0 - x`), I simulated the opposite hand. This effectively doubled the training dataset without needing to collect any new images!

### 3. Model Training
With the data simplified down to just 63 numbers per frame (21 landmarks * 3 axes), I built a much lighter Feed-Forward Neural Network using **TensorFlow/Keras**. 
The architecture consists of a few Dense layers with Dropout to prevent overfitting. Because the input data is purely structural, the model (`model_semne.h5`) trains incredibly fast, requires very little processing power, and achieves high accuracy.

### 4. How the Live Translator Works
The real-time application (`second_try/live.py`) ties everything together beautifully:
1. **Capture:** The webcam captures the live video feed.
2. **Extract:** MediaPipe instantly extracts the 21 hand landmarks from the current frame.
3. **Predict:** The trained Keras model receives these coordinates and predicts the corresponding ASL letter.
4. **Stabilize (The "Loading Bar"):** To prevent the screen from filling up with random letters during transitions, I implemented a stabilization logic. A letter is only typed onto the screen if the model predicts it with **>85% confidence for 15 consecutive frames** (acting like a visual loading bar).

## 🛠️ Requirements
- Python 3.x
- OpenCV (`cv2`)
- MediaPipe
- TensorFlow / Keras
- Pandas, NumPy, Scikit-learn
