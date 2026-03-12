import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# 1. Load the Model and Classes
model = tf.keras.models.load_model('model_semne.h5')
clase_litere = np.load('clase.npy', allow_pickle=True)

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --- VARIABLES FOR UI AND WORD FORMATION ---
cuvant_format = ""
litera_curenta = ""
cadre_consecutive = 0
CADRE_NECESARE = 15  # How many frames you need to hold the sign still (approx 1 sec)

print("Sistem pornit! Controale:")
print(" - Apasă 'C' pentru a șterge textul.")
print(" - Apasă 'SPACE' pentru spațiu.")
print(" - Apasă 'Q' pentru a ieși.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Keep the mirror effect so it feels natural to look at the screen
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands.process(rgb_frame)

    probabilitate = 0.0
    predictie_litera = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # SIMPLE EXTRACTION: No left/right modifications
            coordonate = []
            for lm in hand_landmarks.landmark:
                coordonate.extend([lm.x, lm.y, lm.z])
            
            # AI Prediction
            input_data = np.array([coordonate])
            predictie = model.predict(input_data, verbose=0)
            index_litera = np.argmax(predictie)
            predictie_litera = clase_litere[index_litera]
            probabilitate = predictie[0][index_litera]

            # Draw the skeleton on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- LETTER LOADING LOGIC ---
            if probabilitate > 0.85: # We are at least 85% sure
                if predictie_litera == litera_curenta:
                    cadre_consecutive += 1
                    # When the bar is full, add the letter to the word
                    if cadre_consecutive == CADRE_NECESARE:
                        cuvant_format += predictie_litera
                        cadre_consecutive = 0 # Reset
                else:
                    # Sign changed, reset the bar
                    litera_curenta = predictie_litera
                    cadre_consecutive = 0
            else:
                cadre_consecutive = 0

            # --- DRAWING THE UI FOR THE LETTER (TOP) ---
            if probabilitate > 0.85:
                # Text with the letter and percentage
                cv2.putText(frame, f"{predictie_litera} ({probabilitate*100:.0f}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                # Progress bar
                progres = int((cadre_consecutive / CADRE_NECESARE) * 150)
                cv2.rectangle(frame, (20, 70), (20 + progres, 85), (0, 255, 0), -1)
                cv2.rectangle(frame, (20, 70), (170, 85), (255, 255, 255), 2) # Border

    # --- BOTTOM BAR (FINAL TEXT) ---
    cv2.rectangle(frame, (0, h - 80), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, f"Text: {cuvant_format}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow('Traducator Limbaj Semne AI', frame)

    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): cuvant_format = ""
    elif key == ord(' '): cuvant_format += " "

cap.release()
cv2.destroyAllWindows()