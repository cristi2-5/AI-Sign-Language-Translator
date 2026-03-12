import os
import cv2
import mediapipe as mp
import csv

# 1. Configure MediaPipe for STATIC IMAGES
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # IMPORTANT: Set to True for processing photos
    max_num_hands=1,         # Looking for a single hand in the image
    min_detection_confidence=0.5
)

# --- MODIFY THIS PATH ---
# Set the exact path to the main folder containing the subfolders A, B, C...
DATASET_PATH = r"dataset/asl_alphabet_train/asl_alphabet_train" 
CSV_FILE = "date_kaggle_mediapipe.csv"

print("Încep procesarea imaginilor. Acest lucru poate dura câteva minute...")

# 2. Create/Open the CSV file to write the results
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    
    # Write the table header (Header)
    header = ['Litera']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    writer.writerow(header)

    # 3. Go through each letter folder (A, B, C...)
    for litera in os.listdir(DATASET_PATH):
        folder_litera = os.path.join(DATASET_PATH, litera)
        
        # Make sure it's a folder, not a stray file
        if not os.path.isdir(folder_litera):
            continue
            
        print(f"Procesez pozele pentru litera: {litera}...")
        imagini_procesate_cu_succes = 0
        
        # 4. Go through each image in the letter folder
        for nume_imagine in os.listdir(folder_litera):
            cale_imagine = os.path.join(folder_litera, nume_imagine)
            
            # Read the image
            img = cv2.imread(cale_imagine)
            if img is None:
                continue
                
            # MediaPipe works with RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = hands.process(img_rgb)
            
            # 5. If a hand was found, save the coordinates
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    rand_date = [litera] # First element in the row is the label (A, B, etc.)
                    for lm in hand_landmarks.landmark:
                        rand_date.extend([lm.x, lm.y, lm.z]) # Add the coordinates
                    
                    writer.writerow(rand_date)
                    imagini_procesate_cu_succes += 1

        print(f"  -> Am extras punctele din {imagini_procesate_cu_succes} imagini pentru litera {litera}.")

print(f"\nGATA! Toate datele structurate au fost salvate în fișierul: {CSV_FILE}")