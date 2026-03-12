import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the data from the CSV generated from Kaggle
df = pd.read_csv('date_kaggle_mediapipe.csv')

# 2. Prepare the data (X = points, y = letter)
X = df.drop('Litera', axis=1).values
y = df['Litera'].values

# Convert letters (A, B, C...) to numbers (0, 1, 2...)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. Build the AI Architecture (Simple and Fast)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(63,)), # 21 points * 3 coordinates
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Prevents overfitting
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax') # One output for each letter
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Training (Will take only a few seconds/minutes)
print("Începe antrenarea...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 5. Save the model and the list of letters
model.save('model_semne.h5')
np.save('clase.npy', encoder.classes_)

print("Antrenare finalizată! Modelul a fost salvat ca 'model_semne.h5'")