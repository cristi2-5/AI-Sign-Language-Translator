import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("1. Încarc datele originale...")
df = pd.read_csv('date_kaggle_mediapipe.csv')

print("2. Generez datele pentru cealaltă mână (Oglindire matematică)...")
df_oglindit = df.copy()

# Find all columns representing the X coordinate (x0, x1, x2... x20) and invert them
for i in range(21):
    nume_coloana_x = f'x{i}'
    if nume_coloana_x in df_oglindit.columns:
        df_oglindit[nume_coloana_x] = 1.0 - df_oglindit[nume_coloana_x]

# Merge the two tables (right hand + left hand)
df_final = pd.concat([df, df_oglindit], ignore_index=True)
print(f"Datele au fost dublate! Avem acum {len(df_final)} de exemple de antrenament.")

# 3. Prepare the data
X = df_final.drop('Litera', axis=1).values
y = df_final['Litera'].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Build the Architecture (Same as before)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Training
print("\nÎncepe antrenarea modelului AMBIDEXTRU...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 6. Save the new model
model.save('model_semne.h5')
np.save('clase.npy', encoder.classes_)

print("\nGATA! Modelul știe acum ambele mâini și a fost salvat.")