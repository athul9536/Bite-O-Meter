import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('model/nail_biter_model.h5')
print("Model loaded.")

# Test folders
test_folders = ['dataset/test/Bite', 'dataset/test/NoBite']

# Loop through all images
for folder in test_folders:
    print(f"\nTesting images in {folder}...")
    for img_name in os.listdir(folder):
        if img_name.endswith('.jpg'):  # Only test .jpg files
            image_path = os.path.join(folder, img_name)

            # Load and preprocess the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading {image_path}")
                continue

            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            prediction = model.predict(img)[0][0]

            # Determine label
            label = "NoBite" if prediction > 0.5 else "Bite"

            # Print result
            print(f"{img_name}: {label} ({prediction:.2f})")
