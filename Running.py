import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization

# Initialize media pipe model and detector for hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

letters = ['a', 'b', 'c', 'd', 'f', 'j']

# Model architecture
input_shape = (20, 80, 80, 1)
kernel_size = (3, 3, 3)
lambda_val = 0.002

# Create model
model = Sequential()

# Convolutional Layers
# Block 1
model.add(Conv3D(filters=64, kernel_size=kernel_size, input_shape=input_shape, activation='relu', padding='same', kernel_regularizer=l2(lambda_val)))
model.add(BatchNormalization())
model.add(MaxPooling3D((2, 2, 2)))

# Block 2
model.add(Conv3D(filters=128, kernel_size=kernel_size, activation='relu', padding='same', kernel_regularizer=l2(lambda_val)))
model.add(BatchNormalization())
model.add(MaxPooling3D((2, 2, 2)))

# Block 3
model.add(Conv3D(filters=256, kernel_size=kernel_size, activation='relu', padding='same', kernel_regularizer=l2(lambda_val)))
model.add(BatchNormalization())
model.add(MaxPooling3D((2, 2, 2)))

# Flatten and fully connected layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(6, activation='softmax'))

# Load model weights
model_dir = os.path.expanduser("~/Documents/DataCollected/FINALmodel_weights.h5")
model.load_weights(model_dir)

# Open a connection to the webcam (usually device 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
    
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Process hand landmarks to determine bounding box
            h, w, c = frame.shape
            x_min, x_max = w, 0
            y_min, y_max = h, 0

            for lm in hand_landmarks.landmark:
                if lm is not None:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y

            if x_min < x_max - 1 and y_min < y_max - 1:
                buffer = 50
                hand_region = frame[y_min - buffer:y_max + buffer, x_min - buffer:x_max + buffer]

                check_y, check_x = hand_region.shape[:2]
                if check_y > 0 and check_x > 0:
                    

                    # Resize hand region to match model input size
                    hand_region_resized = cv2.resize(hand_region, (80, 80))

                    # Preprocess the image for prediction
                    blurred_image = cv2.bilateralFilter(hand_region_resized, d=9, sigmaColor=75, sigmaSpace=75)
                    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
                    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                    binary_image_opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
                    binary_image_closed = cv2.morphologyEx(binary_image_opened, cv2.MORPH_CLOSE, (3, 3))
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    equalized_image = clahe.apply(binary_image_closed)

                    frames.append(equalized_image)

                    # Make prediction
                    if(len(frames) > 20):
                        frames_array = np.reshape(np.array(frames[-20:]), (1, 20, 80, 80, 1))
                        prediction = model.predict(frames_array)
                        highest_prob_index = np.argmax(prediction)
                        predicted_letter = letters[highest_prob_index]

                        # Display predicted letter on the frame
                        cv2.putText(frame, f'Predicted Letter: {predicted_letter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Recording...', frame)

    # Check for 'q' key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
