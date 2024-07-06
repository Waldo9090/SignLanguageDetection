import cv2
import numpy as np
import mediapipe as mp
import os
import random



# Initialize media pipe model and detector for hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


# Function to create a new directory
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def frames_to_video(frames, output_path, fps=30):
    # Determine the shape of the frames
    height, width = frames[0].shape

    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose codec (mp4v for MP4)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # Write each frame to the video file
    for frame in frames:
        out.write(frame)

#Code to rotate the image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated




while True:
    exit = input("Press 'q' to quit and stop collecting data")
    
    if(exit.lower() == 'q'):
        break
    
    label = input("What letter would you like to collect video data for? ")

    # Directory where the array and video will be saved
    save_dir = os.path.expanduser("~/Documents/DataCollected")  # General "Documents" folder
    create_directory(save_dir)

    # File names for the saved array and video
    allVideoFrames_file = os.path.join(save_dir, "allVideoFrames.npy")
    allLabels_file = os.path.join(save_dir, "allLabels.npy")

    # Open a connection to the webcam (usually device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")

    frame_count = 0
    frames = []
    framesHorizontallyFlipped = []
    framesRotated = []
    framesCropped = []

    #Random angle for rotating augmentation later in code
    random_angle = random.uniform(-60, 60)  # Rotate by a random angle between -60 and 60 degrees when augmenting later

    #Random start values and percent for cropping augmentation later in code
    random_percent = random.uniform(75, 90)
    crop_h = int(80 * random_percent/100)
    crop_w = int(80 * random_percent/100)
    start_x = random.randint(0, 80 - crop_w)
    start_y = random.randint(0, 80 - crop_h)

    while frame_count < 20:  # Record 20 frames
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #OpenCV rake BGR but mediapipe takes RGB so conversion
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks: 
            for hand_landmarks in results.multi_hand_landmarks:

                # Process the hand landmarks
                h, w, c = frame.shape
                x_min, x_max = w, 0
                y_min, y_max = h, 0
                
                
                #Create bounding box by finding th eminimum and the maximum
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
                
                if x_min < x_max-1 and y_min < y_max-1:
                    buffer = 50

                    hand_region = frame[y_min-buffer:y_max+buffer, x_min-buffer:x_max+buffer]
                    checky, checkx = hand_region.shape[:2]
                    if checky > 0 and checkx > 0:
                        if len(frames) == 20:
                            break
                        
                        # Resizing images to be a uniform size
                        hand_region_resized = cv2.resize(hand_region, (80, 80)) 
                        
                        
                        blurred_image = cv2.bilateralFilter(hand_region_resized, d=9, sigmaColor=75, sigmaSpace=75)

                        # Convert the image to grayscale
                        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

                        # Apply adaptive thresholding
                        binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

                        
                        #Removing noise using morphological filters
                        binary_image_opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
                        binary_image_closed = cv2.morphologyEx(binary_image_opened, cv2.MORPH_CLOSE, (3,3))
                        
                        # Apply adaptive histogram equalization to enhance contrast
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        equalized_image = clahe.apply(binary_image_closed)

                        
                                                            
                        frames.append(equalized_image)
                        
                        #Apply flipping augmentation
                        horizontallyFlipped = cv2.flip(equalized_image, 1)
                        
                        framesHorizontallyFlipped.append(horizontallyFlipped)
                        
                        #Apply rotating augmentation
                        rotated_image = rotate_image(equalized_image, random_angle)  
                        
                        framesRotated.append(rotated_image)
                        
                        #Apply cropping of random percent of image
                        cropped = equalized_image[start_y:start_y + crop_h, start_x:start_x + crop_w]

                        # Resize the cropped image back to the original size
                        resized_cropped = cv2.resize(cropped, (80, 80), interpolation=cv2.INTER_LINEAR)
                        
                        framesCropped.append(resized_cropped)
                                        
                        frame_count += 1

        # Display the frame
        cv2.imshow('Recording...', frame)
        
        # Wait 200 milliseconds between frames
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
        
        




    # Convert list of frames to a 4D NumPy array
    framesHorizontallyFlipped_array = np.expand_dims(np.array(framesHorizontallyFlipped), axis=0)
    ori_array = np.expand_dims(np.array(frames), axis=0)
    rotated_array = np.expand_dims(np.array(framesRotated), axis=0)
    cropped_array = np.expand_dims(np.array(framesCropped), axis=0)
    frames_array = np.concatenate([framesHorizontallyFlipped_array,  ori_array, rotated_array, cropped_array], axis = 0)
    
    print(frames_array.shape)

    # Save frames to allVideoFrames.npy
    if os.path.exists(allVideoFrames_file):
        # Load existing frames and append new frames as a new batch
        existing_frames = np.load(allVideoFrames_file)
        frames_array = np.concatenate([existing_frames, frames_array], axis=0)

    # Save the combined frames_array to file
    np.save(allVideoFrames_file, frames_array)
    print(f"The shape of all video frames array {frames_array.shape}")  # Should be (Number of examples, 20, height, width, channels)
    print(f"Frames saved as NumPy array to {allVideoFrames_file}")

    # Save label to allLabels.npy
    labels_array = np.array([label] * 4)
    
    
    if os.path.exists(allLabels_file):
        # Load existing labels and append new label
        existing_labels = np.load(allLabels_file)
        labels_array = np.concatenate([existing_labels, labels_array], axis=0)

    # Save the combined labels_array to file
    np.save(allLabels_file, labels_array)
    print(f"The shape of all labels array {labels_array.shape}")
    print(f"Labels saved as NumPy array to {allLabels_file}")


    unique_letters = np.unique(labels_array)
    counts = np.zeros(len(unique_letters), dtype=int)

    for i, letter in enumerate(unique_letters):
        counts[i] = np.sum(labels_array == letter)

    letter_counts = dict(zip(unique_letters, counts))

    #Create general videos folder
    videos_folder = os.path.join(save_dir , "Videos")
    create_directory(videos_folder)

    #Combine frames to video and save as mp4 according to number
    frames_to_video(framesCropped, os.path.join(videos_folder , label + f"{letter_counts[label]-3}.mp4"))
    frames_to_video(framesRotated, os.path.join(videos_folder , label + f"{letter_counts[label]-2}.mp4"))
    frames_to_video(frames, os.path.join(videos_folder , label + f"{letter_counts[label]-1}.mp4"))
    frames_to_video(framesHorizontallyFlipped, os.path.join(videos_folder , label + f"{letter_counts[label]}.mp4"))


# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
