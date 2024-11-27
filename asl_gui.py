import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Load the pre-trained ASL recognition model and labels
model = tf.keras.models.load_model("mp-landmarks-to-asl-nn.keras")
label_classes = np.load("label_classes.npy", allow_pickle=True)

# Mediapipe setup for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Start the video capture
cap = cv2.VideoCapture(0)  # Open the default camera
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Camera initialized successfully.")

def process_frame(frame):
    """Process the frame to detect hand landmarks and predict ASL sign."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and normalize hand landmark coordinates
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)

            # Predict ASL sign using the model
            prediction = model.predict(landmarks)
            predicted_class = label_classes[np.argmax(prediction)]
            confidence = np.max(prediction)
            print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

            # Display prediction on the frame
            if confidence > 0.7:
                cv2.putText(frame, f"ASL: {predicted_class}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "ASL: Uncertain", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Main loop to process video frames
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally for a mirrored view
        frame = process_frame(frame)  # Process the frame for ASL detection

        cv2.imshow("ASL Recognition", frame)  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()