import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open webcam feed
cap = cv2.VideoCapture(0)


# Use to calculate hand angle (angle of wrist to middle finger and center vertical line)
def calculate_hand_angle(hand_landmarks):
    # Define wrist (landmark 0) and middle finger MCP (landmark 9)
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
    middle_mcp = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y])

    # Define the vertical up vector (0 degrees reference)
    vertical_vector = np.array([0, -1])  # Pointing up in y-direction

    # Calculate the hand vector from wrist (landmark 0) to middle finger MCP (landmark 9)
    hand_vector = middle_mcp - wrist

    # Calculate the angle in radians between the hand vector and the vertical vector
    angle_rad = np.arccos(
        np.dot(hand_vector, vertical_vector) / (np.linalg.norm(hand_vector) * np.linalg.norm(vertical_vector))
    )

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    # Determine the sign based on the x-coordinates
    if middle_mcp[0] > wrist[0]:
        angle_deg = abs(angle_deg)  # Positive angle
    else:
        angle_deg = -abs(angle_deg)  # Negative angle

    return angle_deg


# Calculate Euclidean distance between two points.
def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def is_in_line(base, middle, tip):
    """
    Check if three landmarks are in line:
    the middle point should lie between the base and tip.
    """
    # Check if the middle point lies between the base and tip by comparing distances
    return (
            calculate_distance(base, middle) < calculate_distance(base, tip)
            and calculate_distance(middle, tip) < calculate_distance(base, tip)
    )

# use to calculate straight fingers
def count_fingers(hand_landmarks):
    count = 0
    wrist = hand_landmarks.landmark[0]  # Wrist as the reference point

    # Thumb: Check if the thumb is straight using both distance and alignment conditions
    if (
            calculate_distance(wrist, hand_landmarks.landmark[4]) > calculate_distance(wrist,
                                                                                       hand_landmarks.landmark[3])
            and is_in_line(hand_landmarks.landmark[2], hand_landmarks.landmark[3], hand_landmarks.landmark[4])
    ):
        count += 1

    # Other fingers (Index to Pinky): Check distance and alignment for each finger
    for base, middle, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        if (
                calculate_distance(wrist, hand_landmarks.landmark[tip]) > calculate_distance(wrist,
                                                                                             hand_landmarks.landmark[
                                                                                                 middle])
                and is_in_line(hand_landmarks.landmark[base], hand_landmarks.landmark[middle],
                               hand_landmarks.landmark[tip])
        ):
            count += 1

    return count


while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert image color and process with MediaPipe
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Count extended fingers
            fingers_count = count_fingers(hand_landmarks)

            # Calculate and display hand angle
            hand_angle = calculate_hand_angle(hand_landmarks)
            cv2.putText(image, f'Hand Angle (degree): {hand_angle:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            # display number of straight fingers
            cv2.putText(image, f'Fingers: {fingers_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
