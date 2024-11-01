import cv2
import mediapipe as mp
import numpy as np
import math

thumb_angle = 20.0

# print coordinators of each finger
# Define landmarks for each finger
finger_landmarks = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20]
}

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


# thumb tip angle
def calculate_thumb_angle(hand_landmarks):
    # Define wrist (landmark 0) and thumb tip (landmark 4)
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])

    # Define the vertical up vector (0 degrees reference) based on wrist
    vertical_vector = np.array([0, -1])  # Pointing up in the y-direction

    # Calculate the thumb vector from wrist (landmark 0) to thumb tip (landmark 4)
    thumb_vector = thumb_tip - wrist

    # Calculate the angle in radians between the thumb vector and the vertical vector
    angle_rad = np.arccos(
        np.dot(thumb_vector, vertical_vector) / (np.linalg.norm(thumb_vector) * np.linalg.norm(vertical_vector))
    )

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    # Determine the sign based on the x-coordinates
    if thumb_tip[0] > wrist[0]:
        angle_deg = abs(angle_deg)  # Positive angle if thumb tip is to the right of the wrist
    else:
        angle_deg = -abs(angle_deg)  # Negative angle if thumb tip is to the left of the wrist

    # if abs(angle_deg) >= thumb_angle:
    #     return True
    # else:
    #     return False

    return abs(angle_deg)


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


""" American Sigh Language part """


# Calculate Euclidean distance between two points.
def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


# American Sign Language
# Define a helper function to check if a finger is extended
def is_finger_extended(bot, base, mid, tip):
    return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mid].y < hand_landmarks.landmark[base].y < \
        hand_landmarks.landmark[bot].y


def check_A(hand_landmarks):
    if is_finger_extended(*finger_landmarks["Thumb"]):
        # fingers_to_check = ["Index", "Middle", "Ring", "Pinky"]  # Fingers to check
        if all(
                hand_landmarks.landmark[finger_landmarks[finger][3]].y > hand_landmarks.landmark[
                    finger_landmarks[finger][2]].y >
                hand_landmarks.landmark[finger_landmarks[finger][0]].y
                for finger in ["Index", "Middle", "Ring", "Pinky"]
        ):
            return True
    return False


# If all fingers are extended, this could be "B"
def check_B():
    if all(is_finger_extended(bot, base, mid, tip) for bot, base, mid, tip in
           [finger_landmarks[fingermarks] for fingermarks in ["Thumb", "Index", "Middle", "Ring", "Pinky"]]):
        # If all fingers are extended, this could be "B"
        return True
    return False


def check_C(hand_landmarks):

    """ Thumb finger is extended """
    """ "Index", "Middle", "Ring", "Pinky" fingers are semi extended """
    if is_finger_extended(*finger_landmarks["Thumb"]):
        if all(
                abs(hand_landmarks.landmark[finger_landmarks[finger][0]].y - hand_landmarks.landmark[
                    finger_landmarks[finger][1]].y) >
                2 * abs(hand_landmarks.landmark[finger_landmarks[finger][2]].y - hand_landmarks.landmark[
                    finger_landmarks[finger][3]].y)
                for finger in ["Index", "Middle", "Ring", "Pinky"]
        ):
            return True
    return False


def check_D(hand_landmarks):
    """ Index finger is extended """
    """ Thumb finger angle is smaller than 20 degree """
    """ "Middle", "Ring", "Pinky" fingers are not extended """
    thumb_angle = 20.0
    if is_finger_extended(*finger_landmarks["Index"]) and calculate_thumb_angle(hand_landmarks) < thumb_angle:
        if not all(is_finger_extended(bot, base, mid, tip) for bot, base, mid, tip in
                   [finger_landmarks[fingermarks] for fingermarks in ["Middle", "Ring", "Pinky"]]):
            # If all fingers, except Thumb are extended, this could be "E"
            return True
    return False


def check_E(hand_landmarks):
    # Calculate the thumb angle once
    """ E Thumb is inside the palm, 4 other fingers are not extended """
    thumb_angle = 20.0
    if calculate_thumb_angle(hand_landmarks) < thumb_angle:
        if not all(is_finger_extended(bot, base, mid, tip) for bot, base, mid, tip in
                   [finger_landmarks[fingermarks] for fingermarks in ["Index", "Middle", "Ring", "Pinky"]]):
            # If all fingers, except Thumb are extended, this could be "E"
            return True
    return False


def detect_asl_letter(hand_landmarks):
    """Detect simple ASL letters (A, B, C, D, E) based on hand landmarks."""

    # Check for specific ASL letters
    # check for A
    if check_A(hand_landmarks):
        return "A"
    elif check_B():
        return "B"
    elif check_D(hand_landmarks):
        # If index and middle fingers are extended, this could be "D"
        return "D"
    elif check_E(hand_landmarks):
        # If all fingers are curled towards the palm, this could be "E"
        return "E"
    elif check_C(hand_landmarks):
        # If fingers form a circular shape around the wrist, this could be "C"
        return "C"

    # If no matches found
    return "Unknown"


""" American Sigh Language part """

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

            # Get bounding box coordinates
            x_values = [landmark.x for landmark in hand_landmarks.landmark]
            y_values = [landmark.y for landmark in hand_landmarks.landmark]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)

            # Convert normalized coordinates to pixel coordinates
            image_height, image_width, _ = image.shape
            x_min, x_max = int(x_min * image_width), int(x_max * image_width)
            y_min, y_max = int(y_min * image_height), int(y_max * image_height)

            # Add padding to make the rectangle larger
            padding = 20  # You can adjust the padding value as needed
            x_min = max(x_min - padding, 0)
            x_max = min(x_max + padding, image_width)
            y_min = max(y_min - padding, 0)
            y_max = min(y_max + padding, image_height)

            # Draw the bounding rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Count extended fingers
            fingers_count = count_fingers(hand_landmarks)

            # Calculate and display hand angle
            hand_angle = calculate_hand_angle(hand_landmarks)
            cv2.putText(image, f'Hand Angle (degree): {hand_angle:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            # display number of straight fingers
            cv2.putText(image, f'Fingers: {fingers_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Detect ASL letter
            asl_letter = detect_asl_letter(hand_landmarks)
            # Display the detected ASL letter
            cv2.putText(image, f'ASL Letter: {asl_letter}', (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255),
                        2)

    # Display the image
    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
