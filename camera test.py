# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:05:13 2024

CPS843 hand tracking test
"""

import cv2
import mediapipe as mp

# Open the default camera (0 for the default camera).
cap = cv2.VideoCapture(0)

#Identify and initialize hand module and hand tracking pipelines needed.
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#Module to draw the hand landmarks
mpDrawing = mp.solutions.drawing_utils


print("Press 'x' to exit the video feed.")

# Loop to continuously capture frames
while True:
    # Capture frame-by-frame from the camera input.
    ret, frame = cap.read()
    
    # If frame is not read, perminate the loop and the camera feed.
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert the image color to RGB in order for mediapipe to proces the hand.
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the entire frame to detect hands
    detectedHand = hands.process(rgbFrame)
    
    #If hands are detected, identify and draw the main components of the hand using our initialised module.
    if detectedHand.multi_hand_landmarks:
        for handComponent in detectedHand.multi_hand_landmarks:
            mpDrawing.draw_landmarks(frame, handComponent, mpHands.HAND_CONNECTIONS)
    
    #Display the current frame result
    cv2.imshow('Camera Feed', frame)
    
    #Exit the loop given an x keystroke
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the camera and close all OpenCV windows
cap.release()

cv2.destroyAllWindows()