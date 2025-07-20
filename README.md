Driver Drowsiness Detection
Author: Md Nadeem Ashraf Ansari
Date: July 2025

Overview :
This project is a real-time driver drowsiness detection system using computer vision. It monitors the driver’s face through a webcam to detect signs of fatigue such as yawning and dangerous head nodding. It is intended as a road safety tool to help prevent accidents related to drowsiness behind the wheel.

Features :
Real-time yawning detection

Head pose detection for nodding (simulating drowsiness)

Camera-based monitoring with live alerts

Simple and efficient for personal computers

How It Works :
Uses your webcam to capture live video of your face

Detects when your mouth is wide open (yawning) for several frames

Detects when you nod your head down in a way that indicates drowsiness

Displays alerts ("Yawning Detected!", "Drowsy Head Nod!") directly on the webcam feed

Requirements :
Python 3.12 (64-bit recommended)

dlib

opencv-python

imutils

numpy

You also need the dlib shape predictor file:

Download and place shape_predictor_68_face_landmarks.dat in your project folder.

Installation :
Clone or download this repository to your PC.

Install required Python packages:

bash
pip install opencv-python dlib imutils numpy
If dlib fails, see how to install using a precompiled wheel.

Download the predictor:

Direct download link (Google Drive - Request Access required) : https://drive.google.com/file/d/1xHn7ChLvUWYDAdaBTWfrlIOAEg-VzztS/view?usp=drive_link

Place shape_predictor_68_face_landmarks.dat in the same folder as main_detection.py.

How to Run :
Connect your webcam.

Open a terminal in the project folder.

Run:

bash
python main_detection.py
Look at the camera and simulate drowsy behavior:

Open your mouth wide (simulate yawning)

Nod your head down (simulate sleepy nodding)

Output:

The webcam window will display "Yawning Detected!" or "Drowsy Head Nod!" if those signs are detected.

To quit, focus the webcam window and press the q key.

Files and Structure
text
drowsiness_detection_enhanced/
│
├── main_detection.py
├── requirements.txt
├── shape_predictor_68_face_landmarks.dat
├── [models/]         # (empty, for future ML models)
├── [utils/]          # (empty, for future helper scripts)
├── [app/]            # (empty, for future apps)
main_detection.py: Main script for webcam detection.

shape_predictor_68_face_landmarks.dat: Landmark model for face detection.

requirements.txt: List of required Python packages.
