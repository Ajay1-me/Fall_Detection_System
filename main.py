import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

motion_triggered = False
alert_triggered = False
prev_position = None
fall_detected = False