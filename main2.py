import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import pygame
import time

# Load YOLO model
model = YOLO("yolov10s.pt")

# Initialize sound
pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm.wav")  # Add your own alarm file here
fall_detected = False
last_alarm_time = 0

# Load class names
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# Open video
cap = cv2.VideoCapture('./fall_videos/fall6.mov')

count = 0
while True:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    fall_this_frame = False

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        if c == 'person':
            height = y2 - y1
            width = x2 - x1
            aspect_ratio = height / width if width != 0 else 0

            if aspect_ratio < 1:  # Possible fall if person is more horizontal
                cvzone.putTextRect(frame, 'FALL DETECTED', (x1, y1), 1, 2, colorR=(255, 0, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                fall_this_frame = True
            else:
                cvzone.putTextRect(frame, c, (x1, y1), 1, 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Play alarm only once per fall (every 5 seconds cooldown)
    current_time = time.time()
    if fall_this_frame and (not fall_detected or current_time - last_alarm_time > 5):
        alarm.play()
        fall_detected = True
        last_alarm_time = current_time
    elif not fall_this_frame:
        fall_detected = False

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
