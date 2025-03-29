import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import pygame
import time

model = YOLO("yolov10s.pt")

# Initialize sound
pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm.wav")  # Make sure this file exists
fall_detected = False
last_alarm_time = 0

# Load class names
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

cap = cv2.VideoCapture('./fall_videos/fall2.mov')

# Initialize tracking dictionary for person positions
previous_centers = {}
fall_threshold = 25  # Minimum downward movement in Y to consider a fall

count = 0
frame_index = 0

while True:
    ret, frame = cap.read()
    frame_index += 1
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

            # Calculate center Y position
            center_y = (y1 + y2) // 2
            object_id = index  # Use index as a temporary ID since we don't have trackers

            y_movement_flag = False

            if object_id in previous_centers:
                prev_y = previous_centers[object_id]
                delta_y = center_y - prev_y

                # Check for rapid downward movement
                if delta_y > fall_threshold:
                    y_movement_flag = True

            # Update center position for next frame
            previous_centers[object_id] = center_y

            # Trigger fall alert by either aspect ratio or Y-direction fall
            if aspect_ratio < 1 or y_movement_flag:
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
