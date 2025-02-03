import time

import cv2  # python-opencv
import numpy as np
import yaml
from ultralytics import YOLO

# NOTE: This is the trained model
# TODO: checkout how to train multiple dataset and make it a single model?
model = YOLO("./yolov5nu.onnx", task="detect")

with open("./properties.yaml", "r") as properties_file:
    properties = yaml.safe_load(properties_file)


# used to detect and draw the bounding box
def detect_objects(frame):
    results = model(frame)
    for result in results:
        boxes = result.boxes  # select all boxes
        for box in boxes:

            class_id = int(box.cls)  # Get the class ID , {0,1,2...}
            print(f"class id : {class_id}")
            class_name = model.names[class_id]
            print(f"class name : {class_name}")
            confidence = float(box.conf)
            print(f"confidence : {confidence}")
            # print(f"Detected: {class_name} ({confidence:.2f})")  # Print to terminal
    annotated_frame = results[
        0
    ].plot()  # Get the annotated frame with bounding boxes and labels
    return annotated_frame


# Start video capture from USB camera
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    detected_frame = detect_objects(frame)

    if properties["SHOW_CLASS_NAME"]:
        cv2.imshow("Detected PPE Objects", detected_frame)
        print(f"Object detected")

    time.sleep(0.03)  # Adjust to control the speed

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
