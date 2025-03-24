import time

import cv2  # python-opencv
import numpy as np
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt  # Add this import

# NOTE: This is the trained model
# TODO: checkout how to train multiple dataset and make it a single model?
model = YOLO("./yolo11n.onnx", task="detect")

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

frame_count = 0  # Counter for saving frames

while True:
    try:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        detected_frame = detect_objects(frame)

        if properties["SHOW_CLASS_NAME"]:
            try:
                cv2.imshow("Detected PPE Objects", detected_frame)
            except cv2.error as e:
                print(f"cv2.imshow failed: {e}")
                output_path = f"./output_frame_{frame_count}.jpg"
                cv2.imwrite(output_path, detected_frame)
                print(f"Frame saved as {output_path}")
                frame_count += 1

                # Display the saved frame using matplotlib
                img = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                plt.imshow(img)
                plt.title(f"Frame {frame_count}")
                plt.axis("off")
                plt.show()

        time.sleep(0.03)  # Adjust to control the speed

    except KeyboardInterrupt:
        print("Exiting gracefully...")
        break

video_capture.release()
cv2.destroyAllWindows()