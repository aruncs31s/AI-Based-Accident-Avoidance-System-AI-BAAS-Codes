# NOTE: Should run only on raspberry pi
import time
from RPLCD.i2c import CharLCD
import cv2  # python-opencv
import numpy as np 
import yaml
from ultralytics import YOLO

# sudo i2cdetect -y 1
LCD_ADDR = 0x27

class LCD:
    def __init__(self,ADDR):
        self.lcd = CharLCD('PCF8574', ADDR, cols=16, rows=2)
        # Clear the LCD
        self.lcd.clear()
        self.lcd.write_string(u"Object Detection")
        time.sleep(5)
        # Move to the second line
        self.lcd.crlf()

    def display(self,status,class_name):
        # TODO: Check if this is neccessary
        self.lcd.clear()
        # self.lcd.cursor_pos = (1, 0)
        if status:
            self.lcd.write_string("Object Detected")
            self.lcd.crlf()
            self.lcd.write_string(u" {class_name}")
        else:
            self.lcd.write_string("No Object Detected")

        


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


def main():
    newLCD = LCD(LCD_ADDR)
    model = YOLO("src/yolo11n.onnx", task="detect")
    with open("src/properties.yaml", "r") as properties_file:
        properties = yaml.safe_load(properties_file)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Video error")
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
            newLCD.display(True,)
        time.sleep(0.03)  # Adjust to control the speed

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
