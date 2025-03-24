from ultralytics import YOLO

model = YOLO(r"./yolov5nu.pt")

model.export(format="onnx")  # creates 'yolo11n.onnx'

onnx_model = YOLO("yolov5nu.onnx")
