from multiprocessing import freeze_support

from ultralytics import YOLO


def main():
    model = YOLO("yolov5nu.pt")  # Use YOLOv5n model

    model.train(
        data="./datasets/data.yaml",
        epochs=1,
        imgsz=640,
        batch=8,
        name="model",
        device="cpu",
        workers=4,
    )


if __name__ == "__main__":
    freeze_support()
    main()
