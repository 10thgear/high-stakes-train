from ultralytics import YOLO

model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    model.train(data="highstakes.yaml", epochs=100)
    # model.train(data = 'safehat.yaml', epochs = 10, device = 'cpu')
    model.val()
