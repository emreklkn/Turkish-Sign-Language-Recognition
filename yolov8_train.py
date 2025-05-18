from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="tid_yolov8",
    device='cpu'   # GPU yoksa CPU olarak değiştir
)

print(results)
