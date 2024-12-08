from ultralytics import YOLO

model = YOLO("yolov8s.pt") 


dataset_yaml = "dataset.yaml"

num_workers = 12

model.train(
    data=dataset_yaml,
    epochs=25, 
    batch=8,
    imgsz=320,
    device='cpu',
    workers=num_workers 
)

model.save("yolov8_trained.pt")
