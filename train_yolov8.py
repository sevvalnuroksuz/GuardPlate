from ultralytics import YOLO
import cv2
from ultralytics import YOLO
import easyocr
from datetime import datetime

# ✅ YOLOv8 modelini başlat
model = YOLO("yolov8n.yaml")  # n= nano, s=small, m=medium... ihtiyaca göre değiştirilebilir

# ✅ Eğitimi başlat
model.train(
    data="dataset/data.yaml",  # Roboflow'dan gelen .yaml dosyasının yolu
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs",  # Çıktılar bu klasöre kaydolur
    name="arac_plaka_model",  # Çıktı klasör ismi
    pretrained=True  # COCO ağırlıklarıyla başlat
)
