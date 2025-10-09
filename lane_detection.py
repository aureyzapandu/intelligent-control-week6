from ultralytics import YOLO
import cv2

# Load model YOLOv8 Instance Segmentation
model = YOLO("G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 6\intelligent-control-week6\yolov8n-seg.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path, show=True)
    results[0].save("lane_detection_result.jpg")
    
# Contoh penggunaan
detect_rail_lane(r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 6\intelligent-control-week6\rail-segmentation\test\images\1000195092_0020-0_jpeg.rf.9ffe2f9e851d8c87334fe548e7324f36.jpg")