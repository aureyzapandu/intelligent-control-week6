from ultralytics import YOLO
import cv2
import numpy as np
from canny_edge import canny_edge_detection

# Load YOLOv8 Instance Segmentation model
model = YOLO(r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 6\intelligent-control-week6\yolov8n-seg.pt")

def combined_detection(image_path):
    """Menggabungkan Canny Edge Detection dengan Lane Detection"""

    # Jalankan Canny Edge Detection (hasil dikembalikan sebagai array)
    edges = canny_edge_detection(image_path)

    # Jalankan Lane Detection dengan YOLOv8-seg
    results = model(image_path)
    lane_img = results[0].plot()

    # Overlay hasil Lane Detection dengan Canny Edge
    combined = cv2.addWeighted(lane_img, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # Simpan dan tampilkan hasil
    cv2.imshow("Combined Detection", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("combined_result.jpg", combined)
    print("Hasil gabungan disimpan sebagai 'combined_result.jpg'")

# Contoh penggunaan
combined_detection(
    r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 6\intelligent-control-week6\rail-segmentation\valid\images\1000195092_0005-0_jpeg.rf.00e151ef8c1d507805f6382d515299b8.jpg"
)
