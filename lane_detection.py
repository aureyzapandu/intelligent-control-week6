from ultralytics import YOLO
import cv2
import os

# Load model YOLOv8 Instance Segmentation
model = YOLO(r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 6\intelligent-control-week6\yolov8n-seg.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation dan menyimpan hasilnya otomatis di folder proyek"""
    
    # Path folder kamu (otomatis ambil dari path gambar)
    base_folder = os.path.dirname(r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 6\intelligent-control-week6")
    save_path = os.path.join(base_folder, "lane_detection.jpg")

    # Jalankan deteksi
    results = model(image_path)
    annotated_frame = results[0].plot()  # hasil dengan bounding box & mask
    
    # Tampilkan hasil sebentar
    cv2.imshow("Lane Detection", annotated_frame)
    cv2.waitKey(2000)  # tampil selama 2 detik
    cv2.destroyAllWindows()

    # Simpan hasil otomatis
    cv2.imwrite(save_path, annotated_frame)
    print(f"[✅] Hasil deteksi disimpan otomatis di:\n{save_path}")

# Contoh penggunaan
detect_rail_lane(
    r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 6\intelligent-control-week6\rail-segmentation\test\images\1000195092_0020-0_jpeg.rf.9ffe2f9e851d8c87334fe548e7324f36.jpg"
)
