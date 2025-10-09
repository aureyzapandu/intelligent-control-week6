import cv2

def canny_edge_detection(image_path):
    """Melakukan deteksi tepi dengan algoritma Canny dan mengembalikan array hasilnya"""
    img = cv2.imread(image_path, 0)  # baca gambar dalam grayscale
    if img is None:
        raise ValueError(f"Gagal membaca gambar dari path: {image_path}")
    
    edges = cv2.Canny(img, 100, 200)  # deteksi tepi
    return edges  # langsung return array, bukan path file


# Contoh penggunaan
canny_edge_detection(r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 6\intelligent-control-week6\rail-segmentation\train\images\1000195092_0012-0_jpeg.rf.3f5469298db2e943f4ddea9cf124c399.jpg")