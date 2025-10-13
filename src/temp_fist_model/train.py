# pip install ultralytics
from ultralytics import YOLO

if (__name__ == "__main__"):
    # 1. Tạo model YOLO nhỏ, random weights
    model = YOLO("src/temp_fist_model/best_scratch.pt")  # n = nano, kiến trúc nhỏ
    # chú ý: không load pretrained

    # 2. Train từ đầu
    model.train(
        data="src/temp_fist_model/data.yaml",   # cấu hình dataset (train/val paths, nc, names)
        epochs=25,             # chỉnh tùy dataset
        batch=16,              # batch size
        imgsz=224,             # kích thước ảnh
        pretrained=False       # quan trọng: weights khởi tạo từ đầu
    )

    # Lưu weights sau khi train xong
    model.save("src/temp_fist_model/best_scratch.pt")
