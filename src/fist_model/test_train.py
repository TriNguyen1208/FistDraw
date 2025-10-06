import cv2
import numpy as np
import os

def draw_bbox_from_label(image_path, label_path):
    """
    Đọc ảnh và file label (.txt) để vẽ bounding box.
    File label chứa: x_center y_center w h (tất cả đều thuộc [0, 1])
    """
    # --- 1. Đọc ảnh gốc ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"LỖI: Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    H, W, _ = image.shape
    
    # Tạo bản sao để vẽ lên
    output_image = image.copy()
    
    # --- 2. Đọc file label ---
    bboxes = []
    if not os.path.exists(label_path) or os.stat(label_path).st_size == 0:
        print(f"CẢNH BÁO: File label không tồn tại hoặc rỗng: {label_path}. Không có bbox nào được vẽ.")
    else:
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    # Loại bỏ khoảng trắng và chia chuỗi
                    parts = line.strip().split()
                    
                    # Giả sử file chỉ chứa x_center y_center w h (4 giá trị)
                    if len(parts) >= 4:
                        # Chuyển đổi sang float và lưu
                        x, y, w, h = map(float, parts[:4])
                        bboxes.append((x, y, w, h))
                    else:
                        print(f"CẢNH BÁO: Dòng label không hợp lệ, bỏ qua: {line.strip()}")
        except Exception as e:
            print(f"LỖI khi đọc file label: {e}")
            return

    # --- 3. Vẽ Bounding Box ---
    if bboxes:
        for x_norm, y_norm, w_norm, h_norm in bboxes:
            # Chuyển đổi từ tọa độ chuẩn hóa [0, 1] sang tọa độ pixel
            
            # Tính toán kích thước pixel
            box_w = int(w_norm * W)
            box_h = int(h_norm * H)
            
            # Tính toán tọa độ tâm pixel
            x_center = int(x_norm * W)
            y_center = int(y_norm * H)

            # Tính toán tọa độ góc trên bên trái (x1, y1) và góc dưới bên phải (x2, y2)
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)
            
            # Đảm bảo tọa độ nằm trong biên độ ảnh
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

            # Vẽ hình chữ nhật (Màu xanh lá cây)
            color = (0, 255, 0) 
            thickness = 2
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
            
            # Thêm nhãn (nếu cần, ở đây chỉ là một placeholder)
            label_text = "Object"
            cv2.putText(output_image, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
            
            print(f"Vẽ bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")


    # --- 4. Hiển thị Ảnh ---
    cv2.imshow("Annotated Image", output_image)
    cv2.waitKey(0) # Chờ một phím bấm
    cv2.destroyAllWindows()


# --- CHẠY CHƯƠNG TRÌNH ---

# Đặt đường dẫn file của bạn
# VÍ DỤ 1: File label hợp lệ
num = 400
IMAGE_FILE = f"src/fist_model/data/train/images_224/image_{num}.jpg" 
LABEL_FILE = f"src/fist_model/data/train/labels/image_{num}.txt"

# VÍ DỤ 2: File label rỗng (hoặc không tồn tại)
# IMAGE_FILE = "my_other_image.jpg"
# LABEL_FILE = "empty_label.txt" 

# Lưu ý: Cần có file ảnh thực tế và file label tương ứng để kiểm tra!
draw_bbox_from_label(IMAGE_FILE, LABEL_FILE)