import cv2
import torch
import numpy as np
from src.fist_model.model.model import FistDetection # Đảm bảo class này đã được định nghĩa đúng

# --- 1. Cấu hình và Tải Mô hình ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FistDetection().to(device)
# Tải trọng số mô hình đã huấn luyện
try:
    model.load_state_dict(torch.load("src/fist_model/trained.pth", map_location=device))
    model.eval()
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'src/fist_model/trained.pth'. Hãy kiểm tra lại đường dẫn.")
    exit()

# --- 2. Định nghĩa hằng số ---
INPUT_SIZE = 224
IMAGE_PATH = "src/fist_model/data/train/images_224/image_284.jpg" # THAY ĐỔI ĐƯỜNG DẪN NÀY ĐẾN FILE ẢNH CỦA BẠN

# --- 3. Đọc Ảnh ---
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {IMAGE_PATH}")
    exit()

# Lấy kích thước gốc của ảnh
original_height, original_width, _ = frame.shape

# --- 4. Tiền xử lý (Preprocessing) ---
# Sao chép khung hình gốc để xử lý
img = cv2.resize(frame.copy(), (INPUT_SIZE, INPUT_SIZE))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Chuyển đổi sang Tensor, chuẩn hóa và thêm chiều batch
input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
input_tensor = input_tensor.unsqueeze(0).to(device)

# --- 5. Chuyển tiếp (Forward Pass) và Dự đoán ---
with torch.no_grad():
    output = model(input_tensor)
    # Lấy kết quả (p, x, y, w, h)
    output = output[0].cpu().numpy() # (5,)

p, x, y, w, h = output

# --- 6. Hậu xử lý và Vẽ Bounding Box (Bbox) ---
if p > 0.5: # Kiểm tra độ tin cậy
    # Chuyển bbox về kích thước frame gốc
    x_center = int(x * original_width)
    y_center = int(y * original_height)
    box_w = int(w * original_width)
    box_h = int(h * original_height)

    # Tính toán tọa độ góc (x1, y1) và (x2, y2)
    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    # Đảm bảo tọa độ không vượt ra ngoài biên ảnh
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(original_width, x2), min(original_height, y2)

    # Vẽ bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Vẽ nhãn độ tin cậy
    text_label = f"Fist P: {p:.2f}"
    cv2.putText(frame, text_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    print(f"Phát hiện nắm đấm với độ tin cậy: {p:.2f} tại tọa độ: ({x1}, {y1}) - ({x2}, {y2})")
else:
    print(f"Không phát hiện nắm đấm với độ tin cậy > 0.5. Độ tin cậy cao nhất là: {p:.2f}")

# --- 7. Hiển thị Ảnh Kết Quả ---
cv2.imshow("Fist Detection on Static Image", frame)
print("\nNhấn phím bất kỳ để đóng cửa sổ.")

# Chờ một phím bấm để đóng cửa sổ (0 nghĩa là chờ vô hạn)
cv2.waitKey(0)
cv2.destroyAllWindows()