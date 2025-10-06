from PIL import Image
import os

# Folder gốc chứa ảnh 640x640
input_folder = "valid/images"

# Folder lưu ảnh resized
output_folder = "valid/images_224/"
os.makedirs(output_folder, exist_ok=True)

# Resize tất cả ảnh
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")  # convert để chắc chắn 3 channel
        img_resized = img.resize((224, 224), Image.BILINEAR)
        
        save_path = os.path.join(output_folder, filename)
        img_resized.save(save_path)

print("Done! All images resized to 224x224.")
