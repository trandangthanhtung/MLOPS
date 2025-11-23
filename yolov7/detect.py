import os
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

# -----------------------------
# Cấu hình
# -----------------------------
weights = 'weights/yolov7.pt'
img_folder = 'img'
output_folder = 'output'
img_size = 640  # YOLOv7 mặc định
conf_thres = 0.25
iou_thres = 0.45
device = select_device('')  # '' sẽ tự chọn GPU nếu có, CPU nếu không

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
print(f"Đang load model từ: {weights}")
model = attempt_load(weights, map_location=device)
model.eval()

# -----------------------------
# Lấy danh sách tất cả ảnh
# -----------------------------
image_files = sorted([
    f for f in os.listdir(img_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

print(f"Tìm thấy {len(image_files)} ảnh để xử lý.")

# -----------------------------
# Xử lý từng ảnh
# -----------------------------
for img_name in image_files:
    img_path = os.path.join(img_folder, img_name)
    img0 = cv2.imread(img_path)
    if img0 is None:
        print(f"⚠️ Không đọc được ảnh {img_name}, bỏ qua.")
        continue

    # Resize và chuẩn hóa ảnh
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Forward
    with torch.no_grad():
        pred = model(img)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Vẽ bbox
    for det in pred:  # chỉ có 1 batch
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{int(cls)} {conf:.2f}'
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img0, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Lưu ảnh kết quả
    out_path = os.path.join(output_folder, img_name)
    cv2.imwrite(out_path, img0)
    print(f"✔ Saved: {out_path}")

print("🎉 Hoàn tất detect tất cả ảnh!")
