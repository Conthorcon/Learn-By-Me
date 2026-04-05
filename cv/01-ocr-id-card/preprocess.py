import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp


CKP_PATH = "/home/truongkhanh/work/learn-by-me/cv/01-ocr-id-card/test-model/main/ckpt/seg/best_model.pth"
TEST_SAMPLE = "/home/truongkhanh/work/learn-by-me/cv/01-ocr-id-card/dataset/ABC/test_frames/image/image56.png"

DEVICE = "cpu"

def seg_id_card(img):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    model.load_state_dict(torch.load(CKP_PATH, weights_only=True))
    model.eval()

    img_cpy = img.copy()

    img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB)
    img_cpy = cv2.resize(img_cpy, (512, 512))

    x = img.transpose(2,0,1)/255.0
    x = torch.tensor(x).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)[0][0].cpu().numpy()
        pred = cv2.resize(pred, (img.shape[1], img.shape[0]))

    mask = (pred > 0.5).astype(np.uint8) * 255

    return mask

import cv2
import numpy as np

def find_four_corners(mask_image):
    # 1. Tìm contours
    contours, _ = cv2.findContours(
        mask_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, contours

    # 2. Lấy contour lớn nhất
    max_contour = max(contours, key=cv2.contourArea)

    # 3. Làm mượt góc bị vát bằng cách xấp xỉ đa giác (tùy chọn nhưng nên có)
    # Epsilon càng nhỏ, hình càng sát với thực tế. 0.02 là con số tiêu chuẩn.
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # Ép kiểu về mảng (N, 2)
    pts = contours[0].reshape(-1, 2)

    # 4. Logic Tổng và Hiệu để tìm 4 góc chuẩn
    # Khởi tạo danh sách 4 góc: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    rect = np.zeros((4, 2), dtype="float32")

    # Tổng x + y: Nhỏ nhất là Top-Left (0,0), Lớn nhất là Bottom-Right (W,H)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-Left
    rect[2] = pts[np.argmax(s)] # Bottom-Right

    # Hiệu y - x: Nhỏ nhất là Top-Right, Lớn nhất là Bottom-Left
    # (Vì tại Top-Right: x lớn, y nhỏ => y-x rất bé)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right
    rect[3] = pts[np.argmax(diff)] # Bottom-Left

    # Chuyển về list các mảng để giống định dạng cũ của bạn nếu cần
    return [rect[0], rect[1], rect[2], rect[3]], contours

def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))

    return dst

def preprocess():
    # 1. Đọc ảnh
    img = cv2.imread(TEST_SAMPLE)
    if img is None:
        print("Không thể tải ảnh. Kiểm tra lại đường dẫn TEST_SAMPLE!")
        return

    # 2. Phân đoạn để lấy mask (Hàm seg_id_card của bạn)
    mask = seg_id_card(img)

    # 3. Tìm 4 góc bằng logic Tổng và Hiệu
    corners, _ = find_four_corners(mask)

    corners = np.array(corners)

    # 4. Biến đổi ảnh khớp với kích thước
    warped = perspective_transform(img, corners)

    return warped


if __name__ == "__main__":
    start = time.perf_counter()
    preprocess()
    end = time.perf_counter()
    print(f"Running time: {(end - start):.6f} s")