import matplotlib.pyplot as plt
import cv2

def viz_cornerNcontour(img):
    # Tạo bản sao để vẽ hiển thị
    output_img = img.copy()


    # Vẽ tất cả các đường bao tìm được (màu xanh lá mỏng)
    cv2.drawContours(output_img, contours, -1, (0, 255, 0), 1)

    # Định nghĩa màu sắc và tên cho từng góc để dễ phân biệt
    # Thứ tự: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)] 
    
    for i, pt in enumerate(corners):
        # Chuyển tọa độ về số nguyên để vẽ
        point = tuple(pt.astype(int))
        
        # Vẽ vòng tròn tại góc
        cv2.circle(output_img, point, 10, colors[i], -1)
        
        # Viết số thứ tự góc (0, 1, 2, 3) để kiểm tra trình tự
        cv2.putText(output_img, str(i), (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)

    # 5. Hiển thị kết quả
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Corners & Order")
    plt.axis('off')    
    plt.tight_layout()
    plt.show()

def xyxy_to_xywh(xyxy):
        """Chuyển đổi tọa độ từ [xmin, ymin, xmax, ymax] sang [xmin, ymin, w, h]."""
        x1, y1, x2, y2 = xyxy
        return [x1, y1, x2 - x1, y2 - y1]

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Áp dụng Non-Maximum Suppression (NMS) tự động bằng OpenCV.
    boxes: Danh sách các boxes dạng [xmin, ymin, w, h] (dạng pixel absolute).
    scores: Danh sách các confidence scores tương ứng.
    iou_threshold: Ngưỡng IoU để quyết định chồng chéo (0.5 là mức trung bình).
    """
    # OpenCV yêu cầu boxes dạng [xmin, ymin, w, h] và scores dạng float
    # indices là danh sách các chỉ số của các boxes được giữ lại
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    return indices

def split(result, image):
    # --- CẤU HÌNH ---
    iou_nms_threshold = 0.45                         # Ngưỡng NMS (càng thấp càng ít box chồng)

    # 3. Trích xuất thông tin Bounding Boxes và Scores từ kết quả
    raw_boxes = result.boxes.xyxy.cpu().numpy()  # Lấy tọa độ dạng [xmin, ymin, xmax, ymax]
    scores = result.boxes.conf.cpu().numpy()     # Lấy confidence scores


    # Chuyển đổi boxes sang dạng [xmin, ymin, w, h] mà OpenCV NMS yêu cầu
    boxes_for_nms = [xyxy_to_xywh(box) for box in raw_boxes]

    # 4. Áp dụng NMS để loại bỏ các box chồng chéo
    # cv2.dnn.NMSBoxes trả về một danh sách các chỉ số (ví dụ: [0, 2, 5])
    keep_indices = apply_nms(boxes_for_nms, scores.tolist(), iou_nms_threshold)

    # 5. Duyệt qua các box được giữ lại, cắt và lưu ảnh
    if len(keep_indices) > 0:
        # Flatten danh sách indices nếu cần (tùy phiên bản OpenCV)
        keep_indices = keep_indices.flatten()

        crop_imgs = []

        for idx in keep_indices:
            # Lấy lại tọa độ xyxy gốc
            x1, y1, x2, y2 = raw_boxes[idx]
            
            # Đảm bảo tọa độ là số nguyên và nằm trong phạm vi ảnh
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
            
            # Kiểm tra kích thước box hợp lệ
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Cắt ảnh (Crop)
            crop_img = image[y1:y2, x1:x2]
            
            # Lưu ảnh cắt
            crop_imgs.append(crop_img)

        return crop_imgs