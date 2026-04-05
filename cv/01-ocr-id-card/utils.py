import matplotlib.pyplot as plt


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