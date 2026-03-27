import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_bilinear_resize(image, scale_factor):
    # 取得原始尺寸
    h, w = image.shape
    # 計算新尺寸
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    # 預分配空間 (使用 float32 避免計算過程溢位)
    output = np.zeros((new_h, new_w), dtype=np.float32)

    print(f"正在處理影像... 目標尺寸: {new_w}x{new_h}")

    for i in range(new_h):
        for j in range(new_w):
            # 1. 映射回原圖座標 (反向映射)
            src_x = i / scale_factor
            src_y = j / scale_factor

            # 2. 找到鄰近的 4 個像素點座標
            x1 = int(np.floor(src_x))
            y1 = int(np.floor(src_y))
            x2 = min(x1 + 1, h - 1)  # 防止越界
            y2 = min(y1 + 1, w - 1)

            # 3. 取得這 4 個點的像素值
            v11 = float(image[x1, y1])
            v21 = float(image[x2, y1])
            v12 = float(image[x1, y2])
            v22 = float(image[x2, y2])

            # 4. 求解多項式係數 (a, b, c, d) 
            # 根據公式 v(x,y) = ax + by + cxy + d 在局部單位格 [0,1]x[0,1] 的解：
            # 令局部位移 dx = x - x1, dy = y - y1
            dx = src_x - x1
            dy = src_y - y1

            # 係數矩陣解法 (Eq. 2-15 的直接變量應用)
            d = v11
            a = v21 - v11
            b = v12 - v11
            c = v11 - v21 - v12 + v22

            # 5. 套用公式計算目標像素值
            val = a * dx + b * dy + c * dx * dy + d
            output[i, j] = np.clip(val, 0, 255) # 確保數值在合理範圍

    return output.astype(np.uint8)

# --- 主程式執行區 ---

# 1. 讀取影像
img_path = 'lenna.jpg' 
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("找不到 Lenna 圖片，請確認檔案路徑！")
else:
    # 執行 Task 1: 放大 1.5 倍 
    scale = 1.5
    result = my_bilinear_resize(img, scale)

    # 執行 Task 2 預處理: 二值化 (Threshold = 128)
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # 2. 結果視覺化
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original (Grayscale)")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title(f"Bilinear Upscaled (x{scale})")
    plt.imshow(result, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Task 2: Binary Lenna (T=128)")
    plt.imshow(binary_img, cmap='gray')

    plt.tight_layout()
    plt.show()

    # 存檔供作業繳交
    cv2.imwrite('lenna_upscaled.png', result)
    cv2.imwrite('lenna_binary.png', binary_img)
    print("存檔完成：lenna_upscaled.png, lenna_binary.png")
    
    
    
# Task 2: 計算 D8 距離 
p = (200, 200)
q = (250, 250)
d8_dist = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
print(f"Task 2 - Chessboard Distance (D8): {d8_dist}")

# Task 3: 模擬畫出等優先曲線 (這是一個示意圖，符合理論趨勢)
N = [32, 128, 512]
# 對於高細節影像，曲線通常較平緩
k_line1 = [8, 7, 6] # 第一條等品質線
k_line2 = [6, 5, 4] # 第二條等品質線

plt.figure()
plt.plot(N, k_line1, 'o-', label='Quality 10')
plt.plot(N, k_line2, 's-', label='Quality 7')
plt.xlabel('Resolution (N)')
plt.ylabel('Bit Depth (k)')
plt.title('Isopreference Curves (High Detail - Lenna)')
plt.legend()
plt.show()