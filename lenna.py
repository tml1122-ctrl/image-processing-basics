import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取原始 Lenna 影像
img_original = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)

if img_original is None:
    print("錯誤：找不到影像檔案！")
else:
    # 取得原始影像的長與寬
    height, width = img_original.shape
    
    # 2. 計算新影像的尺寸 (1/2 寬高)
    new_height = height // 2
    new_width = width // 2
    
    # 3. 預先分配輸出影像的空間 (使用 numpy.zeros)
    # 這是 Python 處理影像的標準做法，確保記憶體效能
    img_shrunken = np.zeros((new_height, new_width), dtype=np.uint8)
    
    # 4. 實作 Row-Column Deletion (Decimation)
    # 使用巢狀迴圈 (Nested for loops) 手動映射像素
    # 根據公式：J(i, j) = I(i * 2, j * 2)
    for i in range(new_height):
        for j in range(new_width):
            # 從原圖跳格抓取像素 (取樣步長為 2)
            img_shrunken[i, j] = img_original[i * 2, j * 2]
            
    # 5. 顯示結果並存檔
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original ({width}x{height})")
    plt.imshow(img_original, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Shrunken ({new_width}x{new_height})")
    plt.imshow(img_shrunken, cmap='gray')
    
    plt.show()
    cv2.imwrite('lenna_shrunken.png', img_shrunken)
    print("影像處理完成並已存檔。")