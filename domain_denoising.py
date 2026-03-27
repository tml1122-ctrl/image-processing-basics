# -*- coding: utf-8 -*-
"""
影像處理實驗：頻率域週期性雜訊移除與 Spectral Key 解碼
技術棧：NumPy, OpenCV, Matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_butterworth_notch_mask(shape, u0, v0, D0, n):
    """
    建立一對共軛對稱點的 Butterworth 陷波濾波器。
    (u0, v0) 是相對於頻譜中心的偏移座標。
    """
    M, N = shape
    H = np.ones((M, N), dtype=float)
    u = np.arange(M)
    v = np.arange(N)
    U, V = np.meshgrid(u, v, indexing='ij')
    mid_u, mid_v = M // 2, N // 2
    
    # 計算相對於中心點的絕對座標
    D1 = np.sqrt((U - (mid_u + u0))**2 + (V - (mid_v + v0))**2)
    D2 = np.sqrt((U - (mid_u - u0))**2 + (V - (mid_v - v0))**2)
    
    # 避免除以 0 導致錯誤
    D1[D1 == 0] = 1e-6
    D2[D2 == 0] = 1e-6
    
    # Butterworth 陷波濾波器公式
    H = 1 / (1 + ((D0**2) / (D1 * D2))**n)
    return H

# --- 主程式 ---

# 1. 讀取影像 (強制轉灰階)
img = cv2.imread('Spectral_Challenge.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    # 若無此圖片，請更換為你的測試圖檔路徑
    print("提示：請確保 'Spectral_Challenge.png' 存在於當前目錄。")
    # 這裡建立一個簡單的測試陣列避免程式崩潰
    img = np.zeros((512, 512), dtype=np.uint8)

M, N = img.shape

# 2. 進行傅立葉轉換 (FFT)
f = np.fft.fft2(img.astype(float))
fshift = np.fft.fftshift(f)

# --- Task A & B: 診斷與設計濾波器 ---
# 分析圖片，雜訊是非常規則的陣列。
# 利用頻譜對稱性處理
mid_u, mid_v = M // 2, N // 2
H_total = np.ones((M, N), dtype=float)

# 定義陷波濾波器參數
notch_D0 = 5.0  # 截止頻率 (頻寬)
notch_n = 2     # Order (階數)

print("正在掃描頻譜並建立 Butterworth 陷波濾波器陣列...")
scan_margin = 15  # 忽略中心點周圍 15 像素
count = 0

for u_offset in range(-mid_u + 1, mid_u):
    for v_offset in range(-mid_v + 1, mid_v):
        # 只處理上半平面，利用對稱性一次濾一對點
        if u_offset < 0:
            continue
            
        # 忽略中心 DC 點和邊緣
        if (abs(u_offset) < scan_margin and abs(v_offset) < scan_margin) or \
           (u_offset == 0 and v_offset == 0):
            continue
            
        # 讀取該點的強度
        val = np.abs(fshift[mid_u + u_offset, mid_v + v_offset])
        
        # 設定閾值找出強雜訊點
        if val > 1e6:
            H_notch = get_butterworth_notch_mask((M, N), u_offset, v_offset, notch_D0, notch_n)
            H_total *= H_notch
            count += 1

print(f"掃描完成。總共濾除了 {count} 對週期性雜訊點。")

# --- Task C: 尋找隱藏 'Spectral Key' ---
print("\n正在尋找 Spectral Key...")
possible_keys = []
for u_offset in range(-mid_u + 1, mid_u):
    for v_offset in range(-mid_v + 1, mid_v):
        if (abs(u_offset) < scan_margin and abs(v_offset) < scan_margin) or \
           (u_offset == 0 and v_offset == 0):
            continue
            
        complex_val = fshift[mid_u + u_offset, mid_v + v_offset]
        val = np.abs(complex_val)
        
        # 尋找強度特殊的點 (根據實驗數據調整範圍)
        if 5e6 < val < 2e7:
            # 檢查是否接近純實數 (虛部遠小於實部)
            if abs(complex_val.imag) < (abs(complex_val.real) * 0.01):
                possible_keys.append({
                    'coord': (mid_u + u_offset, mid_v + v_offset),
                    'offset': (u_offset, v_offset),
                    'val': complex_val
                })

if possible_keys:
    print(f"成功找到 Spectral Key! (找到 {len(possible_keys)} 對可能點)")
    key = possible_keys[0]
    print(f"頻譜坐標 (u, v): {key['coord']}")
    print(f"相對於中心的偏移: {key['offset']}")
    print(f"隱藏 Message (常數值): {key['val'].real:.0f}")
else:
    print("未能自動找到 Spectral Key。")

# 3. 套用濾波器並回到空間域
fshift_filtered = fshift * H_total
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_restored = np.abs(img_back)

# --- 4. 顯示結果 ---
magnitude_spectrum = np.log1p(np.abs(fshift))
magnitude_spectrum_filtered = np.log1p(np.abs(fshift_filtered))

plt.figure(figsize=(16, 12))
plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('1. Original Corrupted Image')
plt.subplot(232), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('2. Log Magnitude Spectrum')
plt.subplot(233), plt.imshow(H_total, cmap='gray'), plt.title(f'3. Butterworth Notch Mask (D0={notch_D0})')
plt.subplot(234), plt.imshow(magnitude_spectrum_filtered, cmap='gray'), plt.title('4. Filtered Log Spectrum')
plt.subplot(235), plt.imshow(img_restored, cmap='gray'), plt.title('5. Restored Image')
plt.subplot(236), plt.plot(np.arange(N), H_total[mid_u, :]), plt.title('6. Filter Horizontal Cut-section'), plt.grid()

plt.tight_layout()
plt.show()