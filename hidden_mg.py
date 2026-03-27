import cv2
import numpy as np

# 1. Read the image (Grayscale)
img = cv2.imread('Corrupted_Lenna.png', 0)

if img is None:
    print("Error: Could not find 'Corrupted_Lenna.png'!")
else:
    # 2. Denoise (Max Filter / Dilation)
    # Using a 3x3 kernel to remove pepper noise
    kernel = np.ones((3,3), np.uint8)
    img_denoised = cv2.dilate(img, kernel)

    # 3. Gamma Correction (gamma = 0.4 to brighten)
    img_norm = img_denoised / 255.0
    img_gamma = np.uint8(np.power(img_norm, 0.4) * 255)

    # 4. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_final = clahe.apply(img_gamma)

    # 5. Edge Detection for the Password
    # Applying a slight blur helps clean remaining artifacts
    img_blur = cv2.GaussianBlur(img_final, (3, 3), 0)
    edges = cv2.Laplacian(img_blur, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))

    # Save and show results
    cv2.imshow('Restored Image', img_final)
    cv2.imshow('Edge Map - LOOK HERE FOR PASSWORD', edges)
    
    cv2.imwrite('restored.png', img_final)
    cv2.imwrite('edges_password.png', edges)
    
    print("Process complete. Check 'restored.png' and 'edges_password.png'.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()