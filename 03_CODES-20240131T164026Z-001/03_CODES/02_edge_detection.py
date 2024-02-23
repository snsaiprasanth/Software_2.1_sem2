import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input images
input_img_rgb = r"D:\masters\Iaac barcelona\AT barcelona\02 software -1_2\Software_2.1_sem2\03_CODES-20240131T164026Z-001\n_iaac_01.png"
input_img_ir = r"D:\masters\Iaac barcelona\AT barcelona\02 software -1_2\Software_2.1_sem2\03_CODES-20240131T164026Z-001\n_iaac_02.png"

img1 = cv2.imread(input_img_rgb,cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(input_img_ir, cv2.IMREAD_GRAYSCALE)

# Set a threshold value for binary conversion
thresh_value = 210

# Preprocess images (optional)
# img1 = cv2.GaussianBlur(img1, (5, 5), 0)

# Apply binary thresholding
_, binary_image = cv2.threshold(img1, thresh_value, 255, cv2.THRESH_BINARY)

# Apply Canny edge detection
edges1 = cv2.Canny(binary_image, 100, 200)
edges2 = cv2.Canny(img2, 100, 200)

# Display the original and edge-detected images side by side
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('Image 1')

plt.subplot(2, 2, 2)
plt.imshow(edges1, cmap='gray')
plt.title('Edges 1')

plt.subplot(2, 2, 3)
plt.imshow(img2, cmap='gray')
plt.title('Image 2')

plt.subplot(2, 2, 4)
plt.imshow(edges2, cmap='gray')
plt.title('Edges 2')

plt.show()

# Save the processed images
cv2.imwrite(r'output\edge_RGB.png',edges2)
cv2.imwrite(r'output\edge_IR.png',edges1)