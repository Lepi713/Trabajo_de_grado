import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread(r"E:\Tesis\DCIM_1_agosto_2023\100FPLAN\DJI_0010.JPG")
image_copy = image.copy()  # Create a copy for drawing contours

# Convert BGR to HSV (Hue, Saturation, Value) color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for green color in HSV
lower_green = np.array([35, 50, 50])  # Adjust these values based on your green range
upper_green = np.array([90, 255, 255])

# Define lower and upper bounds for yellow color in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Create a mask for green areas in the image
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Create a mask for yellow areas in the image
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Combine green and yellow masks to create a final mask for non-green and non-yellow areas
mask_non_green_yellow = cv2.bitwise_or(mask_green, mask_yellow)

# Invert the mask (to get non-green and non-yellow areas)
mask_non_green_yellow_inv = cv2.bitwise_not(mask_non_green_yellow)

# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=mask_non_green_yellow_inv)

# Convert the result to grayscale
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use thresholding to create a binary image
_, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours within the area range of 300 to 500 pixels
filtered_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 5 < area < 900:
        filtered_contours.append(cnt)

# Draw contours on the original image
cv2.drawContours(image_copy, filtered_contours, -1, (0, 255, 0), 2)

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(141), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(142), plt.imshow(mask_green, cmap='gray'), plt.title('Mask (Green)')
plt.subplot(143), plt.imshow(mask_yellow, cmap='gray'), plt.title('Mask (Yellow)')
plt.subplot(144), plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)), plt.title('Image with Contours (Filtered)')
plt.show()
