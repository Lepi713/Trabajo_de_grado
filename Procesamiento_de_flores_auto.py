import os
import cv2
import numpy as np

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

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

    # Exclude black areas from the mask
    black_mask = cv2.inRange(image, (0, 0, 0), (0, 0, 0))
    mask_non_green_yellow_inv[black_mask == 255] = 0

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

    # Filter contours within the area range of 2 to 900 pixels
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2 < area < 900:
            filtered_contours.append(cnt)

    return filtered_contours, mask_non_green_yellow_inv

# Path to the folder containing images
folder_path = r"E:\Tesis\Para entrenamiento\Fotos_entrada\Usadas\Intento 2"

# Output folder for processed images
output_folder = r"E:\Tesis\Para entrenamiento\Salida\Salida automatico\Intento 2"

# Output folder for masks
mask_folder = r"E:\Tesis\Para entrenamiento\Salida\Salida automatico\Intento 2\Mascaras"

# Create the output and mask folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

# Dictionary to store the count of contours for each image
contour_count_dict = {}

# Process each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".JPG") or filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        contours, mask = process_image(image_path)

        # Create a blank mask for the filtered contours
        filtered_mask = np.zeros_like(mask)

        # Draw the filtered contours on the blank mask
        cv2.drawContours(filtered_mask, contours, -1, (255), thickness=cv2.FILLED)

        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cv2.drawContours(cv2.imread(image_path), contours, -1, (0, 255, 0), 2))

        # Save the filtered mask to the mask folder
        mask_path = os.path.join(mask_folder, filename)
        cv2.imwrite(mask_path, filtered_mask)

        # Store the count of contours for this image
        contour_count_dict[filename] = len(contours)

# Write contour count information to a text file
with open("contour_count.txt", "w") as file:
    for filename, count in contour_count_dict.items():
        file.write(f"{filename}: {count} contours\n")

print("All images processed. Contour count information saved to contour_count.txt.")
