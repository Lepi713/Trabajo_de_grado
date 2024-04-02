#Prueba utilizando filtrado por colores en las imagenes RGB, verde, morado, blanco y cafe

import cv2
import numpy as np
import os

# Function to process images
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Define lower and upper bounds for green color in RGB
    lower_green = np.array([0, 50, 0],   dtype="uint8")
    upper_green = np.array([50, 255, 50], dtype="uint8")

    # Define lower and upper bounds for white color in RGB
    lower_white = np.array([100, 100, 100], dtype="uint8")
    upper_white = np.array([255, 255, 255], dtype="uint8")

    # Define lower and upper bounds for purple color in RGB
    lower_purple = np.array([50, 0, 50],  dtype="uint8")
    upper_purple = np.array([255, 100, 255], dtype="uint8")

    # Define lower and upper bounds for brown color in RGB
    lower_brown = np.array([50, 60, 10],  dtype="uint8")
    upper_brown = np.array([255, 200, 100], dtype="uint8")

    # Mask out green, white, purple, and brown regions
    green_mask = cv2.inRange(image, lower_green, upper_green)
    white_mask = cv2.inRange(image, lower_white, upper_white)
    purple_mask = cv2.inRange(image, lower_purple, upper_purple)
    brown_mask = cv2.inRange(image, lower_brown, upper_brown)

    # Combine masks for regions to exclude
    exclude_mask = cv2.bitwise_or(brown_mask, purple_mask)

    # Apply Gaussian blur to combined mask to reduce noise
    exclude_mask = cv2.GaussianBlur(exclude_mask, (5, 5), 0)

    # Invert the exclude mask
    exclude_mask = cv2.bitwise_not(exclude_mask)

    # Combine the green mask and the inverted exclude mask
    green_mask = cv2.bitwise_and(green_mask, exclude_mask)

    # Combine green and white masks
    combined_mask = cv2.bitwise_or(green_mask, white_mask)

    # Apply Gaussian blur to combined mask to reduce noise
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # Threshold the mask to create a binary image
    _, thresholded = cv2.threshold(combined_mask, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black canvas for the mask
    mask = np.zeros_like(thresholded)

    # Filter contours based on area and exclude the ones surrounded by brown
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 500:  # Adjust the area threshold as needed
            # Check if contour is surrounded by brown
            x, y, w, h = cv2.boundingRect(contour)
            roi = brown_mask[y:y+h, x:x+w]
            if cv2.countNonZero(roi) == 0:
                filtered_contours.append(contour)
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    
    cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)
    # Display the mask with only the filtered contours
    # cv2.imshow('Filtered Contours Mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Display the original image with filtered contours
    # cv2.imshow('Filtered Contours on Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the modified image
    output_path_img = os.path.join(output_folder_img, os.path.basename(image_path))
    cv2.imwrite(output_path_img, image)
    output_path_mas = os.path.join(output_folder_mas, os.path.basename(image_path))
    cv2.imwrite(output_path_mas, mask)

# Folder containing images
input_folder = r"E:\Procesamiento de imagenes\Pruebas_ortofoto"
output_folder_mas = r"E:\Procesamiento de imagenes\Pruebas_ortofoto\Salida_mascaras"
output_folder_img = r"E:\Procesamiento de imagenes\Pruebas_ortofoto\Salida_imagenes"

# List all files in the input folder
image_files = os.listdir(input_folder)

# Process each image
for file in image_files:
    if file.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):  # Check if it's an image file
        image_path = os.path.join(input_folder, file)
        process_image(image_path)
