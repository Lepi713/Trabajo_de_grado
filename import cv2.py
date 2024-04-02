import cv2
import numpy as np
import os

# Function to process images
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Threshold the image to create a binary image
    _, thresholded = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black canvas for the mask
    mask = np.zeros_like(image)

    # Filter contours based on area
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 900:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    # To draw the filtered contours on the original image (optional)
    cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)

    # Display the mask with only the filtered contours
    #cv2.imshow('Filtered Contours Mask', mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Display the original image with filtered contours
    #cv2.imshow('Filtered Contours on Image', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

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
        