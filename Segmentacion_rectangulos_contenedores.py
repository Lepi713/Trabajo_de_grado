"""
Se esta segmentando la imagen por rectangulos contenedores, posteriormente se le aplica wateshed y un filtro de color verde para dejar solo las flores dentro de 
los rectangulos, por ultimo se esta invirtiendo la imagen para que las regiones de interes sean de color blanco y el resto negro, generando la mascara de solo las flores

"""

import cv2
import numpy as np

def apply_watershed(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect green regions in the image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Invert the green mask
    mask_green = cv2.bitwise_not(mask_green)

    # Combine the green mask with the grayscale image
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_green)

    # Apply thresholding
    ret, thresh = cv2.threshold(masked_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed algorithm
    markers = cv2.watershed(image, markers)

    # Create mask
    mask = np.zeros_like(gray, dtype=np.uint8)

    # Mark watershed boundaries in the mask
    mask[markers == -1] = 255
    
    # Find contours within the watershed region
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours in the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    return mask


def apply_watershed_to_all_rectangles(input_file, image_file):
    # Read the input image
    input_image = cv2.imread(image_file)
    height, width, _ = input_image.shape

    # Create a mask with the same dimensions as the original image
    mask = np.ones((height, width), dtype=np.uint8) * 255  # Initialize with white pixels

    # Read coordinates from the input file and process each region
    with open(input_file, "r") as file:
        for line in file:
            x1, y1, x2, y2 = map(int, line.strip().split(","))
            
            # Ensure x1 < x2 and y1 < y2
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            # Extract region of interest from the input image
            region = input_image[y1:y2, x1:x2]

            # Check if the extracted region is not empty (i.e., width and height are not zero)
            if region.shape[0] != 0 and region.shape[1] != 0:
                # Apply Watershed algorithm to the region
                processed_region = apply_watershed(region)

                # Update the corresponding region in the mask
                mask[y1:y2, x1:x2] = processed_region

    return mask

def invert_colors(image):
    # Invert colors
    inverted_image = cv2.bitwise_not(image)

    return inverted_image

if __name__ == "__main__":
    input_file = r"E:\Tesis\Para entrenamiento\Salida\Coordenadas\1_100_0020_1.txt"
    image_file = r"E:\Tesis\Para entrenamiento\Fotos_entrada\1_100_0020.JPG"
    output_file = r"E:\Tesis\Para entrenamiento\Salida\image_with_rectangles_2.png"
    mask = apply_watershed_to_all_rectangles(input_file, image_file)

    inverted_mask = invert_colors(mask)

    # Save the processed image
    cv2.imwrite(output_file, inverted_mask)

    # Display the mask
    cv2.imshow("Mask", inverted_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
