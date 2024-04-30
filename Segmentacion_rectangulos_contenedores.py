import cv2
import numpy as np

def apply_watershed(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
    image[markers == -1] = [255, 0, 0]  # Mark watershed boundaries

    return image

def apply_watershed_to_all_rectangles(input_file, image_file, output_file):
    # Read the input image
    input_image = cv2.imread(image_file)
    height, width, _ = input_image.shape

    # Create a black image with the same dimensions as the original image
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

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

                # Paste the processed region onto the black image at the corresponding coordinates
                black_image[y1:y2, x1:x2] = processed_region

    # Save the resulting image
    cv2.imwrite(output_file, black_image)

if __name__ == "__main__":
    input_file = r"E:\Tesis\Para entrenamiento\Salida\Coordenadas\1_100_0020_1.txt"
    image_file = r"E:\Tesis\Para entrenamiento\Fotos_entrada\1_100_0020.JPG"
    output_file = r"E:\Tesis\Para entrenamiento\Salida\image_with_rectangles_1.png"
    apply_watershed_to_all_rectangles(input_file, image_file, output_file)
