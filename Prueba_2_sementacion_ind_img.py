#Codigo para seleccion de rectangulos contenedores para obtener las coordenadas de los rectangulos

import cv2
import numpy as np
import ctypes

# Function to get screen resolution
def get_screen_resolution():
    user32 = ctypes.windll.user32
    width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return width, height

# Global variables
drawing = False  # True if mouse is pressed
ix, iy = -1, -1
contours = []
objects = []
rectangles = []  # To store rectangle coordinates

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, rectangles
    global ix, iy, drawing, img, mask, contours

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            img_temp = img.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        mask = cv2.inRange(img, np.array([0, 255, 0]), np.array([0, 255, 0]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects.append(mask.copy())
        rectangles.append((ix, iy, x, y))  # Store rectangle coordinates

# Load image
img = cv2.imread(r"E:\Tesis\Para entrenamiento\Fotos_entrada\1_104_0410.JPG")
img_copy = img.copy()
mask = np.zeros_like(img)

# Get image dimensions
img_height, img_width = img.shape[:2]

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

# Draw contours on the original image
for contour in contours:
    cv2.drawContours(img_copy, [contour], -1, (0, 255, 0), 2)

# Save final contour and final image
final_contour = np.zeros_like(img)
for contour in contours:
    cv2.drawContours(final_contour, [contour], -1, (0, 255, 0), 2)

cv2.imwrite('final_contour.jpg', final_contour)
cv2.imwrite('final_image.jpg', img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Export rectangle coordinates to a text file
with open('rectangle_coordinates.txt', 'w') as file:
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        file.write(f"Top-left corner: ({x1}, {y1}), Bottom-right corner: ({x2}, {y2})\n")
