import cv2
import numpy as np

# Global variables
drawing = False  # True if mouse is pressed
ix, iy = -1, -1
contours = []
objects = []

# Mouse callback function
def draw_contour(event, x, y, flags, param):
    global ix, iy, drawing, img, mask, contours

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        mask = cv2.inRange(img, np.array([0, 255, 0]), np.array([0, 255, 0]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects.append(mask.copy())

# Load image
img = cv2.imread(r"E:\Tesis\Para entrenamiento\Fotos_entrada\1_100_0020.JPG")
img_copy = img.copy()
mask = np.zeros_like(img)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_contour)

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
