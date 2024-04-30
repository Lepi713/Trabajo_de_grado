#Separacion de los rectangulos contenedeores para la generacion de la mascara de la imagen para entrenamiento

from PIL import Image, ImageDraw

def extract_rectangles(input_file, output_file):
    # Create a black image
    width, height = 1600, 1300  # Adjust this as per your requirement
    image = Image.new("RGB", (width, height), color="black")
    draw = ImageDraw.Draw(image)

    # Open the input file containing rectangle coordinates
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Iterate over each line containing rectangle coordinates
    for line in lines:
        # Extract coordinates
        x1, y1, x2, y2 = map(int, line.strip().split(","))

        # Draw rectangle on the image
        draw.rectangle([x1, y1, x2, y2], outline="white")

    # Save the image with rectangles drawn
    image.save(output_file)
    print("Image with rectangles saved as", output_file)

if __name__ == "__main__":
    input_file = "rectangle_coordinates.txt"
    output_file = "image_with_rectangles.png"
    extract_rectangles(input_file, output_file)
