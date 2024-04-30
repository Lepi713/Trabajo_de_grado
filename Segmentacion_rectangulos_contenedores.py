#Separacion de los rectangulos contenedeores para la generacion de la mascara de la imagen para entrenamiento

from PIL import Image

def extract_rectangles(input_file, image_file, output_file):
    # Open the input image
    input_image = Image.open(image_file)

    # Create a black image
    width, height = input_image.size
    output_image = Image.new("RGB", (width, height), color="black")

    # Iterate over each rectangle in the input file
    with open(input_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, line.strip().split(","))

            # Ensure x1 < x2 and y1 < y2
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            
            # Extract pixels from the input image
            region = input_image.crop((x1, y1, x2, y2))
            
            # Paste the pixels onto the output image
            output_image.paste(region, (x1, y1))

    # Save the image with rectangles drawn
    output_image.save(output_file)
    print("Image with extracted rectangles saved as", output_file)

if __name__ == "__main__":
    input_file = r"E:\Tesis\Para entrenamiento\Salida\Coordenadas\1_100_0020_1.txt"
    image_file = r"E:\Tesis\Para entrenamiento\Fotos_entrada\1_100_0020.JPG"
    output_file = r"E:\Tesis\Para entrenamiento\Salida\image_with_rectangles.png"
    extract_rectangles(input_file, image_file, output_file)

