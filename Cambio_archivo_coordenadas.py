#Ajuste de las coordenadas para leer en el codigo de separacion de los rectangulos


def transform_coordinates(input_file, output_file):
    transformed_lines = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    coordinates = line.split('Top-left corner: (')[1].split('), Bottom-right corner: (')
                    x1, y1 = coordinates[0].split(', ')
                    x2, y2 = coordinates[1].replace(')', '').split(', ')
                    transformed_line = f"{x1}, {y1}, {x2}, {y2}"
                    transformed_lines.append(transformed_line + '\n')
                except IndexError:
                    print(f"Skipping line due to unexpected format: {line}")

    with open(output_file, 'w') as f:
        f.writelines(transformed_lines)

# Replace 'input.txt' and 'output.txt' with your file names
transform_coordinates(r"E:\Tesis\Para entrenamiento\Salida\Coordenadas\1_100_0020.txt", r"E:\Tesis\Para entrenamiento\Salida\Coordenadas\1_100_0020_1.txt")
