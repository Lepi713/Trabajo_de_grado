from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import os
import pandas as pd

def load_image(image_path):
    """
    Load an image and convert it to a numpy array of grayscale values.
    """
    image = Image.open(image_path).convert('L')
    return np.array(image)

def binarize_image(image, threshold=127):
    """
    Binarize the image by thresholding.
    """
    return (image > threshold).astype(int)

def calculate_confusion_matrix(image1, image2):
    """
    Calculate the confusion matrix comparing pixel values of two images.
    """
    # Flatten the images to 1D arrays
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    
    # Calculate confusion matrix
    cm = confusion_matrix(image1_flat, image2_flat)
    
    # Extract TP, TN, FP, FN
    TN, FP, FN, TP = cm.ravel()
    
    return TP, TN, FP, FN

def calculate_matthews_corrcoef(image1, image2):
    """
    Calculate the Matthews correlation coefficient for two images.
    """
    # Flatten the images to 1D arrays
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    
    # Calculate Matthews correlation coefficient
    mcc = matthews_corrcoef(image1_flat, image2_flat)
    
    return mcc

# Paths to the folders containing images
folder_path1 = r"E:\Tesis\Para entrenamiento\Salida\Fotos_procesadas"
folder_path2 = r"E:\Tesis\Para entrenamiento\Salida\Salida automatico\Intento 2\Mascaras"

# Get the list of image files from both folders
image_files1 = sorted(os.listdir(folder_path1))
image_files2 = sorted(os.listdir(folder_path2))

# Ensure both folders have the same number of images
if len(image_files1) != len(image_files2):
    raise ValueError("Both folders must contain the same number of images")

# DataFrame to save the confusion matrix values
results_df = pd.DataFrame(columns=['Image1', 'Image2', 'TP', 'TN', 'FP', 'FN', 'MCC'])

# Process each pair of images
for img_file1, img_file2 in zip(image_files1, image_files2):
    image_path1 = os.path.join(folder_path1, img_file1)
    image_path2 = os.path.join(folder_path2, img_file2)
    
    # Load the images
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    
    # Ensure images are the same size
    if image1.shape != image2.shape:
        raise ValueError(f"Images {img_file1} and {img_file2} must have the same dimensions")
    
    # Binarize the images
    image1_bin = binarize_image(image1)
    image2_bin = binarize_image(image2)
    
    # Calculate the confusion matrix values
    TP, TN, FP, FN = calculate_confusion_matrix(image1_bin, image2_bin)
    
    # Calculate the Matthews correlation coefficient
    mcc = calculate_matthews_corrcoef(image1_bin, image2_bin)
    
    # Append the results to the DataFrame
    results_df = results_df._append({
        'Image1': img_file1,
        'Image2': img_file2,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'MCC': mcc
    }, ignore_index=True)
    
    print(f"Processed {img_file1} - {img_file2}: MCC = {mcc}")

# Save the results DataFrame to a CSV file
results_df.to_csv('confusion_matrix_values.csv', index=False)

print(f"Results saved to confusion_matrix_values.csv")
