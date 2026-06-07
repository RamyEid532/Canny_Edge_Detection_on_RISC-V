import cv2
import numpy as np

# Load the image in grayscale mode
img = cv2.imread('my_photo.jpeg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: File not found! Make sure the filename is exactly 'my_photo.jpeg'")
else:
    # Resize the image to 512x512 to match the project alignment requirements
    img_resized = cv2.resize(img, (512, 512))

# Save the output as a raw binary grayscale file (Input for C++ code)
    img_resized.tofile('input.raw')
    print("Success! 'input.raw' file is ready for processing.")
