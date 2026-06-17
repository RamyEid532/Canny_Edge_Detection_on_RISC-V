import cv2
import numpy as np

img = cv2.imread('my_photo.jpeg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: File not found!")
else:
    img_resized = cv2.resize(img, (512, 512))
    img_resized.tofile('input.raw')
    print("Success! 'input.raw' file is ready for processing.")
    
    # المقارنة مع OpenCV
    edges = cv2.Canny(img_resized, 50, 150)
    cv2.imwrite('reference.png', edges)
    print("Success! 'reference.png' file is ready for comparison.")

    # تحويل output.raw لـ png بعد تشغيل الكود
    import os
    if os.path.exists('output.raw'):
        output = np.fromfile('output.raw', dtype=np.uint8).reshape(512, 512)
        cv2.imwrite('output.png', output)
        print("Success! 'output.png' file is ready for comparison.")
    else:
        print("Run the C++ code first to generate output.raw")
