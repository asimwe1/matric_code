# main.py - Data Matrix Decoder using pyzbar
import cv2
from pyzbar import pyzbar
from PIL import Image
import numpy as np

# Config
image_path = 'image.png'

# Step 1: Load and try direct decode
print("Loading image...")
try:
    # Use OpenCV for flexibility
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load {image_path}. Check the file path.")
        exit()
    
    # Decode directly
    decoded_objects = pyzbar.decode(img)
    
    if decoded_objects:
        for obj in decoded_objects:
            if obj.type == 'DATAMATRIX':
                print("SUCCESS! Decoded Data Matrix:", obj.data.decode('utf-8'))
                print("Type:", obj.type)
                print("Rect:", obj.rect)
                exit()  # Done!
            else:
                print("Found barcode but not Data Matrix:", obj.type)
    else:
        print("No barcode found. Trying preprocessing...")
        
        # Step 2: Preprocess for better detection (grayscale + threshold)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try decoding the thresholded image
        decoded_objects = pyzbar.decode(thresh)
        
        if decoded_objects:
            for obj in decoded_objects:
                if obj.type == 'DATAMATRIX':
                    print("SUCCESS (after preprocessing)! Decoded Data Matrix:", obj.data.decode('utf-8'))
                    print("Type:", obj.type)
                    print("Rect:", obj.rect)
                    exit()
                else:
                    print("Found barcode after preprocess but not Data Matrix:", obj.type)
        
        # Step 3: Try resizing if image is too small (your case)
        print("Image too small? Trying upscale...")
        height, width = gray.shape
        upscale_factor = 4  # 4x larger for tiny images
        resized = cv2.resize(gray, (width * upscale_factor, height * upscale_factor), interpolation=cv2.INTER_CUBIC)
        _, thresh_resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        decoded_objects = pyzbar.decode(thresh_resized)
        
        if decoded_objects:
            for obj in decoded_objects:
                if obj.type == 'DATAMATRIX':
                    print("SUCCESS (after upscaling)! Decoded Data Matrix:", obj.data.decode('utf-8'))
                    print("Type:", obj.type)
                    print("Rect:", obj.rect)
                    exit()
        
        print("FAILED: No valid Data Matrix detected.")
        print("Reasons for your image:")
        print("  - Too low resolution (~60x60 px; needs 100+ px)")
        print("  - Pixelated/noisy (no clear black/white squares)")
        print("  - Missing finder pattern (L-shaped border)")
        print("Try a clearer image or generate a test one (code below).")
        
except Exception as e:
    print(f"Error: {e}")