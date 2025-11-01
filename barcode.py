import cv2
from pyzbar import pyzbar
import numpy as np

def robust_decode_barcode(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try multiple preprocessing techniques
    methods = [
        ("original", gray),
        ("thresh", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)),
        ("clahe", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)),
    ]

    best = None
    for name, processed in methods:
        barcodes = pyzbar.decode(processed, symbols=[pyzbar.ZBarSymbol.EAN13])
        if barcodes:
            data = barcodes[0].data.decode()
            if len(data) == 13 and verify_ean13(data):
                best = data
                # Draw result
                (x, y, w, h) = barcodes[0].rect
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imshow(f"Best: {data} ({name})", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return best

    return None

def verify_ean13(code):
    if len(code) != 13 or not code.isdigit():
        return False
    digits = code[:12]
    check = int(code[12])
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits)))
    return check == (10 - total % 10) % 10

# Run
if __name__ == "__main__":
    result = robust_decode_barcode("barcode.png")
    if result:
        print(f"Correct EAN-13: {result}")
    else:
        print("Failed to decode reliably.")