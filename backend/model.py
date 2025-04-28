import sys
from PIL import Image

def run_model(image_path):
    # Replace with your actual ML logic
    image = Image.open(image_path)
    return f"Image size: {image.size}"

if __name__ == "__main__":
    img_path = sys.argv[1]
    result = run_model(img_path)
    print(result)
