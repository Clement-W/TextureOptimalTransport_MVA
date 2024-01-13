from PIL import Image
import os

def get_image_shapes(folder_path):
    image_shapes = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                image_shapes[filename] = (width, height)

    return image_shapes

folder_path = "TextureOptimalTransport_MVA/tex"
image_shapes = get_image_shapes(folder_path)

for filename, shape in image_shapes.items():
    print(f"{filename}: {shape}")