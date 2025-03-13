from PIL import Image
import sys


def resize_image(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            resized_img = img.resize((512, 512))
            resized_img.save(output_path)
            print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    input_path = "backend/knee_mri_image.jpg"
    output_path = "backend/resized_image.jpg"
    resize_image(input_path, output_path)
