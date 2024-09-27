""" Script to resize images to 512x512 pixels. """

import os
from PIL import Image, ImageOps

def resize_image(image_path):
    """
    Resize the image to a size of 512x512 pixels.

    Args:
        image_path (str): The path to the image file.

    Returns:
        None
    """
    with Image.open(image_path) as img:
        width, height = img.size
        
        # If one or both sides are smaller than 512, adjust them.
        if width < 512 or height < 512:
            # Calculation of the new widths necessary to reach at least 512 pixels
            new_width = 512 if width < 512 else width
            new_height = 512 if height < 512 else height

            # Filling the image to get the new dimensions
            img = ImageOps.expand(img, (
                (new_width - width) // 2,
                (new_height - height) // 2,
                (new_width - width + 1) // 2,
                (new_height - height + 1) // 2
            ), fill=img.getpixel((0, 0)))
        
        # Image resizing to 512x512
        img = img.resize((512, 512), Image.LANCZOS)
        img.save(image_path)

def process_directory(directory):
    """
    Process all the images in the given directory by resizing them.

    Args:
        directory (str): The path to the directory containing the images.

    Returns:
        None
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                resize_image(file_path)
                print(f'Processed {file_path}')

# Put the path of the directory containing the images here
directory_path = 'path/to/directory'
process_directory(directory_path)
