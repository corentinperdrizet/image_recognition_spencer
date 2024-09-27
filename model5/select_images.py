""""Description: Select images based on user preferences and bandwidth limit."""
# Analyze all images in the directory 'images_provider' 
# and select images based on user preferences and bandwidth limit (here fixed at 10 MB).
# The execution also displays the selected images with reconstituted RGB bands and the class probability that determined their selection.
#
# Default preferences: 'Complex cultivation patterns', 'Burnt areas', 'Port areas', 'Coastal lagoons', 
# 'Land principally occupied by agriculture, with significant areas of natural vegetation', 'Mixed forest', 
# 'Sclerophyllous vegetation', 'Mineral extraction sites', 'Water courses', 'Sparsely vegetated areas', 
# 'Dump sites', 'Industrial or commercial units', 'Annual crops associated with permanent crops', 
# 'Intertidal flats', 'Natural grassland', 'Water bodies', 'Continuous urban fabric', 'Rice fields', 
# 'Road and rail networks and associated land', 'Olive groves', 'Vineyards', 'Permanently irrigated land', 
# 'Transitional woodland/shrub', 'Pastures', 'Salines', 'Broad-leaved forest', 'Agro-forestry areas', 'Peatbogs', 
# 'Bare rock', 'Discontinuous urban fabric', 'Construction sites', 'Coniferous forest', 'Moors and heathland', 
# 'Non-irrigated arable land', 'Airports', 'Fruit trees and berry plantations', 'Sport and leisure facilities', 
# 'Inland marshes', 'Green urban areas', 'Sea and ocean', 'Salt marshes', 'Estuaries', 'Beaches, dunes, sands'
#
# Example usage:
# $ python3 select_images.py 'Inland marshes'

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
from config import model, mlb

# Default class preferences
default_class_labels = [
    'Complex cultivation patterns', 'Burnt areas', 'Port areas', 'Coastal lagoons',
    'Land principally occupied by agriculture, with significant areas of natural vegetation', 'Mixed forest',
    'Sclerophyllous vegetation', 'Mineral extraction sites', 'Water courses', 
    'Dump sites', 'Industrial or commercial units', 'Annual crops associated with permanent crops',
    'Intertidal flats', 'Natural grassland', 'Water bodies', 'Continuous urban fabric', 'Rice fields',
    'Road and rail networks and associated land', 'Olive groves', 'Vineyards', 'Permanently irrigated land',
    'Transitional woodland/shrub', 'Pastures', 'Salines', 'Broad-leaved forest',
    'Agro-forestry areas', 'Peatbogs', 'Bare rock', 'Discontinuous urban fabric', 'Construction sites', 'Coniferous forest',
    'Moors and heathland', 'Non-irrigated arable land', 'Airports', 'Fruit trees and berry plantations',
    'Sport and leisure facilities', 'Inland marshes', 'Green urban areas', 'Sea and ocean', 'Sparsely vegetated areas',
    'Salt marshes', 'Estuaries', 'Beaches, dunes, sands'
]

def load_rgb_image(folder_path):
    bands = ['B04', 'B03', 'B02']  # Red, Green, Blue bands
    images = []
    for band in bands:
        file_path = os.path.join(folder_path, f"{folder_path.split('/')[-1]}_{band}.tif")
        with Image.open(file_path) as img:
            images.append(np.array(img))

    image_rgb = np.dstack(images)
    image_rgb = image_rgb / image_rgb.max()  # Normalize the RGB image
    return image_rgb

def load_images_from_folder(folder_path):
    band_files = {'B01': None, 'B02': None, 'B03': None, 'B04': None, 'B05': None, 'B06': None,
                  'B07': None, 'B08': None, 'B09': None, 'B11': None, 'B12': None, 'B8A': None}
    
    for filename in os.listdir(folder_path):
        for band in band_files.keys():
            if band in filename:
                band_files[band] = load_image(os.path.join(folder_path, filename))
    return band_files

def load_image(file_path):
    with Image.open(file_path) as img:
        image_array = np.array(img)
        image_tensor = image_array.astype(np.float32) / 255.0  # Normalize
    return image_tensor

def prepare_inputs(images):
    img_20 = np.stack([images['B01'], images['B09']], axis=-1)
    img_60 = np.stack([images['B05'], images['B06'], images['B07'], images['B8A'], images['B11'], images['B12']], axis=-1)
    img_120 = np.stack([images['B02'], images['B03'], images['B04'], images['B08']], axis=-1)

    input_20 = np.expand_dims(img_20, axis=0)
    input_60 = np.expand_dims(img_60, axis=0)
    input_120 = np.expand_dims(img_120, axis=0)

    return [input_20, input_60, input_120]

def select_images(image_dir, model, mlb, class_preferences, bandwidth_limit=10):
    results = []
    total_size = 0.0

    for folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        images = load_images_from_folder(folder_path)
        inputs = prepare_inputs(images)
        
        predictions = model.predict(inputs)
        class_probabilities = {cls: predictions[0][i] for i, cls in enumerate(mlb.classes_)}

        highest_prob_class, highest_prob = None, 0

        for cls in class_preferences:
            prob = class_probabilities.get(cls, 0)
            if prob > 0.5:
                highest_prob_class, highest_prob = cls, prob
                break

        if highest_prob_class:
            file_size = sum([os.path.getsize(os.path.join(folder_path, f)) for f in os.listdir(folder_path)]) / (1024 * 1024)  # MB
            results.append({
                'folder_path': folder_path,
                'highest_prob_class': highest_prob_class,
                'highest_prob': highest_prob,
                'file_size': file_size
            })

    # Trier les images en fonction des priorités et probabilités décroissantes
    results = sorted(results, key=lambda x: (class_preferences.index(x['highest_prob_class']), -x['highest_prob']))

    selected_images = []
    total_size = 0.0

    for result in results:
        if total_size + result['file_size'] <= bandwidth_limit:
            selected_images.append(result)
            total_size += result['file_size']
        else:
            break
    
    return selected_images, total_size

def display_images(selected_images):
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(selected_images):
        rgb_image = load_rgb_image(result['folder_path'])
        plt.subplot(len(selected_images) // 5 + 1, 5, i + 1)
        plt.imshow(rgb_image)
        plt.title(f"{result['highest_prob_class']} ({result['highest_prob']:.2f})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select images based on class preferences and bandwidth limit.')
    parser.add_argument('preferences', metavar='P', type=str, nargs='*',
                        help='List of class preferences. Default: "Inland marshes", "Green urban areas", "Urban fabric"')
    args = parser.parse_args()

    user_preferences = args.preferences if args.preferences else default_class_labels
    full_preference_list = user_preferences + [cls for cls in default_class_labels if cls not in user_preferences]

    print("Current preferences order:", full_preference_list)

    image_dir = 'images_provider'
    bandwidth_limit = 10  # MB

    selected_images, total_bandwidth_used = select_images(image_dir, model, mlb, full_preference_list, bandwidth_limit)

    print("Images to send (sorted by class probabilities):")
    for result in selected_images:
        print(result['folder_path'].split('/')[-1])

    print(f"Total bandwidth used: {total_bandwidth_used:.2f} MB")

    display_images(selected_images)
