## Model Description
This model performs image segmentation using the U-Net architecture, a powerful model commonly employed in medical imaging for pixel-level classification. However, significant modifications were made to adapt the original U-Net model, which typically weighs several hundred megabytes, for satellite imagery tasks. In this case, the model is trained to recognize and segment various features in satellite images, including buildings, land, roads, vegetation, water, and unlabeled regions. The output is a detailed segmentation map, where each pixel is classified into one of these categories.

## Useful Information
- Model Size: 1.1 MB
- Dataset: The model was trained using a dataset containing satellite images of differents size and their corresponding masks, all coming from diverse landscape in Dubai.
The dataset can be downloaded by following this link : https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery
- Input: 512 x 512 pixels satellite images - RGB.

## Evaluation
- Precision: The model achieved an accuracy of 80\%, meaning that 80\% of the pixels are correctly classified into their respective categories.
- Comment : Although the model performs well overall, it faces challenges in accurately defining roads, particularly because the dataset contains images of Dubai, where many roads are made of limestone and appear quite white. This similarity in color and texture between the roads and surrounding landscape elements makes them harder to distinguish. However, the model remains effective for the other classes, such as buildings, vegetation, and water.

## Important Files

1. model.py: Defines the architecture of the model.
2. training.py: Used to train the model.
3. config.py: Allows you to choose the model for predictions and configure the paths for different directories.
4. predict.py: The simplest script to classify an image. Use it as follows:

`$ python3 predict.py 'path/to/an/image'`

-> Displays the original image, the predicted mask, and prints the pixel proportions of each class.

5. select_images.py: Description: Select images based on user preferences and bandwidth limit.
The script analyzes all images in the directory 'images_provider' and selects images 
based on the percentage of pixels corresponding to the preferred class. The default 
bandwidth limit is fixed at 10 MB. The execution also displays the selected images.

Default preference: Building

Example usage: `$ python3 select_images.py Water`

New preference: Water

## Directory Description

1. data/: Contains the dataset.
2. models/: Contains the saved models.
3. logs/: Contains logs for visualizing the model training with TensorBoard (optional and used only during training).
4. tst/: A folder where images for prediction testing are stored, and also contains test scripts.
5. images_provider/: The folder where images are placed for selection using select_images.py.
6. utils/: Contains useful scripts that assisted in the model development, such as:
   - A script to split the dataset into training and validation sets.
   - A script to resize all the dataset images to 512 x 512.
   - A script to convert the model from .keras to .h5.
   - Data preparation scripts.
7. evaluation/: This folder contains scripts useful for evaluating the model:
   - display_prediction.py: Display the prediction of the model on the validation set. Display the original image, the real mask and the predicted mask
