## Model Description
This model performs image classification using a convolutional neural network (CNN). The task of this model is to take an input satellite image and classify it into one of the following 10 predefined land cover classes:

AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

In this classification task, the model processes each input image and associates it with the most likely class based on its features.

## Useful Information
- Model Size: 1.3 MB
- Dataset: The model was trained on the EuroSAT dataset, which is publicly available at the following link: https://github.com/phelber/EuroSAT. The dataset contains 27,000 satellite images from 10 different classes. Each image is 64x64 pixels, with a resolution of approximately 10 meters per pixel.
- Input: 64x64 pixels satellite images - RGB.

## Evaluation
- Precision: The model achieves a classification precision of 93.03% on the validation set.

## Important Files

1. model.py: Defines the architecture of the model.
2. training.py: Used to train the model.
3. config.py: Allows you to choose the model for predictions and configure the paths for different directories.
4. predict.py: The simplest script to classify an image. Use it as follows:

`$ python3 predict.py 'path/to/an/image'`

`$ Prediction: Forest`


5. select_images.py: Select images based on user preferences and bandwidth limit (here fixed at 10mB). It also displays the selected images.

   Default Preferences: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

   Example Usage:
   `$ python3 select_images.py Residential Highway Forest`

   New preferences: Residential Highway Forest, AnnualCrop, Herb.Vegetation, Industrial, Pasture, PermanentCrop, River, SeaLake

## Directory Description

1. data/: Contains the dataset.
2. models/: Contains the saved models.
3. logs/: Contains logs for visualizing the model training with TensorBoard (optional and used only during training).
4. tst/: A folder where images for prediction testing are stored, and also contains test scripts.
5. images_provider/: The folder where images are placed for selection using select_images.py.
6. utils/: Contains useful scripts that assisted in the model development, such as:
   - A script to split the dataset into training and validation sets.
   - A script to convert the model from .keras to .h5.
   - Data preparation scripts.
7. evaluation/: This folder contains scripts useful for evaluating the model:
   - confusion.py: Displays the confusion matrix for the model.
     Execution:
      `$ python3 confusion.py`
   - prediction.py: Displays predictions and can be used to:
     - Analyze a single image and visualize the probability of belonging to each class.
     - Analyze all images in a folder and display predictions in a grid.
     Execution: change the path in the file then :
     
      `$ python3 prediction.py`
