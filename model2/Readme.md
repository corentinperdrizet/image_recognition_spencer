## Model Description
This model performs binary classification to detect the presence of boats in satellite images. It is specifically designed to classify whether an image contains a boat or no boat. The model processes each input image and outputs a probability for each of the two classes, assigning the image to the class with the highest probability.

## Useful Information
- Model Size: 496 KB
- Dataset: The model was trained using the MASATI dataset, which contains a variety of 512 x 512 pixels satellite images capturing coastal and maritime areas. This dataset includes images featuring both boats and no-boat scenarios, making it ideal for binary classification tasks. More informations about the dataset is publicly available at the following link: https://www.iuii.ua.es/datasets/masati/
- Input: 512 x 512 pixels satellite images - RGB.

## Evaluation
- Precision: The model achieves a classification precision of 88.64% on the validation set.

## Important Files

1. model.py: Defines the architecture of the model.
2. training.py: Used to train the model.
3. config.py: Allows you to choose the model for predictions and configure the paths for different directories.
4. predict.py: The simplest script to classify an image. Use it as follows:

`$ python3 predict.py 'path/to/an/image'`

`$ Boat probability: 1.00`


5. select_images.py: Select images based on the probability that they contain a boat and limited by the bandwidth limit (here fixed at 10mB). It also displays the selected images and their probability.

   Example Usage:
   `$ python3 select_images.py`


## Directory Description

1. data/: Contains the dataset.
2. models/: Contains the saved models.
3. logs/: Contains logs for visualizing the model training with TensorBoard (optional and used only during training).
4. tst/: A folder where images for prediction testing are stored, and also contains test scripts.
5. images_provider/: The folder where images are placed for selection using select_images.py.
6. utils/: Contains useful scripts that assisted in the model development, such as:
   - A script to convert the model from .keras to .h5.
   - Data preparation scripts.
7. evaluation/: This folder contains scripts useful for evaluating the model:
   - confusion.py: Displays the confusion matrix for the model.
     Execution:
      `$ python3 confusion.py`
   - prediction.py: Displays predictions :
     - Randomly take 10 boat images and 10 non-boat images then analyze them and display their probability of having a boat in it.
     
      `$ python3 prediction.py`
