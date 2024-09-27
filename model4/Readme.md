## Model Description
This model performs binary classification to predict whether a given satellite image shows an area that is susceptible to wildfires or not. In fact the model was trained on satellite images of areas that previously experienced wildfire. The model analyzes the features of the input satellite image to classify it into one of two categories:
- Susceptible to wildfire
- Not susceptible to wildfire

The model aims to assist in early detection and prevention efforts by identifying high-risk areas.

## Useful Information
- Model Size: 496 KB
- Dataset: The model was trained using the Wildfire Prediction Dataset, which consists of satellite images of areas in Canada that have previously experienced wildfires. More informations about the dataset is publicly available at the following link: https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset
- Input: 350 x 350 pixels satellite images -RGB.

## Evaluation
- Precision: The model achieves a classification precision of 98.51% on the validation set.

## Important Files

1. model.py: Defines the architecture of the model.
2. training.py: Used to train the model.
3. config.py: Allows you to choose the model for predictions and configure the paths for different directories.
4. predict.py: The simplest script to classify an image. Use it as follows:

`$ python3 predict.py 'path/to/an/image'`

`$ Area susceptible to wildfire probability: 1.00`


5. select_images.py: Select images based on the probability that they could experience a wildfire and limited by the bandwidth limit (here fixed at 10mB). It also displays the selected images and their probability.

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
     - Randomly take 10 wildfire images and 10 non-wildfire images then analyze them and display their probability of being exposed to a wildfire.
     
      `$ python3 prediction.py`
