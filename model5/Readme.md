## Model Description
This model performs multilabel classification using satellite images composed of 12 spectral bands. The input images belong to multiple classes simultaneously, chosen from a set of 43 predefined land cover categories. The task of this model is to predict which classes the input image is most likely to belong to.

The 43 classes are:

'Complex cultivation patterns', 'Burnt areas', 'Port areas', 'Coastal lagoons', 'Land principally occupied by agriculture, with significant areas of natural vegetation', 'Mixed forest', 'Sclerophyllous vegetation', 'Mineral extraction sites', 'Water courses', 'Sparsely vegetated areas', 'Dump sites', 'Industrial or commercial units', 'Annual crops associated with permanent crops', 'Intertidal flats', 'Natural grassland', 'Water bodies', 'Continuous urban fabric', 'Rice fields', 'Road and rail networks and associated land', 'Olive groves', 'Vineyards', 'Permanently irrigated land', 'Transitional woodland/shrub', 'Pastures', 'Salines', 'Broad-leaved forest', 'Agro-forestry areas', 'Peatbogs', 'Bare rock', 'Discontinuous urban fabric', 'Construction sites', 'Coniferous forest', 'Moors and heathland', 'Non-irrigated arable land', 'Airports', 'Fruit trees and berry plantations', 'Sport and leisure facilities', 'Inland marshes', 'Green urban areas', 'Sea and ocean', 'Salt marshes', 'Estuaries', 'Beaches, dunes, sands'

The model must handle multiple outputs since each image can belong to several of these classes at the same time.

## Useful Information
- Model Size: 2.9 MB
- Dataset: The model was trained on the \textbf{BigEarth} dataset, which consists of satellite images with 12 bands representing various spectral data. The bands are structured as follows:
    - 2 bands of size 20x20 pixels (B01, B09)
    - 6 bands of size 60x60 pixels (B05, B06, B07, B8A, B11, B12)
    - 4 bands of size 120x120 pixels (B02, B03, B04, B08)

These bands provide different resolutions and spectral information, 
making the dataset highly versatile for multilabel classification tasks.

- Input:
    - Input 1: shape=(20, 20, 2) - 2 channels for B01, B09
    - Input 2: shape=(60, 60, 6) - 6 channels for B05, B06, B07, B8A, B11, B12
    - Input 3: shape=(120, 120, 4) - 4 channels for B02, B03, B04, B08

The input must therefore be a folder containing all the corresponding bands, each band must end with its number and must be a .tif, for example the image example_image is a folder containing all these files: 
- example_image_B01.tif
- example_image_B02.tif
- example_image_B03.tif
- example_image_B04.tif
- example_image_B05.tif
- example_image_B06.tif
- example_image_B07.tif
- example_image_B08.tif
- example_image_B09.tif
- example_image_B11.tif
- example_image_B12.tif
- example_image_B8A.tif

Be careful there is no B10 but there is a B8A

The model uses three separate neural networks for each input size and concatenates the results to make the final prediction.




## Evaluation

Accuracy : The model achieved an accuracy of 45%, which means that in multilabel classification tasks, it correctly predicts a subset of the true classes in less than half of the cases on average. This includes correctly identifying both the presence and absence of certain classes (true positives and true negatives), but the relatively low score indicates the model’s difficulty in consistently predicting all relevant classes.

Handling the Dataset: The dataset consists of 12 image bands, each at three different resolutions. To manage this, three separate neural networks are required to process each resolution independently. These networks are then concatenated into a final combined network. While this architecture is necessary for handling the varying band resolutions, it effectively triples the size of the model. To compensate for this, the size of each individual network must be reduced to keep the model lightweight. However, reducing the size of these smaller networks also compromises their performance, leading to a loss in efficiency and a further decline in classification accuracy.

Challenges of Lightweight Models: Lightweight models are particularly ill-suited for multilabel classification, especially when dealing with a high number of classes like the 43 in this dataset, combined with the challenge of processing image bands of different size. Multiclass classification with a large number of categories and diverse input data like this typically requires more robust, larger models for better performance, making the use of a lightweight architecture a significant limitation in this context.

## Important Files

1. model.py: Defines the architecture of the model.
2. training.py: Used to train the model.
3. config.py: Allows you to choose the model for predictions and configure the paths for different directories.
4. predict.py: The simplest script to classify an image. Use it as follows:

`$ python3 predict.py 'path/to/an/image'`

`$ Predictions for image: Sparsely vegetated areas, Transitional woodland/shrub, Sea and ocean`


5. select_images.py: This script selects images based on user-defined class preferences and a fixed bandwidth limit of 10MB. The images are analyzed to determine which class they most likely belong to, based on model predictions. Images that match the highest priority classes with a probability greater than 0.5 are selected first, and the process continues until the bandwidth limit is reached. The RGB bands of the selected images are displayed along with the class probabilities that determined their selection.

Default Preferences: 'Complex cultivation patterns', 'Burnt areas', 'Port areas', 'Coastal lagoons',
'Land principally occupied by agriculture, with significant areas of natural vegetation', 'Mixed forest',
'Sclerophyllous vegetation', 'Mineral extraction sites', 'Water courses', 'Sparsely vegetated areas',
'Dump sites', 'Industrial or commercial units', 'Annual crops associated with permanent crops',
'Intertidal flats', 'Natural grassland', 'Water bodies', 'Continuous urban fabric', 'Rice fields',
'Road and rail networks and associated land', 'Olive groves', 'Vineyards', 'Permanently irrigated land',
'Transitional woodland/shrub', 'Pastures', 'Salines', 'Broad-leaved forest', 'Agro-forestry areas', 'Peatbogs',
'Bare rock', 'Discontinuous urban fabric', 'Construction sites', 'Coniferous forest', 'Moors and heathland',
'Non-irrigated arable land', 'Airports', 'Fruit trees and berry plantations', 'Sport and leisure facilities',
'Inland marshes', 'Green urban areas', 'Sea and ocean', 'Salt marshes', 'Estuaries', 'Beaches, dunes, sands'

Example Usage:

`$ python3 select_images.py 'Inland marshes' 'Beaches, dunes, sands'`

In this example, the script prioritizes images that belong to 'Inland marshes', followed by 'Beaches, dunes, sands', selecting those that have a probability higher than 0.5. If no images exceed the threshold for a given class, it moves to the next preferred class in the list. The selected images are displayed, and the list of images is printed in the terminal. The bandwidth used by the selected images is also displayed.

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
   - display_predictions.py: Loads the images from a directory, processes them using the trained model, and displays both the reconstituted RGB image and the predicted probabilities for each class. The script also marks the true classes in red for easy comparison.
     
     
     Execution:
      - Change the path to the image directory, this image directory should be a directory that contains images and each images is also a directory organized by bands (e.g., `B01`, `B02`, etc.).
      - Run the script:
      
      `$ python3 display_predictions.py`
