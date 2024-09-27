# Spencer Satellite Image Recognition Project

This repository is my contribution to the Spencer project, which contains five AI models designed for satellite image recognition. The models are built to be as lightweight as possible due to serverless computing requirements and bandwidth constraints for transmitting the models to the satellite. Each model addresses a different task in satellite image recognition:

- **Model 1**: Classification into 10 classes.
- **Model 2**: Binary classification (detection) of ships.
- **Model 3**: Semantic segmentation.
- **Model 4**: Binary classification of areas at risk of wildfires.
- **Model 5**: Multiclass classification.

Each model has its own README within its subfolder, providing more detailed information on how the specific model works. 

## Project Structure

Each model's subfolder is generally structured in a similar way. While each model has its own README to explain specific differences, they all contain two key scripts that work in the same way:


### `predict.py`
This script takes an image as input and performs a prediction. You can run it using:

```bash
$ python3 predict.py 'path/to/an/image'
```

### `select_images.py`
This script analyzes all images in the `images_provider` directory, makes predictions on each image, and selects which images to prioritize for transmission back to Earth. The selection is based on user-specified preferences and is constrained by a bandwidth limit (here set to 10MB). Run it with:

```bash
$ python3 select_images.py'
```
More details on changing preferences and the selection process are provided in each model's specific README.

## Models Download


The models do not need to be retrained. However, the `.h5` model files may not download properly from GitHub. You can download them from the following Google Drive link: [\[GOOGLE DRIVE LINK\]](https://drive.google.com/drive/u/0/folders/1A6bLzWu_pqFJtWKqWd_DxLXVpWTuw2qn).

I converted the models to .h5 format to retain only the weights, making them lighter and more practical for this project. However, I didn’t convert them back to TensorFlow Lite (.tflite) because, although the layers are compatible, issues remain—particularly with the handling of Batch Normalization. This layer significantly boosts the model’s performance, but it is not well supported by TensorFlow Lite.

## Image Input Size

Each model is designed to work with a specific input image size, which is detailed in the respective README of each model. However, all the scripts are built to automatically resize images if the input size doesn't match the model's expected dimensions. For instance, a model trained with 64x64 pixel images can still process images of different sizes, such as 600x850 pixels. However, for optimal performance, it is recommended to use images with resolutions similar to the dataset on which the model was trained, as the size and pixel-to-meter ratio can affect accuracy. Be sure to explore the dataset specifications for each model to understand the best resolution to use.