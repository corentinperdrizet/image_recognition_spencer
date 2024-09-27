# Description: This script is used to generate the 
# confusion matrix for the model based on the test dataset.

############################################
####          Confusion tab             ####
############################################

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)
project_root = os.path.dirname(parent_directory)
sys.path.append(project_root)

from config import train_dir, test_dir, model

# Define the data generators
test_datagen = ImageDataGenerator(rescale=1./255)

######################################################
####    Confusion matrix on Validation dataset    ####
######################################################

test_generator_test = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# Collect the predictions
test_predictions = model.predict(test_generator_test)
test_predicted_classes = np.argmax(test_predictions, axis=1)
test_true_classes = test_generator_test.classes
class_labels = list(test_generator_test.class_indices.keys())

# Calculate the accuracy
accuracy = np.sum(test_predicted_classes == test_true_classes) / len(test_predicted_classes)
print(f'Accuracy: {accuracy}')

# Calculate the confusion matrix
conf_matrix = confusion_matrix(test_true_classes, test_predicted_classes, normalize='true')  # Normalize the confusion matrix

# Show the confusion matrix
plt.figure(figsize=(15, 12))
ax = sns.heatmap(conf_matrix, annot=True, fmt=".2%", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Normalized Confusion Matrix\nAccuracy: {accuracy:.2%}')
plt.ylabel('Real Class')
plt.xlabel('Predicted Class')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()
