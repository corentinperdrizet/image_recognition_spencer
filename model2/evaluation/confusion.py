""" Display the confusion matrix for a binary classification model. """

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import sys

# Add the project root to the path
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)
project_root = os.path.dirname(parent_directory)
sys.path.append(project_root)

from config import model, test_dir

# Prepare the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(512, 512),  # Ensure this size matches the expected input for your model
    batch_size=16,
    class_mode='binary',  # Use 'binary' for binary classification
    shuffle=False)

# Collect predictions and true classes
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)
true_classes = test_generator.classes

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Normalize the confusion matrix to display percentages
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Calculate the corrected accuracy
accuracy_corrected = np.trace(conf_matrix) / float(np.sum(conf_matrix))
print(f'Accuracy: {accuracy_corrected:.2f}')

# Display the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2%", cmap='Blues')
plt.title('Normalized Confusion Matrix\nAccuracy: {:.2f}%'.format(accuracy_corrected * 100))
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
