import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('fruits360_cnn_model.keras')

# Load and predict a new fruit image
img_path = 'cc.jpg'
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, 0) / 255.0  # Normalize

# Get prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

# Get class names (you need to save these during training)
# You can save class indices during training:
# import json
# with open('class_indices.json', 'w') as f:
#     json.dump(train_generator.class_indices, f)

print(f"Predicted class index: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
