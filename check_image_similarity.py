import tensorflow as tf
import numpy as np

# Load images into a numpy array
images = np.array([tf.keras.preprocessing.image.load_img(path) for path in image_paths])

# Preprocess images for the VGG16 model
preprocessed_images = tf.keras.applications.vgg16.preprocess_input(images)

# Load the VGG16 model
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

# Get the output of the second last layer as the feature vector
features_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract features for all images
features = features_extractor.predict(preprocessed_images)

# Compute similarity between feature vectors using cosine similarity
similarity_matrix = np.dot(features, features.T) / np.outer(np.linalg.norm(features, axis=1), np.linalg.norm(features, axis=1))
