# Import necessary libraries
import numpy as np
import tensorflow as tf
from src.model import build_model
from src.differential_privacy import apply_differential_privacy, compute_privacy_epsilon

# Load the preprocessed data
data = np.load('data/processed/mnist.npz')
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

# Build the model
input_shape = (28, 28, 1)
num_classes = 10
model = build_model(input_shape, num_classes)

# Apply differential privacy
learning_rate = 0.001
l2_norm_clip = 1.0
noise_multiplier = 1.1
num_microbatches = 32
model = apply_differential_privacy(model, 'sparse_categorical_crossentropy', learning_rate, l2_norm_clip, noise_multiplier, num_microbatches)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Compute the privacy epsilon
num_samples = x_train.shape[0]
batch_size = 32
epochs = 10
epsilon = compute_privacy_epsilon(num_samples, batch_size, noise_multiplier, epochs)
print(f'Privacy epsilon: {epsilon}')