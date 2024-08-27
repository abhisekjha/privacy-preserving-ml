import numpy as np

# Parameters
num_samples = 100
image_shape = (28, 28)
num_classes = 10

# Generate random images
x_data = np.random.rand(num_samples, *image_shape).astype(np.float32)

# Generate random labels
y_data = np.random.randint(0, num_classes, num_samples)

# Save to compressed file
np.savez_compressed('data/processed/sample_mnist', x_data=x_data, y_data=y_data)
