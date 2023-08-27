import tensorflow as tf
import numpy as np
import pickle

def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

# Load a smaller subset of the CIFAR-10 dataset (e.g., 1000 samples)
subset_size = 100
data_batch_1 = load_cifar_batch('data/data_batch_1')
data = data_batch_1[b'data'][:subset_size]

# Normalize pixel values to [0, 1]
data = data.astype(np.float32) / 255.0

# Reshape data to [num_samples, height, width, channels]
data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Create TensorFlow dataset
batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.shuffle(buffer_size=len(data))
dataset = dataset.batch(batch_size, drop_remainder=True)

# Verify dataset shape
print("Dataset shape:", dataset.element_spec)