import tensorflow as tf
import os

# Directory where the images are located
DATA_DIR = '<dataset>'  # Replace 'path_to_directory' with the path to the parent directory

def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    img = img / 255.0  # normalize to [0,1]
    return img


# Get all image paths
all_image_paths = []
for dirpath, _, filenames in os.walk(DATA_DIR):
    for fname in filenames:
        if fname.endswith('.jpg'):
            all_image_paths.append(os.path.join(dirpath, fname))

# Quick check
print("First 5 image paths:", all_image_paths[:5])

# Quick check
print("First 5 image paths:", all_image_paths[:5])

# Create TensorFlow dataset
batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=len(all_image_paths))
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

# Verify dataset shape
for batch in dataset.take(1):
    print("Batch shape:", batch.shape)
