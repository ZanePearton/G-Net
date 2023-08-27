
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

class GAN:
    def __init__(self, noise_dim):
        self.noise_dim = noise_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        # TensorBoard setup
        self.log_dir = 'logs/'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)

    def build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.noise_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(3 * 32 * 32, activation='sigmoid'),
            tf.keras.layers.Reshape((32, 32, 3))
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)  # No activation function here
        ])
        return model

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

def load_and_preprocess_data(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].astype(np.float32) / 255.0
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data

def visualize_progress(epoch, generator, noise_dim):
    noise = tf.random.normal([16, noise_dim])
    generated_images = generator(noise, training=False)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i] * 0.5 + 0.5)
        ax.axis('off')
    plt.show()

if __name__ == "__main__":
    noise_dim = 100
    data = load_and_preprocess_data('dat/data_batch_2')
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(64, drop_remainder=True)

    gan = GAN(noise_dim)
    step_counter = 0
    epochs = 100
for epoch in range(epochs): 
    for batch in dataset:
        batch_size = batch.shape[0]
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_images = gan.generator(noise, training=True)
            real_output = gan.discriminator(batch, training=True)
            fake_output = gan.discriminator(generated_images, training=True)

            gen_loss = gan.generator_loss(fake_output)
            disc_loss = gan.discriminator_loss(real_output, fake_output)

        # Log to TensorBoard
        with gan.train_summary_writer.as_default():
            tf.summary.scalar('Generator Loss', gen_loss, step=step_counter)
            tf.summary.scalar('Discriminator Loss', disc_loss, step=step_counter)

        step_counter += 1

        gradients_of_generator = gen_tape.gradient(gen_loss, gan.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, gan.discriminator.trainable_variables)

        gan.generator_optimizer.apply_gradients(zip(gradients_of_generator, gan.generator.trainable_variables))
        gan.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, gan.discriminator.trainable_variables))

    print(f"Epoch {epoch}/{epochs}, Disc Loss: {disc_loss.numpy()}, Gen Loss: {gen_loss.numpy()}")
    visualize_progress(epoch, gan.generator, noise_dim)
