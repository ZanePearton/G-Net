import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

class GAN:
    def __init__(self, noise_dim):
        self.noise_dim = noise_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        # self.generator_optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.0002, beta_1=0.5)
        # self.discriminator_optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.0002, beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)

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
            tf.keras.layers.Conv2D(64, (3, 3), strides=(
                2, 2), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), strides=(
                2, 2), padding='same', activation='relu'),
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

def visualize_progress(epoch, generator, noise_dim, real_images, avg_disc_loss, avg_gen_loss):
    num_real_images = len(real_images)
    noise = tf.random.normal([num_real_images, noise_dim])  # Adjusted noise shape
    generated_images = generator(noise, training=False)

    fig, axes = plt.subplots(2, num_real_images, figsize=(16, 4))

    # Loop for real images
    for i in range(num_real_images):
        axes[0, i].imshow(real_images[i] * 0.5 + 0.5)
        axes[0, i].axis('off')
        axes[0, i].set_title('Real')
    
    # Loop for generated images
    for i in range(num_real_images):
        axes[1, i].imshow(generated_images[i] * 0.5 + 0.5)
        axes[1, i].axis('off')
        axes[1, i].set_title('Generated')

    plt.suptitle(f"Epoch: {epoch}, Avg Disc Loss: {avg_disc_loss:.4f}, Avg Gen Loss: {avg_gen_loss:.4f}")
    plt.tight_layout()

    save_dir = 'images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))

    if epoch % 5 == 0:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    noise_dim = 100
    batchsize = 5
    data = load_and_preprocess_data('dat/data_batch_2')
    dataset = tf.data.Dataset.from_tensor_slices(
        data).shuffle(len(data)).batch(batchsize, drop_remainder=True)

    gan = GAN(noise_dim)
    step_counter = 0
    epochs = 100
    for epoch in range(epochs):
        disc_loss_sum = 0  # Initialize disc_loss_sum at the start of each epoch
        gen_loss_sum = 0   # Initialize gen_loss_sum at the start of each epoch

        for batch in dataset:
            batch_size = batch.shape[0]
            noise = tf.random.normal([batch_size, noise_dim])

            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                generated_images = gan.generator(noise, training=True)
                real_output = gan.discriminator(batch, training=True)
                fake_output = gan.discriminator(generated_images, training=True)

                gen_loss = gan.generator_loss(fake_output)
                disc_loss = gan.discriminator_loss(real_output, fake_output)

                disc_loss_sum += disc_loss  # Accumulate the disc_loss
                gen_loss_sum += gen_loss    # Accumulate the gen_loss

            # Log to TensorBoard
            with gan.train_summary_writer.as_default():
                tf.summary.scalar('Generator Loss', gen_loss, step=step_counter)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=step_counter)

            step_counter += 1

            gradients_of_generator = gen_tape.gradient(
                gen_loss, gan.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, gan.discriminator.trainable_variables)

            gan.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, gan.generator.trainable_variables))
            gan.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, gan.discriminator.trainable_variables))

        # Compute the average losses for the epoch
        avg_disc_loss = disc_loss_sum / len(dataset)
        avg_gen_loss = gen_loss_sum / len(dataset)

        print(f"Epoch {epoch}/{epochs}, Avg Disc Loss: {avg_disc_loss.numpy()}, Avg Gen Loss: {avg_gen_loss.numpy()}")
        real_batch = next(iter(dataset.take(1)))[:8]
        visualize_progress(epoch, gan.generator, noise_dim, real_batch, avg_disc_loss, avg_gen_loss)