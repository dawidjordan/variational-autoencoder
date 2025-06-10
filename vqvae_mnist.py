import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os


latent_dim = 64
num_embeddings = 128
commitment_cost = 0.25
epochs = 10
batch_size = 32
epoch_losses = []


(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0


train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(60000)
    .batch(batch_size)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_images)
    .shuffle(10000)
    .batch(batch_size)
)


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        initializer = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initializer(shape=(embedding_dim, num_embeddings)),
            trainable=True, name="embeddings"
        )

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        distances = (
            tf.reduce_sum(flat_inputs**2, axis=1, keepdims=True)
            - 2 * tf.matmul(flat_inputs, self.embeddings)
            + tf.reduce_sum(self.embeddings**2, axis=0, keepdims=True)
        )

        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + tf.stop_gradient(quantized - inputs)
        return quantized, loss


class VQVAE(tf.keras.Model):
    def __init__(self, latent_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(32, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2D(latent_dim, 1, strides=1, padding='same'),
        ])
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid'),
        ])

    def call(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.vq_layer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss


model = VQVAE(latent_dim, num_embeddings, commitment_cost)
optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        x_recon, vq_loss = model(x)
        recon_loss = tf.reduce_mean(tf.square(x - x_recon))
        loss = recon_loss + vq_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return recon_loss, vq_loss

def plot_loss_over_epochs(losses, output_file="vqvae_loss.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title("Zmiana wartości funkcji straty podczas treningu")
    plt.xlabel("Epoka")
    plt.ylabel("Funkcja straty")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


for epoch in range(1, epochs + 1):
    recon_loss_avg = tf.keras.metrics.Mean()
    vq_loss_avg = tf.keras.metrics.Mean()
    for batch in train_dataset:
        recon_loss, vq_loss = train_step(batch)
        recon_loss_avg(recon_loss)
        vq_loss_avg(vq_loss)
    epoch_losses.append(recon_loss_avg.result().numpy())
    print(f"Epoch {epoch}, Recon Loss: {recon_loss_avg.result():.4f}, VQ Loss: {vq_loss_avg.result():.4f}")


def generate_images(model, data, output_file="vqvae_recon.png"):
    data = data[:16]
    reconstructions, _ = model(data)
    reconstructions = reconstructions.numpy()

    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(reconstructions[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.savefig(output_file)
    plt.close()

generate_images(model, test_images)
plot_loss_over_epochs(epoch_losses)