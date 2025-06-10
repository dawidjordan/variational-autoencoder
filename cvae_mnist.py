import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import time


latent_dim = 2
epochs = 10
batch_size = 32
epoch_losses = []


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
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


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(32, 3, strides=(2, 2), activation='relu'),
            layers.Conv2D(64, 3, strides=(2, 2), activation='relu'),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim),
        ])

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(units=7 * 7 * 32, activation='relu'),
            layers.Reshape(target_shape=(7, 7, 32)),
            layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(1, 3, strides=1, padding='same'),
        ])

    def encode(self, x):
        mean_logvar = self.encoder(x)
        z_mean, z_log_var = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def reparameterize(self, mean, logvar):
        return Sampling()((mean, logvar))

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits


model = CVAE(latent_dim=latent_dim)
optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_and_save_images(model, epoch, test_sample, output_dir="generated"):
    predictions = model.decode(test_sample, apply_sigmoid=True)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"image_at_epoch_{epoch:02d}.png"))
    plt.close()

def plot_latent_space(model, test_images, test_labels, output_file="latent_space.png"):
    z_means = []
    labels = []

    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i+batch_size]
        label_batch = test_labels[i:i+batch_size]
        batch = tf.convert_to_tensor(batch.reshape(-1, 28, 28, 1).astype("float32") / 255.0)
        mean, _ = model.encode(batch)
        z_means.append(mean.numpy())
        labels.append(label_batch)

    z_means = np.concatenate(z_means, axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_means[:, 0], z_means[:, 1], c=labels, cmap="tab10", s=3)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel("Z[0]")
    plt.ylabel("Z[1]")
    plt.title("Wizualizacja przestrzeni latentnej")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def generate_latent_images_grid(model, grid_size=20, output_file="latent_space_grid.png"):
    scale = 2.0
    figure = np.zeros((28 * grid_size, 28 * grid_size))

    grid_x = np.linspace(-scale, scale, grid_size)
    grid_y = np.linspace(-scale, scale, grid_size)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = tf.convert_to_tensor([[xi, yi]])
            x_decoded = model.decode(z, apply_sigmoid=True)
            digit = x_decoded.numpy().reshape(28, 28)
            figure[i * 28: (i + 1) * 28,
                   j * 28: (j + 1) * 28] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.axis('Off')
    plt.title("Wygenerowane obrazy z przestrzeni latentnej")
    plt.savefig(output_file)
    plt.close()

def plot_loss_over_epochs(losses, output_file="cvae_loss.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title("Zmiana wartości funkcji straty podczas treningu")
    plt.xlabel("Epoka")
    plt.ylabel("Funkcja straty")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def plot_log_variance_distribution(model, test_images, test_labels):
    logvars = []
    labels = []

    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i+batch_size]
        label_batch = test_labels[i:i+batch_size]
        batch = tf.convert_to_tensor(batch.reshape(-1, 28, 28, 1).astype("float32") / 255.0)
        _, logvar = model.encode(batch)
        logvars.append(logvar.numpy())
        labels.append(label_batch)

    logvars = np.concatenate(logvars, axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=(10, 6))
    for digit in range(10):
        idx = labels == digit
        plt.hist(logvars[idx, 0], bins=30, alpha=0.4, label=f"Cyfra {digit}")
    plt.title("Rozkład log wariancji przestrzeni latentnej według klasy")
    plt.xlabel("log(σ²)")
    plt.ylabel("Liczba próbek")
    plt.legend()
    plt.grid(True)
    plt.savefig("log_variance_distribution.png")
    plt.close()


random_vector_for_generation = tf.random.normal(shape=[16, latent_dim])

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()
    print(f"Epoch: {epoch}, Test set ELBO: {elbo:.2f}, time: {end_time - start_time:.2f}s")
    epoch_losses.append(-elbo)
    
    generate_and_save_images(model, epoch, random_vector_for_generation)

print("Trening zakończony.")

plot_latent_space(model, test_images, test_labels)
generate_latent_images_grid(model)
plot_loss_over_epochs(epoch_losses)
plot_log_variance_distribution(model, test_images, test_labels)

print("Przestrzen ukryta zapisana do 'latent_space.png'")
print("Siatka wygenerowanych cyfr zapisana do 'latent_space_grid.png'")
print("Wykres funkcji ELBO zapisany do 'cvae_loss.png'")
print("Wykres rozkładu log wariancji zapisany do 'log_variance_distribution.png'")