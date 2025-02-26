from tensorflow.keras import layers, models
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Define directories
train_hq_dir = 'ml_train_data\train\hq'
train_lq_dir = 'ml_train_data\train\lq'
val_hq_dir = 'ml_train_data\val\hq'
val_lq_dir = 'ml_train_data\val\lq'


# Helper function to load and preprocess images (grayscale)
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)  # Grayscale (1 channel)
    image = tf.image.resize(image, [760, 760])  # Resize to target size
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image

# Function to load paired HQ and LQ images
def load_image_pair(lq_path, hq_path):
    lq_image = preprocess_image(lq_path)
    hq_image = preprocess_image(hq_path)
    return lq_image, hq_image

# Function to get file paths
def get_image_paths(lq_dir, hq_dir):
    lq_image_paths = [os.path.join(lq_dir, fname) for fname in os.listdir(lq_dir)]
    hq_image_paths = [os.path.join(hq_dir, fname) for fname in os.listdir(hq_dir)]
    return lq_image_paths, hq_image_paths

# Get file paths for training and validation sets
train_lq_paths, train_hq_paths = get_image_paths(train_lq_dir, train_hq_dir)
val_lq_paths, val_hq_paths = get_image_paths(val_lq_dir, val_hq_dir)

# Create TensorFlow Dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((train_lq_paths, train_hq_paths))
train_dataset = train_dataset.map(lambda lq, hq: load_image_pair(lq, hq))
train_dataset = train_dataset.batch(1)  # Adjust the batch size if necessary

# Create TensorFlow Dataset for validation
val_dataset = tf.data.Dataset.from_tensor_slices((val_lq_paths, val_hq_paths))
val_dataset = val_dataset.map(lambda lq, hq: load_image_pair(lq, hq))
val_dataset = val_dataset.batch(1)  # Adjust the batch size if necessary

def residual_block(x, filters):
    res = layers.Conv2D(filters, 3, padding='same')(x)
    res = layers.BatchNormalization()(res)
    res = layers.ReLU()(res)
    res = layers.Conv2D(filters, 3, padding='same')(res)
    res = layers.BatchNormalization()(res)
    res = layers.Add()([x, res])
    return res

def build_generator():
    inputs = layers.Input(shape=[760, 760, 1])
    x = layers.Conv2D(64, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Downsampling
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(9):  # Number of residual blocks
        x = residual_block(x, 256)

    # Upsampling
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2D(1, 7, padding='same', activation='tanh')(x)
    return models.Model(inputs=inputs, outputs=outputs)

# Discriminator: PatchGAN
def build_discriminator():
    inputs = layers.Input(shape=[760, 760, 1])  # Grayscale input (1 channel)

    x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    output = layers.Conv2D(1, 4, padding='same')(x)

    return models.Model(inputs=inputs, outputs=output)

# Initialize models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
generator_optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.5)

# Loss functions
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Generator Loss
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (100 * l1_loss)
    return total_gen_loss

# Discriminator Loss
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# Training step
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(input_image, training=True)

        disc_real_output = discriminator(target, training=True)
        disc_generated_output = discriminator(generated_image, training=True)

        gen_loss = generator_loss(disc_generated_output, generated_image, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Training loop
EPOCHS = 100 

for epoch in range(EPOCHS):
    print(f"Starting epoch {epoch+1}/{EPOCHS}")
    for step, (input_image, target) in enumerate(train_dataset):
        gen_loss, disc_loss = train_step(input_image, target)
        if step % 100 == 0:
            print(f"Epoch {epoch+1}, Step {step}: Gen Loss = {gen_loss.numpy()}, Disc Loss = {disc_loss.numpy()}")
    print(f"Epoch {epoch+1} completed: Gen Loss = {gen_loss.numpy()}, Disc Loss = {disc_loss.numpy()}")