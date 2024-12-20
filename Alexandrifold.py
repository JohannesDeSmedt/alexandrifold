import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.utils import image_dataset_from_directory

# Load your custom JPEG images from a directory
def load_custom_data(data_dir, img_size=(64, 64)):
    dataset = image_dataset_from_directory(
        data_dir,
        label_mode=None,
        image_size=img_size,
        batch_size=32,
    )
    for point in dataset:
        print(point.shape)

    dataset = dataset.map(lambda x: x / 255.0)  # Normalize to [0, 1]
    images = np.concatenate([x.numpy() for x in dataset], axis=0)
    return images

# Sampling layer for the latent space
class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

# VAE Model
def create_vae(input_shape, latent_dim):
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    print(encoder.summary())

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(input_shape[0] // 4 * input_shape[1] // 4 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((input_shape[0] // 4, input_shape[1] // 4, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(input_shape[2], 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    print(decoder.summary())

    # VAE
    vae_inputs = encoder_inputs
    z_mean, z_log_var, z = encoder(vae_inputs)
    vae_outputs = decoder(z)
    vae = Model(vae_inputs, vae_outputs, name="vae")

    # VAE Loss
    reconstruction_loss = tf.keras.losses.binary_crossentropy(vae_inputs, vae_outputs)
    reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=[1, 2])
    reconstruction_loss *= input_shape[0] * input_shape[1]
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer="adam")

    print("Reconstruction loss shape:", reconstruction_loss.shape)
    print("KL loss shape:", kl_loss.shape)
    return vae

def transform_images_with_vae(vae, new_images, blend_factor=0.2):
    """
    Transforms new images using a trained VAE to produce outputs
    that align with the training data distribution.

    Args:
        vae (tf.keras.Model): The trained VAE model.
        new_images (numpy.ndarray): Array of new input images, normalized to [0, 1].

    Returns:
        numpy.ndarray: Transformed images outputted by the VAE, scaled to [0, 255].
    """
    # Normalize the input images to match the training scale (if not already normalized)
    if new_images.max() > 1.0:
        new_images = new_images / 255.0

    # Get the encoder and decoder from the VAE model
    encoder = vae.get_layer("encoder")
    decoder = vae.get_layer("decoder")

    # Encode the new images into latent representations
    z_mean, z_log_var, z = encoder.predict(new_images)

    # Sample latent vector
    epsilon = tf.random.normal(tf.shape(z_mean))
    z_sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Blend latent space
    blended_z = blend_factor * z_sample + (1 - blend_factor) * z_mean


    # Decode the latent representations back into images
    transformed_images = decoder.predict(blended_z)

    # Scale the transformed images back to the original range [0, 255] for visualization or saving
    transformed_images = (transformed_images * 255).astype(np.uint8)

    return transformed_images



# Train and save results
def train_vae(data_dir, img_size=(350, 350), latent_dim=2, save_dir="vae_results"):
    # Prepare data
    x_train = load_custom_data(data_dir, img_size)
    print(x_train.shape)
    input_shape = x_train.shape[1:]

    # Create and train the VAE
    vae = create_vae(input_shape, latent_dim)
    vae.fit(x_train, epochs=2000, batch_size=1)

    # Generate and save new images
    os.makedirs(save_dir, exist_ok=True)
    decoder = vae.get_layer("decoder")

    for i in range(10):  # Generate 10 samples
        z_sample = np.random.normal(size=(1, latent_dim))  # Random latent vector
        generated_img = decoder.predict(z_sample)
        generated_img = np.squeeze(generated_img)  # Remove batch dimension
        generated_img = (generated_img * 255).astype(np.uint8)  # Convert to 8-bit format
        img = array_to_img(generated_img)
        img.save(os.path.join(save_dir, f"generated_image_{i + 1}.jpeg"))
    return vae

# if __name__ == "__main__":
#     # Set your data directory containing JPEG images
#     data_dir = "../crops/"
#     train_vae(data_dir)

if __name__ == "__main__":
    # Train your VAE
    data_dir = "../crops/"
    img_size = (300, 300)
    latent_dim = 4
    vae = train_vae(data_dir, img_size=img_size, latent_dim=latent_dim)

    # Load new images to transform
    new_data_dir = "../new_alexanders/"  # Directory containing new images
    new_images = load_custom_data(new_data_dir, img_size=img_size)  # Reuse your load_custom_data function

    # Transform the new images
    transformed_images = transform_images_with_vae(vae, new_images)

    # Save or display the transformed images
    save_dir = "vae_transformed_images"
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(transformed_images):
        array_to_img(img).save(os.path.join(save_dir, f"transformed_image_{i + 1}_ld_{latent_dim}.jpeg"))
