import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# Build the SRCNN model
def build_srcnn():
    model = Sequential()
    
    model.add(Conv2D(filters=128, kernel_size=(9, 9), activation='relu', padding='same', input_shape=(128, 128, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=1, kernel_size=(5, 5), activation='linear', padding='same'))
    
    
    return model

# Function to split images into patches (if needed)
def split_image_into_patches(image, patch_size=(128, 256)):
    patches = []
    h, w = image.shape[:2]
    ph, pw = patch_size
    
    # Ensure the image dimensions are divisible by the patch size
    for i in range(0, h, ph):
        for j in range(0, w, pw):
            patch = image[i:i+ph, j:j+pw]
            if patch.shape[:2] == patch_size:
                patches.append(patch)
    return np.array(patches)

# Load and preprocess the dataset
def load_dataset(good_images_path, bad_images_path, image_size=(2053, 2053), patch_size=(128, 128), max_images=100):
    good_patches = []
    bad_patches = []
    
    for i, filename in enumerate(os.listdir(good_images_path)):
        if i >= max_images:
            break
        if filename.endswith((".png", ".jpg", ".jpeg")):
            # Load and resize good (high-resolution) image
            good_img = load_img(os.path.join(good_images_path, filename), color_mode='grayscale', target_size=image_size)
            good_img = img_to_array(good_img) / 255.0  # Normalize image data
            good_patches.extend(split_image_into_patches(good_img, patch_size))
            
            # Load and resize bad (low-resolution) image
            bad_img = load_img(os.path.join(bad_images_path, filename), color_mode='grayscale', target_size=image_size)
            bad_img = img_to_array(bad_img) / 255.0  # Normalize image data

            #bad_img = upscale_image(bad_img, 2)

            bad_patches.extend(split_image_into_patches(bad_img, patch_size))

    
    good_patches = np.array(good_patches)
    bad_patches = np.array(bad_patches)
    
    return good_patches, bad_patches

# Load dataset paths
good_images, bad_images = load_dataset(r'C:\Users\becksi5\OneDrive - FORVIA\Paper\fake', r'C:\Users\becksi5\OneDrive - FORVIA\Paper\fake_blur_21_low_quality_resized\Keras_Bilinear')

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(bad_images, good_images, test_size=0.2, random_state=42)

# Build and compile the SRCNN model
srcnn = build_srcnn()
srcnn.compile(optimizer=Adam(learning_rate=1e-4), loss=MeanSquaredError(), metrics=['mean_squared_error'])

# Train the model
srcnn.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=20, verbose=1)

# Evaluate the model
loss, mse = srcnn.evaluate(X_test, y_test)
print(f'Test MSE: {mse}')
