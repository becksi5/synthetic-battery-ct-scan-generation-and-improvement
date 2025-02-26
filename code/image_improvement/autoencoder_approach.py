import numpy as np 
import pandas as pd 
import os
import re
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

hires_folder = r'fake_top'
lowres_folder = r'fake_top_blurred_21'

# Get list of images
hires_images = sorted([f for f in os.listdir(hires_folder) if f.endswith('.jpg')]) 
lowres_images = sorted([f for f in os.listdir(lowres_folder) if f.endswith('.jpg')])

# Create a DataFrame
data = pd.DataFrame({
    'low_res': [os.path.join(lowres_folder, img) for img in lowres_images],
    'high_res': [os.path.join(hires_folder, img) for img in hires_images]
})

# Define other variables
batch_size = 10
var_target_size = (760, 760)

# ImageDataGenerator for augmentation
image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

# Use flow_from_dataframe with directly created DataFrame
train_hiresimage_generator = image_datagen.flow_from_dataframe(
    data,
    x_col='high_res',
    target_size=var_target_size,
    color_mode='grayscale',
    class_mode=None,
    batch_size=batch_size,
    seed=42,
    subset='training'
)

train_lowresimage_generator = image_datagen.flow_from_dataframe(
    data,
    x_col='low_res',
    target_size=var_target_size,
    color_mode='grayscale',
    class_mode=None,
    batch_size=batch_size,
    seed=42,
    subset='training'
)

val_hiresimage_generator = image_datagen.flow_from_dataframe(
    data,
    x_col='high_res',
    target_size=var_target_size,
    color_mode='grayscale',
    class_mode=None,
    batch_size=batch_size,
    seed=42,
    subset='validation'
)

val_lowresimage_generator = image_datagen.flow_from_dataframe(
    data,
    x_col='low_res',
    target_size=var_target_size,
    color_mode='grayscale',
    class_mode=None,
    batch_size=batch_size,
    seed=42,
    subset='validation'
)

# Zip train and validation generators
train_generator = zip(train_lowresimage_generator, train_hiresimage_generator)
val_generator = zip(val_lowresimage_generator, val_hiresimage_generator)

def imageGenerator(train_generator):
    for (low_res, hi_res) in train_generator:
        yield (low_res, hi_res)


input_img = Input(shape=(*var_target_size, 1))

l1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
l2 = Conv2D(64, (3, 3), padding='same', activation='relu')(l1)
l3 = MaxPooling2D(padding='same')(l2)
l3 = Dropout(0.3)(l3)
l4 = Conv2D(128, (3, 3), padding='same', activation='relu')(l3)
l5 = Conv2D(128, (3, 3), padding='same', activation='relu')(l4)
l6 = MaxPooling2D(padding='same')(l5)
l7 = Conv2D(256, (3, 3), padding='same', activation='relu')(l6)

l8 = UpSampling2D()(l7)

l9 = Conv2D(128, (3, 3), padding='same', activation='relu')(l8)
l10 = Conv2D(128, (3, 3), padding='same', activation='relu')(l9)

l5_resized = Conv2D(128, (3, 3), padding='same', activation='relu')(l5)

l11 = add([l5_resized, l10])
l12 = UpSampling2D()(l11)
l13 = Conv2D(64, (3, 3), padding='same', activation='relu')(l12)
l14 = Conv2D(64, (3, 3), padding='same', activation='relu')(l13)

l2_resized = Conv2D(64, (3, 3), padding='same', activation='relu')(l2)

l15 = add([l14, l2_resized])

decoded = Conv2D(1, (3, 3), padding='same', activation='relu')(l15)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


train_samples = train_hiresimage_generator.samples
val_samples = val_hiresimage_generator.samples

train_img_gen = imageGenerator(train_generator)
val_image_gen = imageGenerator(val_generator)


checkpoint = ModelCheckpoint(
    r"C:\Users\becksi5\OneDrive - FORVIA\08_HELLA_Feature Detection_Transfer\05_Training Data\Zelle Oben\fake_oben_blurred_21\ml_train_data\autoencoder_top.h5",  # Filename includes epoch number
    monitor="val_loss",                  # Metric to monitor
    mode="min",                          # Mode can be 'min', 'max', or 'auto'
    save_best_only=True,                # Save the model at every epoch
    verbose=1
)

# Define other callbacks
earlystop = EarlyStopping(
    monitor='val_loss', 
    min_delta=0, 
    patience=9,
    verbose=1,
    restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=5, 
    verbose=1, 
    factor=0.2, 
    min_lr=1e-8
)

callbacks = [checkpoint, earlystop, learning_rate_reduction]

# Train the model
hist = autoencoder.fit(
    train_img_gen,
    steps_per_epoch=train_samples // batch_size,
    validation_data=val_image_gen,
    validation_steps=val_samples // batch_size,
    epochs=30,
    callbacks=callbacks  # Pass the list of callbacks
)
