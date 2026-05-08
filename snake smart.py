# =========================================================
# Snake Smart App
# Deep Learning Based Indian Snake Classification System
#
# Technologies Used:
# - TensorFlow / Keras
# - OpenCV
# - VGG16 Transfer Learning
# - ImageDataGenerator
#
# Author: Ramakrishnan 
# =========================================================


# =========================================================
# OPTIONAL: GOOGLE DRIVE SUPPORT (FOR GOOGLE COLAB)
# =========================================================

#USE_GOOGLE_DRIVE = False

##if USE_GOOGLE_DRIVE:
   # from google.colab import drive
   # drive.mount('/content/drive', force_remount=True)

    # Dataset path in Google Drive
    #DATASET_PATH = '/content/drive/MyDrive/major_project/Indian_Snakes'##

#else:
    # =====================================================
    # LOCAL PC DATASET PATH
    # =====================================================

   

DATASET_PATH = r'D:\snake\Indian_Snakes'
#""


# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import applications


# =========================================================
# LOAD DATASET FOLDERS
# =========================================================

train = list(os.walk(DATASET_PATH))

# List all snake categories
label_names = train[0][1]

# Create label dictionary
dict_labels = dict(zip(label_names, list(range(len(label_names)))))

print("Snake Classes:")
print(dict_labels)


# =========================================================
# DATASET PREPROCESSING FUNCTION
# =========================================================

def dataset(path):

    images = []
    labels = []

    for folder in tqdm(os.listdir(path)):

        value_of_label = dict_labels[folder]

        for file in os.listdir(os.path.join(path, folder)):

            path_of_file = os.path.join(path, folder, file)

            # Read image
            image = cv2.imread(path_of_file)

            # Skip corrupted images
            if image is None or image.size == 0:
                print(f"Failed to read image: {path_of_file}")
                continue

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize image
            image = cv2.resize(image, (150, 150))

            images.append(image)
            labels.append(value_of_label)

    # Normalize images
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)

    return images, labels


# =========================================================
# LOAD DATASET
# =========================================================

images, labels = dataset(DATASET_PATH)

# Shuffle dataset
images, labels = shuffle(images, labels)


# =========================================================
# DISPLAY SAMPLE IMAGES
# =========================================================

plt.figure(figsize=(10, 10))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.imshow(images[i])

    plt.xlabel(label_names[labels[i]])

plt.show()


# =========================================================
# DATA AUGMENTATION
# =========================================================

image_size = (224, 224)
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)


# =========================================================
# TRAINING DATASET
# =========================================================

train_ds = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)


# =========================================================
# VALIDATION DATASET
# =========================================================

val_ds = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)


# =========================================================
# DISPLAY AUGMENTED IMAGES
# =========================================================

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 15))

for i in range(5):

    image = next(train_ds)[0][0]

    image = np.squeeze(image)

    ax[i].imshow(image)

    ax[i].axis(False)

plt.show()


# =========================================================
# LOAD PRETRAINED VGG16 MODEL
# =========================================================

vgg_base = applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
vgg_base.trainable = False


# =========================================================
# BUILD CUSTOM MODEL
# =========================================================

inputs = Input(shape=(224, 224, 3))

x = vgg_base(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.5)(x)

# 16 snake classes
outputs = Dense(16, activation=None)(x)

vgg_model = Model(inputs, outputs)

vgg_model.summary()


# =========================================================
# COMPILE MODEL
# =========================================================

optimizer = keras.optimizers.Adam()

loss = keras.losses.CategoricalCrossentropy(from_logits=True)

metrics = [keras.metrics.CategoricalAccuracy()]

vgg_model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)


# =========================================================
# TRAIN MODEL
# =========================================================

epochs = 100

history = vgg_model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    verbose=1
)


# =========================================================
# SAVE TRAINED MODEL
# =========================================================

vgg_model.save('vgg.hdf5')

print("Model saved successfully.")