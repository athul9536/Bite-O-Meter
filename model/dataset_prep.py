import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation and rescaling
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Training data
train_gen = datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation data
val_gen = datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("Classes:", train_gen.class_indices)
print("Training images:", train_gen.samples)
print("Validation images:", val_gen.samples)
