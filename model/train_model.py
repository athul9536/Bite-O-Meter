import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Prepare dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
    subset='training' 
)

val_gen = datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)

# Step 2: Load MobileNetV2 as the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers so only custom layers are trained

# Step 3: Add custom layers on top
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification
model = Model(inputs=base_model.input, outputs=output)

# Step 4: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Step 6: Save the model
model.save('model/nail_biter_model.h5')

print("Model trained and saved as nail_biter_model.h5")
