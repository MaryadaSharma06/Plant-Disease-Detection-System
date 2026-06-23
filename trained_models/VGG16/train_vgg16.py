"""
VGG16 Transfer Learning Script
Fine-tunes VGG16 model for plant disease detection
"""

# Import required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Set parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20  # You can reduce this to 5-10 for faster training
DATA_PATH = 'plantvillage dataset/color/'

# Create directories for saving models
os.makedirs('trained_models/VGG16/', exist_ok=True)

print("\n" + "="*60)
print("STEP 1: Loading and preprocessing data...")
print("="*60)

# Data preprocessing (VGG16 requires specific preprocessing)
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Get class names
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
print(f"\n✓ Number of classes: {num_classes}")
print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {validation_generator.samples}")

print("\n" + "="*60)
print("STEP 2: Building VGG16 transfer learning model...")
print("="*60)

# Load pre-trained VGG16 model without top layer
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all layers except last 4
for layer in base_model.layers[:-4]:
    layer.trainable = False

print(f"✓ VGG16 base model loaded")
print(f"✓ Trainable layers: {len([l for l in base_model.layers if l.trainable])}")

# Build complete model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Model built and compiled successfully")

print("\n" + "="*60)
print("STEP 3: Setting up callbacks...")
print("="*60)

# Callbacks for model training
checkpoint = keras.callbacks.ModelCheckpoint(
    'trained_models/VGG16/vgg16_epoch_{epoch:02d}_acc_{val_accuracy:.2f}.keras',
    save_best_only=False,
    monitor='val_accuracy',
    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

print("✓ Callbacks configured")

print("\n" + "="*60)
print("STEP 4: Training the model...")
print(f"Epochs: {EPOCHS}")
print("="*60)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

print("\n" + "="*60)
print("STEP 5: Saving the final model...")
print("="*60)

# Save the final model
model.save('trained_models/VGG16/vgg16_final_model.keras')
print("✓ Model saved: trained_models/VGG16/vgg16_final_model.keras")

# Save class names
import json
with open('trained_models/VGG16/class_names.json', 'w') as f:
    json.dump(class_names, f)
print("✓ Class names saved: trained_models/VGG16/class_names.json")

print("\n" + "="*60)
print("STEP 6: Generating training plots...")
print("="*60)

# Plot training history
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('VGG16 Model Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('VGG16 Model Loss')

plt.tight_layout()
plt.savefig('trained_models/VGG16/training_history.png')
print("✓ Training plots saved: trained_models/VGG16/training_history.png")

print("\n" + "="*60)
print("VGG16 MODEL TRAINING COMPLETE!")
print("="*60)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print("="*60)
