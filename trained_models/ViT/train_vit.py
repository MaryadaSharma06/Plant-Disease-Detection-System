"""
Vision Transformer (ViT) Training Script
Trains a Vision Transformer model for plant disease detection
"""

# Import required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Set parameters
IMG_SIZE = 128
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 8
MLP_HEAD_UNITS = [2048, 1024]
BATCH_SIZE = 32
EPOCHS = 20  # You can reduce this to 5-10 for faster training
DATA_PATH = 'plantvillage dataset/color/'

# Create directories for saving models
os.makedirs('trained_models/ViT/', exist_ok=True)

print("\n" + "="*60)
print("STEP 1: Loading and preprocessing data...")
print("="*60)

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
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
print("STEP 2: Building Vision Transformer components...")
print("="*60)

# Patch Extraction Layer
class PatchExtractor(layers.Layer):
    """Extract patches from images"""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Patch Encoding Layer
class PatchEncoder(layers.Layer):
    """Encode patches with position embeddings"""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

print("✓ Patch extraction and encoding layers created")

print("\n" + "="*60)
print("STEP 3: Building Vision Transformer model...")
print("="*60)

# Build Vision Transformer
def create_vit_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Create patches
    patches = PatchExtractor(PATCH_SIZE)(inputs)
    
    # Encode patches
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)
    
    # Transformer blocks
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = layers.Dense(PROJECTION_DIM * 2, activation='gelu')(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(PROJECTION_DIM, activation='gelu')(x3)
        x3 = layers.Dropout(0.1)(x3)
        
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])
    
    # Final layer normalization
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # MLP head
    features = representation
    for units in MLP_HEAD_UNITS:
        features = layers.Dense(units, activation='gelu')(features)
        features = layers.Dropout(0.5)(features)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(features)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_vit_model()
model.summary()

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Vision Transformer model built and compiled successfully")

print("\n" + "="*60)
print("STEP 4: Setting up callbacks...")
print("="*60)

# Callbacks for model training
checkpoint = keras.callbacks.ModelCheckpoint(
    'trained_models/ViT/vit_epoch_{epoch:02d}_acc_{val_accuracy:.2f}.keras',
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
print("STEP 5: Training the model...")
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
print("STEP 6: Saving the final model...")
print("="*60)

# Save the final model
model.save('trained_models/ViT/vit_final_model.keras')
print("✓ Model saved: trained_models/ViT/vit_final_model.keras")

# Save class names
import json
with open('trained_models/ViT/class_names.json', 'w') as f:
    json.dump(class_names, f)
print("✓ Class names saved: trained_models/ViT/class_names.json")

print("\n" + "="*60)
print("STEP 7: Generating training plots...")
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
plt.title('Vision Transformer Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Vision Transformer Loss')

plt.tight_layout()
plt.savefig('trained_models/ViT/training_history.png')
print("✓ Training plots saved: trained_models/ViT/training_history.png")

print("\n" + "="*60)
print("VISION TRANSFORMER TRAINING COMPLETE!")
print("="*60)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print("="*60)
