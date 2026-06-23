"""
Test the trained model with a sample image
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os

def test_model(model_path, image_path, class_names_path):
    """Test a trained model with an image"""
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load class names
    print(f"Loading class names from {class_names_path}...")
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        print(f"✓ Loaded {len(class_names)} classes")
    except Exception as e:
        print(f"✗ Error loading class names: {e}")
        return
    
    # Load and preprocess image
    print(f"Loading image from {image_path}...")
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print("✓ Image loaded and preprocessed")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return
    
    # Make prediction
    print("\nMaking prediction...")
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"\n🎯 Detected Disease: {class_names[predicted_class_idx]}")
    print(f"📊 Confidence: {confidence*100:.2f}%")
    print(f"\nTop 3 Predictions:")
    for i, idx in enumerate(top_3_idx, 1):
        print(f"  {i}. {class_names[idx]}: {predictions[0][idx]*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    # Test CNN model
    model_path = "trained_models/CNN/cnn_final_model.keras"
    class_names_path = "trained_models/CNN/class_names.json"
    
    # You need to provide a test image path
    # For example: "plantvillage dataset/color/Tomato___Late_blight/00000001.jpg"
    
    print("Plant Disease Detection - Model Test")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n⚠️  Model not found at: {model_path}")
        print("Please train the model first by running the notebook:")
        print("  models/VGG16/vgg16_training.ipynb")
        print("\nOr change the model_path to point to an existing model.")
    else:
        # Get test image from user
        test_image = input("\nEnter path to test image: ").strip('"').strip("'")
        
        if not os.path.exists(test_image):
            print(f"\n✗ Image not found at: {test_image}")
            print("\nExample usage:")
            print('  python test_model.py')
            print('  Enter: plantvillage dataset/color/Tomato___Late_blight/00000001.jpg')
        else:
            test_model(model_path, test_image, class_names_path)
