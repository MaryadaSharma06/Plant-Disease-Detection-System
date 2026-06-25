"""
Plant Disease Detection Web Application (Updated & Clean Version)
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
from dotenv import load_dotenv

import sys
sys.stdout.reconfigure(encoding='utf-8')

from chatbot import PlantDiseaseChat

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
MODELS = {}
CLASS_NAMES = []
IMG_SIZE = 128
CURRENT_MODEL = 'CNN'
chatbot = None


# ---------------------------------------------------
# LOAD MODEL + CHATBOT
# ---------------------------------------------------
def load_models():
    global MODELS, CLASS_NAMES, chatbot

    print("Loading model...")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(
        project_root,
        'trained_models',
        'CNN',
        'cnn_final_model.keras'
    )

    if os.path.exists(model_path):
        MODELS['CNN'] = keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
        print("✓ CNN Model loaded")
    else:
        print("✗ Model not found")

    # Load class names
    class_path = os.path.join(
        project_root,
        'trained_models',
        'CNN',
        'class_names.json'
    )

    if os.path.exists(class_path):
        with open(class_path) as f:
            CLASS_NAMES.extend(json.load(f))
        print(f"✓ Loaded {len(CLASS_NAMES)} classes")
    else:
        print("✗ class_names.json missing")

    # Initialize chatbot (ENV based)
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    try:
        if groq_key:
            chatbot = PlantDiseaseChat(api_key=groq_key, provider='groq')
            print("✓ Chatbot initialized (Groq)")
        elif gemini_key:
            chatbot = PlantDiseaseChat(api_key=gemini_key, provider='gemini')
            print("✓ Chatbot initialized (Gemini)")
        else:
            chatbot = PlantDiseaseChat()
            print("⚠️ No API key found. Chatbot disabled.")
    except Exception as e:
        print(f"✗ Chatbot error: {e}")


# ---------------------------------------------------
# IMAGE PREPROCESS
# ---------------------------------------------------
def preprocess_image(image_path):
    print("Opening image...", flush=True)

    img = Image.open(image_path).convert("RGB")

    print("Resizing...", flush=True)

    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img, dtype=np.float32) / 255.0

    img = np.expand_dims(img, axis=0)

    print("Preprocessing complete", flush=True)

    return img


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
def predict_disease(image_path, model_name='CNN'):
    try:
        print("Loading image...", flush=True)

        if model_name not in MODELS:
            raise Exception(f"Model '{model_name}' not loaded")

        img = preprocess_image(image_path)
        print(f"Image shape: {img.shape}", flush=True)

        print("Running inference...", flush=True)

        predictions = MODELS[model_name](img, training=False).numpy()

        print("Inference completed", flush=True)

        if predictions is None or len(predictions) == 0:
            raise Exception("Model returned empty predictions")

        idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][idx])

        disease = CLASS_NAMES[idx]

        top_3_idx = np.argsort(predictions[0])[-3:][::-1]

        top_3 = []

        for i in top_3_idx:
            top_3.append({
                "disease": CLASS_NAMES[int(i)],
                "confidence": round(float(predictions[0][i]) * 100, 2)
            })

        return disease, confidence * 100, top_3

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Prediction Error:", str(e), flush=True)
        return None, None, None
# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.route('/')
def index():
    return render_template(
        'index.html',
        models=list(MODELS.keys()),
        current_model=CURRENT_MODEL
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("=" * 60, flush=True)
        print("Prediction request received", flush=True)

        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files['file']

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        model_name = request.form.get("model", CURRENT_MODEL)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            "uploaded.jpg"
        )

        print("Saving image...", flush=True)
        file.save(filepath)

        print("Image saved", flush=True)

        disease, confidence, top_3 = predict_disease(
            filepath,
            model_name
        )

        if disease is None:
            print("Prediction failed", flush=True)

            return jsonify({
                "success": False,
                "error": "Prediction failed"
            }), 500

        print("Returning prediction", flush=True)

        return jsonify({
            "success": True,
            "disease": disease,
            "confidence": round(confidence, 2),
            "top_predictions": top_3,
            "image_url": "/static/uploads/uploaded.jpg"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
@app.route('/get_cards', methods=['POST'])
def get_cards():
    if not chatbot:
        return jsonify({'error': 'Chatbot not initialized'}), 500

    data = request.json
    top_3 = data.get('top_predictions', [])

    if not top_3:
        return jsonify({'error': 'No predictions provided'}), 400

    # Format for the prompt: list of (disease_name, confidence_pct)
    formatted = [(p['disease'], round(p['confidence'], 2)) for p in top_3]

    result = chatbot.get_disease_cards(formatted)
    return jsonify(result)


@app.route('/chat', methods=['POST'])
def chat():
    if not chatbot:
        return jsonify({'error': 'Chatbot not initialized'}), 500

    data = request.json
    msg = data.get('message', '')
    disease = data.get('disease', '')

    response = chatbot.get_response(msg, disease)
    return jsonify({'response': response})


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
load_models()

if not MODELS:
        print("⚠️ No models loaded!")
        
if __name__ == '__main__':

    print("\n🚀 Running on http://localhost:5001\n")

    app.run(debug=True, host='0.0.0.0', port=5001)