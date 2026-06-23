"""
Configuration file for the Plant Disease Detection System
(Updated - Secure & Production Ready)
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# MODEL TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    'IMG_SIZE': 128,
    'BATCH_SIZE': 32,

    'EPOCHS_CNN': 25,
    'EPOCHS_VGG16': 20,
    'EPOCHS_ViT': 20,

    'VALIDATION_SPLIT': 0.2,
    'LEARNING_RATE': 0.001,

    'EARLY_STOP_PATIENCE': 5,
    'REDUCE_LR_PATIENCE': 3,
}

# =============================================================================
# DATA AUGMENTATION CONFIGURATION
# =============================================================================

AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'rescale': 1./255
}

# =============================================================================
# VISION TRANSFORMER CONFIGURATION
# =============================================================================

VIT_CONFIG = {
    'PATCH_SIZE': 16,
    'PROJECTION_DIM': 64,
    'NUM_HEADS': 4,
    'TRANSFORMER_LAYERS': 4,
}

# =============================================================================
# WEB APPLICATION CONFIGURATION
# =============================================================================

WEB_CONFIG = {
    'HOST': '0.0.0.0',
    'PORT': 5001,  # ✅ Changed to avoid conflict
    'DEBUG': True,
    'MAX_FILE_SIZE': 16 * 1024 * 1024,
    'UPLOAD_FOLDER': 'static/uploads',
    'DEFAULT_MODEL': 'CNN',
}

# =============================================================================
# FILE PATHS CONFIGURATION
# =============================================================================

PATHS = {
    'DATASET': '../plantvillage dataset/color/',
    'TRAINED_MODELS': '../trained_models/',
    'CNN_MODEL': '../trained_models/CNN/cnn_final_model.keras',
    'VGG16_MODEL': '../trained_models/VGG16/vgg16_final_model.keras',
    'ViT_MODEL': '../trained_models/ViT/vit_final_model.keras',
}

# =============================================================================
# AI CHATBOT CONFIGURATION (ENV BASED - SECURE)
# =============================================================================

CHATBOT_CONFIG = {
    'GROQ_API_KEY': os.getenv("GROQ_API_KEY"),
    'GEMINI_API_KEY': os.getenv("GEMINI_API_KEY"),

    'GROQ_MODEL': 'mixtral-8x7b-32768',
    'GEMINI_MODEL': 'gemini-3-flash-preview',

    'TEMPERATURE': 0.7,
    'MAX_TOKENS': 1024,
}

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

DISPLAY_CONFIG = {
    'TOP_K_PREDICTIONS': 3,
    'CONFIDENCE_THRESHOLD': 0.5,
}

# =============================================================================
# DEMO / PRODUCTION MODES
# =============================================================================

DEMO_CONFIG = {
    'QUICK_TRAINING': {
        'EPOCHS': 5,
        'BATCH_SIZE': 16,
        'IMG_SIZE': 96,
    },
    'PRODUCTION': {
        'EPOCHS': 25,
        'BATCH_SIZE': 32,
        'IMG_SIZE': 128,
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_training_config(mode='production'):
    if mode == 'demo':
        return DEMO_CONFIG['QUICK_TRAINING']
    return TRAINING_CONFIG


def get_model_path(model_name):
    return PATHS.get(f'{model_name}_MODEL', None)