# Plant Disease Detection using Deep Learning

A complete plant disease classification system built with TensorFlow/Keras and Flask.  
It supports image-based disease prediction using multiple model architectures (CNN, VGG16, ViT) and includes an AI chatbot for treatment/prevention guidance.

## Features

- Predict plant diseases from uploaded leaf images
- Multiple deep learning models:
  - CNN
  - VGG16 (transfer learning)
  - Vision Transformer (ViT)
- Top-3 prediction output with confidence scores
- Flask web app with image upload and real-time inference
- AI chatbot support (Groq or Gemini) for disease advice
- Standalone model testing script (`test_model.py`)

## Dataset

This project uses the **PlantVillage** dataset organized by class folders.

Expected directory structure:

```text
plantvillage dataset/
  color/
  grayscale/
  segmented/
```

The training pipeline and current web app inference are configured for the `color` subset.

## Project Structure

```text
Plant Disease Detection/
  config.py
  requirements.txt
  test_model.py
  web_app/
    app.py
    chatbot.py
    templates/
    static/
  trained_models/
    CNN/
    VGG16/
    ViT/
  plantvillage dataset/
```

## Tech Stack

- Python
- TensorFlow / Keras
- Flask
- NumPy, Pillow
- scikit-learn, matplotlib
- Groq API / Google Gemini API (optional chatbot)

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd "Plant Disease Detection"
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Web App

From project root:

```bash
python web_app/app.py
```

Open your browser at:

- `http://localhost:5000`

## Test a Trained Model from Terminal

```bash
python test_model.py
```

You will be prompted to enter a test image path, for example:

```text
plantvillage dataset/color/Tomato___Late_blight/<image_name>.JPG
```

## Training Models

Training scripts are available in:

- `trained_models/CNN/train_cnn.py`
- `trained_models/VGG16/train_vgg16.py`
- `trained_models/ViT/train_vit.py`

Run them from project root:

```bash
python trained_models/CNN/train_cnn.py
python trained_models/VGG16/train_vgg16.py
python trained_models/ViT/train_vit.py
```

## API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Upload an image and get predictions
- `POST /chat` - Ask disease/treatment questions
- `POST /set-api-key` - Set chatbot API key and provider

## Configuration

Main configuration is in `config.py`:

- Training hyperparameters
- Data augmentation settings
- Web app settings
- Model/data paths
- Chatbot provider settings

## Security Note (Important)

If API keys are present in `config.py`, **remove and rotate them before publishing the repository**.

Recommended approach:

- Store keys in environment variables (for example `GROQ_API_KEY`, `GEMINI_API_KEY`)
- Keep secrets out of source control

## Known Notes

- The web app currently loads the CNN model by default in `web_app/app.py`.
- Large datasets and model checkpoint files may be too big for regular Git commits.

## Future Improvements

- Add Docker support
- Add model evaluation metrics dashboard
- Add user authentication and prediction history
- Export reports for disease diagnosis sessions

## License

Use an open-source license of your choice (MIT is a common option) and add a `LICENSE` file if you plan to make this public.
