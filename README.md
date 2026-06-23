# 🌿 Plant Disease Detection System with AI Assistant

## 📌 Overview

Plant Disease Detection System is an AI-powered web application that helps identify plant diseases from leaf images and provides treatment recommendations using an intelligent chatbot assistant.

The system uses Deep Learning for disease classification and Large Language Models (LLMs) to generate disease insights, prevention tips, and treatment suggestions.

---

## 🚀 Features

* Upload plant leaf images for disease detection
* Deep Learning-based disease classification
* Supports 38 plant disease classes
* Displays prediction confidence scores
* AI-powered chatbot for disease-related queries
* Prevention and treatment recommendations
* Interactive and user-friendly web interface
* Real-time image analysis

---

## 🛠️ Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Flask
* Python

### Machine Learning

* TensorFlow
* Keras
* NumPy
* Pillow (PIL)

### AI Assistant

* Groq API
* Gemini API (Optional)

### Version Control

* Git
* GitHub

---

## 🧠 Models Evaluated

| Model                    | Accuracy |
| ------------------------ | -------- |
| CNN                      | ~97%     |
| VGG16                    | ~99%     |
| Vision Transformer (ViT) | ~89%     |

For deployment, the CNN model was selected due to its excellent balance between accuracy, model size, and inference speed.

---

## 📂 Project Structure

```bash
Plant-Disease-Detection-System/
│
├── web_app/
│   ├── app.py
│   ├── chatbot.py
│   ├── templates/
│   └── static/
│
├── trained_models/
│   ├── CNN/
│   ├── VGG16/
│   └── ViT/
│
├── requirements.txt
├── README.md
└── config.py
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/MaryadaSharma06/Plant-Disease-Detection-System.git
cd Plant-Disease-Detection-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

(Optional)

```env
GEMINI_API_KEY=your_gemini_api_key
```

---

## ▶️ Run Application

```bash
cd web_app
python app.py
```

Open:

```text
http://localhost:5001
```

---

## 📊 Dataset

The project was trained using the PlantVillage dataset containing healthy and diseased leaf images from multiple plant species.

---

## 🔍 Workflow

1. User uploads a plant leaf image.
2. Image is preprocessed and resized.
3. CNN model predicts the disease class.
4. Confidence scores are generated.
5. AI assistant provides:

   * Disease description
   * Symptoms
   * Prevention methods
   * Treatment recommendations

---

## 📸 Screenshots

### Home Page

<img width="848" height="410" alt="image" src="https://github.com/user-attachments/assets/282584da-7a76-41dd-9bb8-0bed1d3ed951" />


### Disease Prediction

<img width="845" height="407" alt="image" src="https://github.com/user-attachments/assets/241e7ebc-7ef5-4987-a7c9-431ab180a0a0" />


### AI Disease Analysis Card
<img width="840" height="405" alt="image" src="https://github.com/user-attachments/assets/f10ff6bd-16ab-4a11-9f11-bbe917c4947d" />

<img width="835" height="409" alt="image" src="https://github.com/user-attachments/assets/1b4b82d1-3f1f-471d-8341-4d83116f24cb" />


### AI Disease Analysis Card

<img width="844" height="407" alt="image" src="https://github.com/user-attachments/assets/da18186d-b027-46bb-8a6e-50467621b482" />




---

## 🔮 Future Enhancements

* User authentication and profile management
* Prediction history tracking
* PDF report generation
* Mobile application support
* Multi-language chatbot support
* Cloud deployment and scalability improvements

---

## 👩‍💻 Author

**Maryada Sharma**

B.Tech Computer Science Engineering
BML Munjal University

GitHub: https://github.com/MaryadaSharma06

---

## 📜 License

This project is developed for educational and research purposes.
