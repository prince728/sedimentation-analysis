# 🧠 AI-Based Text Emotion Detection using BiGRU

An AI-powered text emotion detection application that analyzes user-written text and predicts the emotion expressed in it using a **Bidirectional Gated Recurrent Unit (BiGRU)** deep-learning model.

The application uses **TensorFlow/Keras** for model inference, **FastAPI** for the backend API, and a static web interface for user interaction. It returns the predicted emotion, a corresponding emoji, confidence score, and probability scores for all supported emotions.

## ✨ Features

- **Text Emotion Detection:** Predicts the emotion expressed in a sentence.
- **Six Emotion Classes:** Supports sadness, joy, love, anger, fear, and surprise.
- **BiGRU Deep Learning Model:** Uses a trained neural network for text classification.
- **Text Preprocessing:** Converts text to lowercase, removes apostrophes and special characters, and normalizes whitespace.
- **Confidence Score:** Returns the model's probability for the predicted emotion.
- **Emotion Probabilities:** Provides probability scores for all six emotion classes.
- **REST API:** Exposes prediction and health-check endpoints using FastAPI.
- **Web Interface:** Serves the frontend directly through the FastAPI application.
- **Input Validation:** Validates text input and handles invalid requests.
- **Model Lifecycle Management:** Loads the trained model and tokenizer when the application starts.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep-learning model inference |
| BiGRU | Text emotion classification |
| NumPy | Numerical operations and probability processing |
| FastAPI | Backend API development |
| Pydantic | Request and response validation |
| Uvicorn | ASGI server |
| HTML, CSS, JavaScript | Web interface |
| Jupyter Notebook | Model experimentation and analysis |
| Pickle | Tokenizer serialization |

## 🎭 Supported Emotions

| Emotion | Emoji |
|---|---|
| Sadness | 😢 |
| Joy | 😄 |
| Love | ❤️ |
| Anger | 😠 |
| Fear | 😨 |
| Surprise | 😲 |

## ⚙️ How It Works

The application processes text through the following pipeline:

```text
        User Input
            |
            v
    Text Preprocessing
            |
            v
   Tokenizer Conversion
            |
            v
    Sequence Padding
      (Length = 50)
            |
            v
    Trained BiGRU Model
            |
            v
   Emotion Probabilities
            |
            v
  Highest Probability Class
            |
            v
  Predicted Emotion + Emoji
  Confidence + All Probabilities
            |
            v
       JSON Response
```

### Processing steps

1. **Text input:** The user enters a sentence for analysis.
2. **Preprocessing:** The application converts text to lowercase, removes apostrophes, replaces special characters with spaces, and removes extra whitespace.
3. **Tokenization:** The saved tokenizer converts words into integer sequences.
4. **Padding:** The sequence is padded or truncated to a maximum length of 50 tokens.
5. **Prediction:** The trained BiGRU model generates probabilities for the six emotion classes.
6. **Classification:** The class with the highest probability is selected as the predicted emotion.
7. **Response:** The API returns the original text, predicted emotion, emoji, confidence, and all emotion probabilities.

## 📂 Project Structure

```text
sedimentation-analysis/
│
├── Artifacts/
│   ├── BiGRU_Model.keras
│   └── tokenizer.pkl
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── main.py
├── requirements.txt
├── sedimentation_analysis.ipynb
└── README.md
```

**Key files and directories:**

- `main.py` — FastAPI application, text preprocessing, model loading, and prediction endpoints.
- `Artifacts/BiGRU_Model.keras` — Trained BiGRU model used for inference.
- `Artifacts/tokenizer.pkl` — Saved tokenizer used to convert text into sequences.
- `static/` — Frontend assets served by FastAPI.
- `sedimentation_analysis.ipynb` — Jupyter Notebook for model-related experimentation and analysis.
- `requirements.txt` — Python dependency list.

The frontend filenames above illustrate the intended organization; retain the actual filenames present in your `static/` directory.

## 🚀 Installation and Setup

### Prerequisites

Make sure you have the following installed:

- Python with a version compatible with the dependencies
- pip
- Git
