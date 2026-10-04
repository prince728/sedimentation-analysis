from pathlib import Path
from contextlib import asynccontextmanager

import pickle
import re
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Get the directory where main.py is located
BASE_DIR = Path(__file__).resolve().parent

# Model and tokenizer paths
model_path = BASE_DIR / "Artifacts" / "BiGRU_Model.keras"
tokenizer_path = BASE_DIR / "Artifacts" / "tokenizer.pkl"

# Static folder
static_path = BASE_DIR / "static"


# ============================================================
# 2. CONSTANTS
# ============================================================

# Maximum number of tokens the model accepts
max_sequence_length = 50


# Emotion labels
emotion_labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]


# Emotion emojis
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


# ============================================================
# 3. PREPROCESSING
# ============================================================

def preprocess_text(text: str) -> str:
    """
    Clean the input text so that it matches
    the preprocessing used during model training.

    Steps:
    1. Convert text to lowercase
    2. Remove apostrophes
    3. Remove special characters and punctuation
    4. Remove extra spaces
    """

    # Convert to lowercase
    text = text.lower()

    # Remove apostrophes
    # Example:
    # can't -> cant
    # I'm -> Im
    text = re.sub(r"'", "", text)

    # Keep only:
    # a-z
    # 0-9
    # whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 4. REQUEST / RESPONSE SCHEMAS
# ============================================================

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={
            "example": "I feel so happy and excited"
        }
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ============================================================
# 5. MODEL STORAGE
# ============================================================

# Stores:
#
# {
#     "BiGRU": loaded model,
#     "Tokenizer": loaded tokenizer
# }
#
dl_model = {}


# ============================================================
# 6. APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n========================================")
    print("Loading the model and tokenizer...")
    print("========================================")

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    print(f"Model path: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nModel file not found!\n"
            f"Expected location:\n{model_path}\n\n"
            f"Please make sure this file exists:\n"
            f"Artifacts/BiGRU_Model.keras"
        )

    print("Model file found.")


    # --------------------------------------------------------
    # Check tokenizer file
    # --------------------------------------------------------

    print(f"Tokenizer path: {tokenizer_path}")

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"\nTokenizer file not found!\n"
            f"Expected location:\n{tokenizer_path}\n\n"
            f"Please make sure this file exists:\n"
            f"Artifacts/tokenizer.pkl"
        )

    print("Tokenizer file found.")


    # --------------------------------------------------------
    # Check static directory
    # --------------------------------------------------------

    if not static_path.exists():
        print(
            f"WARNING: Static directory not found:\n"
            f"{static_path}"
        )
    else:
        print("Static directory found.")


    # --------------------------------------------------------
    # Load BiGRU model
    # --------------------------------------------------------

    print("\nLoading BiGRU model...")

    dl_model["BiGRU"] = load_model(model_path)

    print("BiGRU model loaded successfully.")


    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("Loading tokenizer...")

    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)

    print("Tokenizer loaded successfully.")


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n========================================")
    print("Model and tokenizer loaded successfully.")
    print("Server is ready.")
    print("========================================\n")


    # --------------------------------------------------------
    # Server runs here
    # --------------------------------------------------------

    yield


    # --------------------------------------------------------
    # Server shutdown
    # --------------------------------------------------------

    print("\nShutting down server...")

    dl_model.clear()

    print("Model and tokenizer removed from memory.")


# ============================================================
# 7. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="BiGRU Emotion Classification API",
    description="API for predicting emotions from text using a BiGRU model.",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# 8. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    # For development
    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# 9. STATIC FILES
# ============================================================

# Only mount if static directory exists
if static_path.exists():

    app.mount(
        "/static",
        StaticFiles(directory=str(static_path)),
        name="static"
    )


# ============================================================
# 10. HOME PAGE
# ============================================================

@app.get(
    "/",
    include_in_schema=False
)
def server_ui():

    index_path = static_path / "index.html"

    if not index_path.exists():

        raise HTTPException(
            status_code=404,
            detail="index.html not found inside static folder."
        )

    return FileResponse(index_path)


# ============================================================
# 11. HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    return HealthResponse(
        status="Server is running",
        model_loaded=(
            "BiGRU" in dl_model
            and "Tokenizer" in dl_model
        )
    )


# ============================================================
# 12. PREDICT EMOTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_emotion(text_input: TextInput):

    """
    Prediction pipeline:

    User text
          ↓
    Preprocessing
          ↓
    Tokenizer
          ↓
    Integer sequence
          ↓
    Padding
          ↓
    BiGRU model
          ↓
    Emotion probabilities
          ↓
    Highest probability
          ↓
    JSON response
    """


    # --------------------------------------------------------
    # Get model and tokenizer
    # --------------------------------------------------------

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")


    # --------------------------------------------------------
    # Check model availability
    # --------------------------------------------------------

    if BiGRU_model is None or tokenizer_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded yet. "
                "Please try again later."
            )
        )


    # --------------------------------------------------------
    # 1. Preprocess text
    # --------------------------------------------------------

    cleaned_text = preprocess_text(
        text_input.text
    )


    # Check if preprocessing removed everything
    if not cleaned_text:

        raise HTTPException(
            status_code=400,
            detail=(
                "Input text does not contain any "
                "valid characters."
            )
        )


    # --------------------------------------------------------
    # 2. Convert text to integer sequence
    # --------------------------------------------------------

    tokenized_text = tokenizer_model.texts_to_sequences(
        [cleaned_text]
    )


    # --------------------------------------------------------
    # 3. Pad sequence
    # --------------------------------------------------------

    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )


    # --------------------------------------------------------
    # 4. Run model prediction
    # --------------------------------------------------------

    probabilities = BiGRU_model.predict(
        padded_sequence,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # 5. Find highest probability
    # --------------------------------------------------------

    top_emotion_index = int(
        np.argmax(probabilities)
    )


    # --------------------------------------------------------
    # 6. Get predicted emotion
    # --------------------------------------------------------

    predicted_emotion = emotion_labels[
        top_emotion_index
    ]


    # --------------------------------------------------------
    # 7. Get confidence
    # --------------------------------------------------------

    confidence = float(
        probabilities[top_emotion_index]
    )


    # --------------------------------------------------------
    # 8. Create probability dictionary
    # --------------------------------------------------------

    all_probabilities = {
        label: float(probability)
        for probability, label
        in zip(probabilities, emotion_labels)
    }


    # --------------------------------------------------------
    # 9. Get emoji
    # --------------------------------------------------------

    emoji = EMOTION_EMOJIS.get(
        predicted_emotion,
        ""
    )


    # --------------------------------------------------------
    # 10. Return response
    # --------------------------------------------------------

    return PredictionResponse(

        text=text_input.text,

        predicted_emotion=predicted_emotion,

        emoji=emoji,

        confidence=confidence,

        all_probabilities=all_probabilities
    )