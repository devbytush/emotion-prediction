from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import re
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from keras.models import load_model
import pickle
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = FastAPI()

@app.get("/")
def greet():
    return {"Hello gng"}

'''
1. We are going to make some constants like:
A. Model path (BiGRU)
B. Tokenizer path
C. Max sequence length
D. Emotion labels
E. Emotion Emojis
'''
# A. Model path (BiGRU)
MODEL_PATH = "Artifacts/BiGRU_model.keras"

# B. Tokenizer path
TOKENIZER_PATH = "Artifacts/tokenizer.pkl"

# C. Max sequence length
MAX_SEQUENCE_LENGTH = 50

# D. Emotion labels
EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# E. Emotion Emojis
EMOTION_EMOJIS = {
    'sadness': '😢',
    'joy': '😄',
    'love': '❤️',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😲'
}

'''
2. Preprocess the upcoming text data for prediction
Cleans raw text data and prepares it for model prediction.
A. Lowercase the text
B. Remove apostrophes
C. Remove special characters
D. remove extra whitespace
'''

def preprocess_text(text)->str:
    # A. Lowercase the text
    text = text.lower()

    # B. Remove apostrophes
    text = re.sub(r"'", "", text)

    # C. Remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # D. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


'''
3. Request and Response Schemas
A. Text Input
B. Prediction Response
C. Health Check Response'''

class TextInput(BaseModel):
    text: str = Field(...,
                      min_length=1,
                      max_length=2000,
                      description="Input text for emotion prediction",
                      json_schema_extra={
                          "example": "I am feeling very happy today!"
                        })

class PredictionResponse(BaseModel):
    text: str 
    predicted_emotion: str
    confidence: float
    all_emotions: dict[str, float]

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool

'''
4. Model Loading and Lifespan Management
A. Load the model and tokenizer at startup
'''    

dl_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up and loading model...")
    dl_model["BiGRU"] = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        dl_model["Tokenizer"] = pickle.load(f)

    print("Model and tokenizer loaded successfully.")

    yield  # This allows the app to run {asynchronously} while the model is loaded

    dl_model.clear()

'''
5. Mount the static files to the FastAPI app
A. Enable CORS
'''
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True, 
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")

'''
6. Define the API endpoints
A. Server UI at homepage ("/")
B. Health Check Endpoint ("/health")
C. Predict Emotion Endpoint ("/predict")
'''
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse("static/index.html")

@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    model_loaded = "BiGRU" in dl_model and "Tokenizer" in dl_model
    return HealthCheckResponse(status="healthy", model_loaded=model_loaded)

@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    BiGRU_model = dl_model.get("BiGRU")
    Tokenizer = dl_model.get("Tokenizer")
    if BiGRU_model is None or Tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    # Preprocess the input text
    processed_text = preprocess_text(text_input.text)

    # Tokenize and pad the input text
    sequences = Tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    # Make prediction
    predictions = BiGRU_model.predict(padded_sequences)[0]
    top_emotion_index = predictions.argmax()
    all_probabilities = {label: float(prob) for label, prob in zip(EMOTION_LABELS, predictions)}

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=EMOTION_LABELS[top_emotion_index],
        confidence=float(predictions[top_emotion_index]),
        all_emotions=all_probabilities
    )
