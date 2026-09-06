from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from train_model import generate_and_train

import joblib
import os
import uvicorn


app = FastAPI(title="Spam Message Detector")


# CORS: for development allow all origins
# Lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static + templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

MODEL_DIR = "model"

MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")


# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# Automatically train the model on startup if files don't exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
    print("Model files not found. Triggering baseline training pipeline...")
    generate_and_train()


# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)


class Message(BaseModel):
    text: str


# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# Prediction API
@app.post("/predict")
def predict_spam(msg: Message):

    # Defensive: ensure text is string
    txt = (msg.text or "").strip()

    text_vec = vectorizer.transform([txt])

    # Spam probability
    prob = float(model.predict_proba(text_vec)[0][1])

    pred = int(prob > 0.5)

    result = "🚨 Spam Message" if pred == 1 else "✅ Safe Message"

    # Return confidence as percentage
    return {
        "prediction": pred,
        "result": result,
        "confidence": round(prob * 100, 2)
    }


# Auto-run the server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)