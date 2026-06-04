from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from clean_text import clean_text
from features import extract_numeric_features
import __main__
from train import FastSentenceEmbedding
setattr(__main__, "FastSentenceEmbedding", FastSentenceEmbedding)

app = FastAPI(title="Fake Review API")
BASE_DIR = Path(__file__).resolve().parent.parent
PIPE_PATH = BASE_DIR / "outputs" / "pipeline.joblib"
pipe = None
class ReviewRequest(BaseModel):
    text: str
class ReviewResponse(BaseModel):
    label: str
    fake_probability: float
@app.on_event("startup")
def load_model():
    global pipe
    if PIPE_PATH.exists():
        pipe = joblib.load(PIPE_PATH)
@app.post("/predict", response_model=ReviewResponse)
def predict(rev: ReviewRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    s = clean_text(rev.text)
    df = pd.DataFrame([{"text": rev.text, "text_clean": s}])
    num = extract_numeric_features([rev.text])
    X = pd.concat([df, num], axis=1)
    prob = float(pipe.predict_proba(X)[0,1])
    label = "FAKE" if prob >= 0.5 else "REAL"
    return {"label": label, "fake_probability": prob}
@app.post("/batch")
def predict_batch(revs: list[ReviewRequest]):
    return [predict(r) for r in revs]
@app.get("/health")
def health():
    return {"status": "ok", "model_path_exists": PIPE_PATH.exists()}
