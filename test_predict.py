import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from predict import predict_single
from train import build_pipeline
from clean_text import clean_text
from features import extract_numeric_features


def test_predict_single_returns_valid_output(tmp_path):
    texts = [
        "Amazing product, highly recommend it and worth every penny.",
        "I highly recommend this product because it is excellent and worth buying.",
    ]
    cleaned = [clean_text(text) for text in texts]
    X = pd.DataFrame([{"text": t, "text_clean": c} for t, c in zip(texts, cleaned)])
    numeric = extract_numeric_features(texts)
    X = pd.concat([X, numeric], axis=1)

    pipeline = build_pipeline()
    pipeline.fit(X, [1, 0])
    pipeline_path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    result = predict_single(pipeline_path, texts[0])
    assert result["label"] in {"FAKE", "REAL"}
    assert 0.0 <= result["fake_probability"] <= 1.0
