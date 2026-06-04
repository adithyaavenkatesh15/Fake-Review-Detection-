#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

# Fix python path so src can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import joblib, json, pandas as pd, streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt

from clean_text import clean_text
from features import extract_numeric_features
# We MUST import FastSentenceEmbedding so joblib can unpickle it
try:
    import __main__
    from train import FastSentenceEmbedding
    setattr(__main__, "FastSentenceEmbedding", FastSentenceEmbedding)
except ImportError:
    pass

PIPE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "pipeline.joblib"
METRICS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "metrics.json"

SAMPLE_REVIEW_OPTIONS = {
    "Choose an example": "",
    "Fake-looking review": "This is the best product ever!!! I got it for free and totally love it...",
    "Real review style": "The phone arrived late and battery life is poor, but I appreciated the customer support.",
    "Neutral review": "The product works as expected, though the packaging could be improved.",
}

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="centered")
st.title("🕵️ Fake Review Detector")
st.caption("Sentence Embeddings + Behavioral features + Explainable ML")

@st.cache_resource
def load_pipeline():
    if PIPE_PATH.exists(): 
        try:
            return joblib.load(PIPE_PATH)
        except Exception as e:
            st.error(f"Error loading pipeline: {e}")
    return None

def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None

pipe = load_pipeline()
metrics = load_metrics()

example_choice = st.selectbox("Example reviews", list(SAMPLE_REVIEW_OPTIONS.keys()))
if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""
if st.button("Load example review") and example_choice != "Choose an example":
    st.session_state.sample_text = SAMPLE_REVIEW_OPTIONS[example_choice]

if example_choice == "Choose an example":
    st.session_state.sample_text = st.session_state.sample_text

st.write("---")
txt = st.text_area("Paste a product review:", height=200, value=st.session_state.sample_text,
                   placeholder="Type or paste a review here...")
btn = st.button("Analyze Review", type="primary")

if btn:
    if pipe is None:
        st.error("Model not found. Train it first: `make train`")
    elif not txt.strip():
        st.warning("Please paste a review.")
    else:
        s = clean_text(txt)
        df = pd.DataFrame([{"text": txt, "text_clean": s}])
        num = extract_numeric_features([txt])
        X = pd.concat([df, num], axis=1)
        
        prob = float(pipe.predict_proba(X)[0,1])
        label = "FAKE" if prob >= 0.5 else "REAL"
        
        col1, col2 = st.columns(2)
        col1.metric("Prediction", label)
        col2.metric("Probability", f"{prob:.1%}")
        st.progress(prob if label=="FAKE" else 1-prob, text=f"Calculated Score: {prob:.2f}")
        
        # --- SHAP EXPLAINABILITY ---
        st.write("---")
        st.subheader("Decision Breakdown (SHAP)")
        try:
            clf = pipe.named_steps["clf"]
            pre = pipe.named_steps["pre"]
            
            X_trans = pre.transform(X)
            # Use zeros as baseline for explainer
            background = np.zeros((1, X_trans.shape[1]))
            explainer = shap.LinearExplainer(clf, background)
            shap_values = explainer(X_trans)
            
            # Determine how many dimensions our sentence embedding has
            num_emb_dims = X_trans.shape[1] - len(num.columns)
            
            feature_names = [f"embed_{i}" for i in range(num_emb_dims)] + list(num.columns)
            shap_values.feature_names = feature_names
            
            # Combine embedding features into one "Text Embedding" representation, 
            # while keeping behavioral features separated.
            # This makes the waterfall chart actually readable!
            emb_sum = shap_values.values[0, :num_emb_dims].sum()
            emb_base = shap_values.base_values[0]
            
            new_values = np.append([emb_sum], shap_values.values[0, num_emb_dims:])
            new_data = np.append([0], shap_values.data[0, num_emb_dims:])
            new_names = ["Text Meaning (Embeddings)"] + list(num.columns)
            
            import shap.plots as plots
            import copy
            
            # Create a mock Explanation object
            exp = shap.Explanation(
                values=new_values,
                base_values=emb_base,
                data=new_data,
                feature_names=new_names
            )
            
            fig, ax = plt.subplots(figsize=(8, 4))
            plots.waterfall(exp, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")
            
if metrics is not None:
    st.write("---")
    st.subheader("Latest model metrics")
    if "test_roc_auc" in metrics:
        st.write(f"**Test ROC AUC:** {metrics['test_roc_auc']:.3f}")
        st.write(f"**Test average precision:** {metrics['test_avg_precision']:.3f}")

st.info("Tips: Excessive punctuation or repeated clichés influence the model heavily.", icon="💡")
