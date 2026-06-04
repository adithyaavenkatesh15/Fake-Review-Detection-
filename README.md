# Fake Review Detector 🕵️

An industry-ready machine learning system that detects **fake vs real** product reviews. 

This upgraded project leverages **Sentence Embeddings (SentenceTransformers)** alongside **behavioral text features** (e.g., exclamation count, repeated phrases, emoji usage, and sentence structure) to accurately capture nuanced linguistic patterns that distinguish authentic reviews from fake ones. The system is served via a **FastAPI** microservice, provides a rich **Streamlit** user interface with true explainability via **SHAP**, and tracks model iterations automatically using **MLflow**.

---

## 🚀 Key Features

*   **Advanced MLOps Pipeline:** Automatic hyperparameter tuning with **Optuna** and complete experiment tracking via **MLflow**.
*   **Modern NLP Models:** Utilizes `all-MiniLM-L6-v2` Sentence Transformers for rich semantic embeddings, combined with robust behavioral feature engineering.
*   **Explainable AI (XAI):** Built-in **SHAP** (SHapley Additive exPlanations) waterfall plots in the front-end to explain exactly *why* a review was flagged as fake or real.
*   **Robust Microservice Architecture:** Decoupled architecture serving the model asynchronously via **FastAPI** with strict **Pydantic** data validation.
*   **Interactive UI:** Beautiful **Streamlit** interface for analyzing individual reviews, adjusting thresholds, and viewing metrics alongside explanations.
*   **Containerization:** **Dockerized** API and UI layers mapped with `docker-compose` for rapid, deterministic deployment.
*   **Continuous Integration:** Triggered **GitHub Actions** (`ci.yml`) enforcing code quality (Ruff linting) and regression testing.

---

## 🛠️ Technology Stack

*   **Modeling & NLP:** `scikit-learn`, `SentenceTransformers`, `NLTK`, `TextBlob`
*   **Tracking & Tuning:** `MLflow`, `Optuna`
*   **Explainability:** `SHAP`
*   **Backend / API:** `FastAPI`, `Uvicorn`, `Pydantic`
*   **Frontend:** `Streamlit`
*   **Containerization DevOps:** `Docker`, `Docker Compose`, `GitHub Actions`
*   **Validation:** `pytest`, `Ruff`

---

## 📂 Folder Structure

```text
fake-review-detector/
├── app/
│   ├── api.py                    # FastAPI server & endpoints
│   └── streamlit_app.py          # Streamlit UI with SHAP visualizations
├── src/
│   ├── clean_text.py             # Rule-based text preprocessing
│   ├── features.py               # Behavioral & statistical feature extraction
│   └── train.py                  # MLflow pipeline, Optuna tuning, and model training
├── tests/
│   ├── test_clean_text.py        # Preprocessing unit tests
│   ├── test_features.py          # Feature extraction tests
│   └── test_predict.py           # Verification of overall pipeline
├── data/
│   └── reviews_sample.csv        # Example dataset for model training
├── outputs/                      # Saved models and metric logs
│   ├── pipeline.joblib           
│   └── metrics.json              
├── .github/workflows/
│   └── ci.yml                    # Automated GitHub Actions Pipeline
├── Dockerfile                    # Containerization instructions
├── docker-compose.yml            # Multi-container orchestration (api & ui)
├── pyproject.toml                # Project configurations & dependencies
└── README.md                     # Project documentation
```

---

## 💻 Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.9+** installed. A virtual environment is highly recommended.

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies
Install the package along with development dependencies defined in `pyproject.toml`.

```bash
pip install -e .[dev]
```

### 3. Train the Model (Required)
Before running the interactive API or UI, you must generate the serialized model using the `train.py` script. This script automatically logs the run in MLflow, performs Optuna tuning, and exports the final `pipeline.joblib`.

```bash
python src/train.py --csv data/reviews_sample.csv --outdir outputs
```

---

## 🚦 Running the Application

There are multiple ways to deploy and run the system:

### Option A: Using Docker Compose (Recommended)
Docker runs both the FastAPI backend and Streamlit frontend in tandem.

```bash
docker-compose up --build
```
*   **UI:** `http://localhost:8501`
*   **API:** `http://localhost:8000`

### Option B: Running Locally via Terminal
If you prefer running services directly via Python:

**1. Start the FastAPI Server:**
```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

**2. Start the Streamlit UI** *(In a separate terminal)*:
```bash
python -m streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

---

## 🔌 API Documentation

Once the FastAPI server is running, interactive Swagger documentation is automatically generated at `http://localhost:8000/docs`.

### **`POST /predict`**
Submit a single text review to receive its classification score.
**Payload:**
```json
{
  "text": "Best product ever!!! So amazing, got it for free!!!!"
}
```
**Response:**
```json
{
  "label": "FAKE",
  "fake_probability": 0.89
}
```

### **`POST /batch`**
Submit multiple text reviews in a list array.

### **`GET /health`**
Verifies system operation status and whether the serialized model successfully exists.

---

## 🧑‍💻 Contributing
Pull requests are welcome! Ensure that all additions pass the CI checks automatically enabled.

```bash
# Check code syntax formatting
make lint

# Execute all tests
make test
```

## 📜 License
This project is licensed under the MIT License. See `LICENSE` for details.
