# Sentiment Analysis API (Flask + TF-IDF + Logistic Regression)

A small end-to-end sentiment analysis project: training, serving, and demo UI.

Project structure

- `app.py` — Flask API and web UI (`/test`) that loads the saved model and vectorizer.
- `train_model.py` — Training script that preprocesses text, fits a TF-IDF vectorizer and a Logistic Regression classifier, evaluates, and saves artifacts to `models/`.
- `Dockerfile` — Containerizes the app and pre-downloads required NLTK data.
- `requirements.txt` — Python dependencies.
- `data/IMDB Dataset.csv` — IMDB reviews dataset used for training (if present).
- `models/` — Saved artifacts (expected files):
  - `sentiment_model.pkl` — trained classifier
  - `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer

Quick overview

- Text is preprocessed (lowercased, HTML/URLs removed, non-letters removed, tokenized, stopwords removed).
- TF-IDF (`TfidfVectorizer(ngram_range=(1,2), max_features=5000, min_df=2)`) creates the feature vectors.
- Logistic Regression (`sklearn.linear_model.LogisticRegression`) performs binary classification (positive=1 / negative=0).
- The Flask app exposes:
  - `GET /` — API info
  - `GET /health` — health check
  - `POST /predict` — JSON input `{ "text": "..." }` → returns `sentiment`, `confidence`, and `probabilities`
  - `GET /test` — interactive web demo UI

Run locally (recommended)

1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. (Optional) Train the model

If you want to retrain or produce the `models/` artifacts, run:

```bash
python3 train_model.py
```

This will:
- load `data/IMDB Dataset.csv` (make sure path exists),
- preprocess text,
- train TF-IDF + LogisticRegression,
- save `models/sentiment_model.pkl` and `models/tfidf_vectorizer.pkl`.

3. Start the Flask app

```bash
python3 app.py
```

Open the demo UI at `http://localhost:5000/test` or call the predict endpoint with curl:

```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"I loved this movie, it was fantastic!"}'
```

Run with Docker

1. Build the image (the Dockerfile pre-downloads NLTK data):

```bash
docker build -t sentiment-analysis-api .
```

2. Run the container and map a host port (if 5000 is in use choose e.g. 5002):

```bash
docker run -p 5000:5000 sentiment-analysis-api
# or if 5000 is already taken
docker run -p 5002:5000 sentiment-analysis-api
```

Then open `http://localhost:5000/test` (or `:5002/test`).

Troubleshooting

- Error: "Resource punkt_tab not found" or similar NLTK errors
  - Cause: a bad resource name or missing NLTK data. Earlier versions of this repo had a mistaken `nltk.download('punkt_tab')` call which does not exist. That call was removed.
  - Fix locally: ensure required NLTK data is installed in the runtime environment:

```bash
python3 -m pip install -r requirements.txt
python3 - <<'PY'
import nltk
nltk.download('punkt')
nltk.download('stopwords')
PY
```

  - Fix in Docker: the Dockerfile already runs `nltk.download('stopwords')` and `nltk.download('punkt')` during build. If you modified the image, rebuild it.

- Port already allocated when starting Docker
  - If `docker run -p 5000:5000` fails with "Bind for 0.0.0.0:5000 failed: port is already allocated", another process or container is using that port. Either stop the other container or map to a different host port:

```bash
# list containers and ports
docker ps
# stop a container using a port
docker stop <container-id-or-name>
# run on a different host port
docker run -p 5001:5000 sentiment-analysis-api
```

- If tokenizer still raises errors inside Docker
  - Make sure the container you run was built from the repo's `Dockerfile` (it pre-downloads `punkt` and `stopwords`). Rebuild if necessary.
  - If you are running in a restricted network (no internet during build/runtime), download NLTK data locally and mount it as `/app/nltk_data` or use `nltk.data.path.append(...)` in `app.py` pointing to bundled data.

Notes & suggestions

- Preprocessing vs vectorizer stopwords: the repo removes stopwords during preprocessing and could also pass `stop_words='english'` to `TfidfVectorizer`. You only need one of those; using both is redundant.
- Consider packaging preprocessing + vectorizer + model into a scikit-learn `Pipeline` and saving a single pipeline pickle — this prevents train/serve skew and simplifies loading in `app.py`.
- For production use consider adding:
  - logging, input size limits, and request validation
  - a small caching layer for frequent requests
  - model versioning and CI for retraining

Contact / Next steps

If you'd like, I can:
- Add a single scikit-learn `Pipeline` and update `train_model.py`/`app.py` to use it,
- Add simple unit tests for preprocessing and the `/predict` endpoint,
- Add a minimal `Makefile` or script to streamline build/run steps.

LICENSE

This repository currently has no explicit license. Add one if you plan to publish the code.
