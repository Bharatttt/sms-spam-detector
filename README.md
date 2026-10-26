# SMS Spam Detector

Classify an SMS as **spam** or **ham** (a legitimate message) in a small web app or through a JSON API. The app uses a TF-IDF + Logistic Regression pipeline. A trained model is included, so you can run the demo without training anything first. No API key or external AI service is needed.

## Included model

`model/spam_pipeline.joblib` contains the fitted TF-IDF vectorizer and Logistic Regression classifier. It is loaded automatically when the server starts. Raw SMS datasets are not included.

## Run the demo

Python 3.11 is recommended. The application has also been checked locally with Python 3.9. Run the following from PowerShell:

```powershell
git clone https://github.com/Bharatttt/sms-spam-detector.git
cd sms-spam-detector
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8010
```

Open **http://127.0.0.1:8010**, enter a message, and click **Check message**. The page shows the predicted label and both class probabilities. Keep the terminal open; press `Ctrl+C` to stop the server.

On macOS or Linux:

```bash
git clone https://github.com/Bharatttt/sms-spam-detector.git
cd sms-spam-detector
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8010
```

Interactive API documentation is available at **http://127.0.0.1:8010/docs**.

## How it works

1. The browser sends the message to `POST /predict`.
2. Python lowercases the text, removes characters other than English letters and whitespace, and normalizes spacing.
3. TF-IDF converts the cleaned text into numerical features.
4. Logistic Regression estimates the spam probability. A score of 0.5 or higher produces the spam label.
5. The API returns the label and probabilities to the page.

The saved pipeline contains both the vectorizer and classifier. Incoming messages are processed locally. The application does not write submitted messages to a database.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Web demo |
| GET | `/health` | Server status and whether the model is loaded |
| POST | `/predict` | Classify one message |
| GET | `/docs` | Interactive API documentation |

PowerShell example:

```powershell
$body = @{ message = 'Hey, are we still on for dinner tonight at 7?' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8010/predict' -Method Post -ContentType 'application/json' -Body $body
```

Example response recorded from the original local model:

```json
{
  "message": "Hey, are we still on for dinner tonight at 7?",
  "label": "ham",
  "is_spam": false,
  "spam_probability": 0.46,
  "ham_probability": 0.54
}
```

Messages must contain 1 to 2,000 characters. Empty or oversized inputs return HTTP 422; whitespace-only messages return HTTP 400. The API returns HTTP 503 if its model has not loaded.

## Model and recorded results

The original training notes describe a combined corpus of 16 SMS datasets with 2,543 messages after preprocessing and deduplication. The recorded validation split contains 2,161 training messages and 382 validation messages, stratified with `random_state=42`.

These results come from the existing `model/metrics.json`; they have not been recomputed for this release because the source datasets are not included.

| Metric | Recorded value |
|---|---:|
| Accuracy | 91.36% |
| Precision (spam) | 91.44% |
| Recall (spam) | 90.96% |
| F1 (spam) | 91.20% |
| Specificity | 91.75% |

The recorded confusion matrix contains 178 true negatives, 16 false positives, 17 false negatives, and 171 true positives. The training script evaluates a separate validation pipeline, then fits its final model on the full corpus. Testing that final model on those same validation messages would no longer be an independent evaluation.

This is a small English-language classifier. It can miss unfamiliar scams or flag legitimate messages. The preprocessing removes numbers, punctuation, URLs' non-letter characters, and non-English scripts, which can discard useful signals. Probabilities are model estimates and have not been independently calibrated. Use the demo to inspect predictions; do not automatically delete or block messages based on it.

## Retrain with your datasets

The raw SMS files are excluded from this repository. To retrain, obtain datasets you have permission to use and place the 16 CSV files listed in `data_utils.DATASET_FILES` into one folder. Edit that list if your files have different names.

```powershell
.\.venv\Scripts\python.exe train_model.py "C:/path/to/datasets"
```

A trailing slash is not required. Missing filenames are reported before training begins. The script cleans and combines the files, removes duplicate cleaned messages, evaluates the model, and replaces `model/spam_pipeline.joblib` and `model/metrics.json`. Back up these two files first if you want to keep an existing model.

The loader supports several inconsistent CSV layouts, but inspect your parsed labels and messages before relying on a new dataset. Raw data and spreadsheet files are ignored by Git to help prevent accidental publication. Dataset provenance and redistribution terms must be checked separately before publishing a newly trained artifact.

## Docker

The image includes the trained model from this repository.

```bash
docker build -t sms-spam-detector .
docker run --rm -p 127.0.0.1:8010:8000 sms-spam-detector
```

Open **http://127.0.0.1:8010**. Docker serves on port 8000 inside the container; the command maps it to port 8010 on your computer.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The checks cover model loading, example predictions, probability bounds, input validation, unavailable-model behavior, and dataset paths without a trailing slash. GitHub Actions runs all six checks with the included model on Python 3.11, then starts the server and checks the web page, health endpoint, and prediction API.

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI endpoints and the HTML/CSS/JavaScript demo |
| `data_utils.py` | CSV loading, text cleaning, and corpus deduplication |
| `train_model.py` | Training and validation command |
| `model/spam_pipeline.joblib` | Included trained vectorizer and Logistic Regression classifier |
| `model/metrics.json` | Recorded validation results |
| `requirements.txt` | Pinned Python dependencies |
| `Dockerfile` | Container build and startup command |
| `tests/test_classifier.py` | Local regression checks |
| `.github/workflows/checks.yml` | Automated checks on pushes and pull requests |

## Troubleshooting

- **Python is not recognized:** install Python and reopen the terminal, or use an existing Python executable by its full path.
- **No module named uvicorn:** install `requirements.txt` using the same Python executable used to start the app.
- **Port is already in use:** change `--port 8010` to `--port 8011`, then open that port in the browser.
- **Model not found:** keep the `model` folder beside `app.py`, or retrain using your datasets.
- **Model version warning:** use the pinned scikit-learn version in `requirements.txt`.
- **PowerShell blocks activation:** the commands above call the virtual environment's Python directly and do not require activation.

Only load trusted `.joblib` files: Python model serialization can execute code when loading a file. The demo has no authentication or rate limiting and allows cross-origin requests; keep it bound to localhost. Add access controls and review the CORS settings before exposing it to a network.
