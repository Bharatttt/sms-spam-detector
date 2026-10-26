"""
Real-time SMS spam detection API + a minimal built-in web UI.

Run locally:
    uvicorn app:app --reload
Then open http://localhost:8000
"""
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from data_utils import preprocess_text

MODEL_PATH = Path(__file__).parent / 'model' / 'spam_pipeline.joblib'

app = FastAPI(
    title='SMS Spam Detector API',
    description='Real-time SMS ham/spam classification (TF-IDF + Logistic Regression, trained on 2,543 messages).',
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

pipeline = None


@app.on_event('startup')
def load_model():
    global pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f'Model not found at {MODEL_PATH}. Run `python train_model.py <path-to-csvs>` first.'
        )
    pipeline = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description='The SMS text to classify.')


class PredictResponse(BaseModel):
    message: str
    label: str
    is_spam: bool
    spam_probability: float
    ham_probability: float


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': pipeline is not None}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail='Model not loaded.')
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail='Message must not be empty.')

    proba = pipeline.predict_proba([preprocess_text(text)])[0]
    spam_index = list(pipeline.classes_).index(1)
    spam_p = float(proba[spam_index])
    is_spam = spam_p >= 0.5

    return PredictResponse(
        message=req.message,
        label='spam' if is_spam else 'ham',
        is_spam=is_spam,
        spam_probability=round(spam_p, 4),
        ham_probability=round(1 - spam_p, 4),
    )


@app.get('/', response_class=HTMLResponse)
def home():
    return HTML_PAGE


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMS Spam Detector</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f4f6f9; margin: 0; padding: 40px 16px;
    display: flex; justify-content: center;
  }
  .card {
    background: #fff; border-radius: 14px; box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    padding: 32px; width: 100%; max-width: 560px;
  }
  h1 { font-size: 22px; margin: 0 0 4px; color: #1a1a1a; }
  .subtitle { color: #6b7280; font-size: 14px; margin-bottom: 24px; }
  textarea {
    width: 100%; min-height: 110px; padding: 12px 14px; font-size: 15px;
    border: 1.5px solid #e2e5ea; border-radius: 10px; resize: vertical;
    font-family: inherit; outline: none; transition: border-color .15s;
  }
  textarea:focus { border-color: #4472C4; }
  button {
    margin-top: 14px; width: 100%; padding: 12px; font-size: 15px; font-weight: 600;
    background: #2E5597; color: #fff; border: none; border-radius: 10px; cursor: pointer;
    transition: background .15s;
  }
  button:hover { background: #234478; }
  button:disabled { background: #a9b7d1; cursor: not-allowed; }
  #result { margin-top: 22px; display: none; }
  .badge {
    display: inline-block; padding: 6px 16px; border-radius: 999px;
    font-weight: 700; font-size: 14px; letter-spacing: .03em;
  }
  .spam { background: #FDE2E1; color: #C0392B; }
  .ham { background: #E1F3E5; color: #1E8449; }
  .bar-row { margin-top: 16px; font-size: 13px; color: #444; }
  .bar-bg { background: #eef0f3; border-radius: 6px; height: 10px; margin-top: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 6px; }
  .fill-spam { background: #E74C3C; }
  .fill-ham { background: #27AE60; }
  .error { color: #C0392B; font-size: 14px; margin-top: 12px; display: none; }
  .examples { margin-top: 18px; font-size: 12px; color: #9aa1ab; }
  .examples span {
    display: inline-block; background: #f0f2f5; padding: 4px 10px; border-radius: 8px;
    margin: 3px 4px 0 0; cursor: pointer;
  }
  .examples span:hover { background: #e4e7ec; }
</style>
</head>
<body>
  <div class="card">
    <h1>SMS Spam Detector</h1>
    <div class="subtitle">Real-time classification &middot; TF-IDF + Logistic Regression model</div>

    <textarea id="msg" placeholder="Paste or type an SMS message here..."></textarea>
    <button id="btn" onclick="classify()">Check message</button>

    <div class="examples">
      Try: 
      <span onclick="fill(this)">WINNER!! You have been selected to receive a $1000 cash prize. Call now!</span>
      <span onclick="fill(this)">Hey, are we still on for dinner tonight at 7?</span>
    </div>

    <div class="error" id="error"></div>

    <div id="result">
      <span id="badge" class="badge"></span>
      <div class="bar-row">
        Spam probability: <span id="spamPct"></span>%
        <div class="bar-bg"><div id="spamBar" class="bar-fill fill-spam" style="width:0%"></div></div>
      </div>
      <div class="bar-row">
        Ham probability: <span id="hamPct"></span>%
        <div class="bar-bg"><div id="hamBar" class="bar-fill fill-ham" style="width:0%"></div></div>
      </div>
    </div>
  </div>

<script>
function fill(el) {
  document.getElementById('msg').value = el.textContent;
}

async function classify() {
  const msg = document.getElementById('msg').value;
  const btn = document.getElementById('btn');
  const errorBox = document.getElementById('error');
  const result = document.getElementById('result');
  errorBox.style.display = 'none';

  if (!msg.trim()) {
    errorBox.textContent = 'Please enter a message.';
    errorBox.style.display = 'block';
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Checking...';

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Request failed');
    }
    const data = await resp.json();

    const badge = document.getElementById('badge');
    badge.textContent = data.is_spam ? 'SPAM' : 'HAM (legitimate)';
    badge.className = 'badge ' + (data.is_spam ? 'spam' : 'ham');

    const spamPct = (data.spam_probability * 100).toFixed(1);
    const hamPct = (data.ham_probability * 100).toFixed(1);
    document.getElementById('spamPct').textContent = spamPct;
    document.getElementById('hamPct').textContent = hamPct;
    document.getElementById('spamBar').style.width = spamPct + '%';
    document.getElementById('hamBar').style.width = hamPct + '%';

    result.style.display = 'block';
  } catch (e) {
    errorBox.textContent = 'Error: ' + e.message;
    errorBox.style.display = 'block';
    result.style.display = 'none';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Check message';
  }
}
</script>
</body>
</html>
"""
