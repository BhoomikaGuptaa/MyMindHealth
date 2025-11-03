# MindMend — Emotion Insights (Non-clinical)

Quick Streamlit app that classifies text into emotions using a pretrained DistilRoBERTa model.
**Not medical advice.**

## Run (Windows, Command Prompt)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
streamlit run src\app.py
```

## Run (macOS/Linux)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch
streamlit run src/app.py
```
