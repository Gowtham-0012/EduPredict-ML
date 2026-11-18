# EduPredict — Student Performance Predictor

An attractive, responsive Streamlit machine learning demo for InternCircle Project 05.

> **Dataset warning:** This repository includes **synthetic, algorithmically generated** student-like data for demonstrating the end-to-end workflow. Results are **not validated for real students** and should not be used for academic decisions.

## Features
- Modern dark UI with gradient hero, responsive layout, sliders and charts
- Random Forest classification, probability estimate, feature importance
- `joblib` model persistence and Streamlit resource caching
- Input validation, friendly error messages, no personal data collection

## Quick start (Python 3.10–3.13 recommended)

```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_demo_data.py
python train_model.py
python -m streamlit run app.py
```

The ZIP includes generated data, model.joblib and model_info.json, so you can run the app immediately after installing dependencies. The generation and training commands are included for reproducibility.

## Deploy
1. Upload the project to a public GitHub repository. Include `model.joblib`, `model_info.json`, `requirements.txt`, `app.py`, and `.streamlit/config.toml`.
2. Open https://share.streamlit.io and select **Create app** from GitHub.
3. Select repository, `main` branch and `app.py`, then deploy.
4. Test the public link on desktop and mobile.

## Internship submission
- Streamlit Community Cloud app URL
- GitHub repository URL
- Screenshot showing a prediction

## Model details
Four numerical features: `study_hours`, `attendance`, `previous_marks`, `sleep_hours`. Binary result labels: 0 (needs improvement), 1 (likely pass). Held-out accuracy is computed on **synthetic** test records only. Do not interpret as real-world accuracy.
