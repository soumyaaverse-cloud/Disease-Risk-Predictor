# Disease Risk Predictor

A machine learning API that predicts diabetes risk from patient health metrics, built on the PIMA Indians Diabetes dataset. Wraps a trained Random Forest classifier in a lightweight Flask REST API for easy integration into larger systems.

## Overview

This project performs full exploratory data analysis on a real medical dataset, handles missing/invalid data (many "0" values are actually missing readings, not real measurements), engineers features, and benchmarks classifiers before settling on a Random Forest model achieving 82%+ accuracy. The trained model is served via a REST API so predictions can be requested programmatically.

## Features

- Full EDA and data cleaning pipeline (handles invalid zero-values via median imputation)
- Feature engineering (BMI categorization)
- Random Forest classifier tuned to avoid overfitting
- Confusion matrix and feature importance visualizations to understand *why* the model predicts what it predicts, not just its accuracy
- Flask REST API with JSON input/output
- Returns both a binary prediction and a risk probability score

## Tech Stack

- **Language:** Python
- **Data & Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **API:** Flask
- **Testing:** Postman

## Project Structure

```
disease-risk-predictor/
├── data/
│   └── diabetes.csv          # PIMA Diabetes dataset
├── model/
│   ├── train_model.py        # EDA, training, evaluation
│   ├── columns.pkl           # Expected feature columns (ensures consistent inference)
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── app.py                    # Flask API
├── model.pkl                 # Trained model
├── requirements.txt
└── README.md

```
## Setup

1. Clone the repo and navigate into it:
```bash
   git clone <your-repo-url>
   cd disease-risk-predictor
```

2. Create and activate a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Train the model (generates `model.pkl` and `model/columns.pkl`):
```bash
   python model/train_model.py
```

5. Run the API:
```bash
   python app.py
```

The API will be live at `http://127.0.0.1:5000`.

## Usage

Send a POST request to `/predict` with patient health metrics as JSON:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":2,"Glucose":130,"BloodPressure":70,"SkinThickness":25,"Insulin":100,"BMI":28.5,"DiabetesPedigreeFunction":0.5,"Age":33}'
```

**Response:**
```json
{
  "prediction": 0,
  "riskProbability": 0.307,
  "result": "Low risk of diabetes"
}
```

- `prediction`: 0 (low risk) or 1 (high risk)
- `riskProbability`: model's confidence the case is high-risk (0–1)
- `result`: human-readable summary

## Model Performance

- **Accuracy:** 82%+ on held-out test set
- Evaluated with confusion matrix and feature importance analysis (see `model/` folder) rather than accuracy alone, since class imbalance in medical datasets can make accuracy misleading on its own

## Key Design Decisions

- **Median imputation over dropping rows:** Zero values in fields like `BloodPressure` and `BMI` are medically impossible and represent missing data, not real readings. Imputing preserves dataset size instead of discarding valid patient records.
- **Stratified train/test split:** Ensures the class balance (diabetic vs. non-diabetic) is preserved in both sets, since the dataset is naturally imbalanced.
- **Saved column order (`columns.pkl`):** Guarantees inference-time input always matches the exact feature order used in training, preventing silent misprediction from column mismatches.

## Future Improvements

- Add input validation and a Swagger/OpenAPI spec
- Deploy behind a production WSGI server (e.g., Gunicorn) instead of Flask's dev server
- Add a simple frontend form for non-technical demoing
- Experiment with additional models (XGBoost, logistic regression) for comparison

## Author

Soumya Sarathe — B.Tech, NIT Agartala
[GitHub](https://github.com/soumyaaverse-cloud)