from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
FEATURES = ['study_hours', 'attendance', 'previous_marks', 'sleep_hours']
df = pd.read_csv(ROOT / 'data' / 'students.csv')
required = FEATURES + ['result']
if not set(required).issubset(df.columns):
    raise ValueError(f'Missing columns: {set(required)-set(df.columns)}')
if df[required].isna().any().any():
    raise ValueError('Missing values in training data')
if not set(df['result'].unique()).issubset({0, 1}) or df['result'].nunique() != 2:
    raise ValueError('Target must contain both 0 and 1')
X, y = df[FEATURES], df['result'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=180, min_samples_leaf=3, random_state=42)
model.fit(X_train, y_train)
score = accuracy_score(y_test, model.predict(X_test))
joblib.dump(model, ROOT / 'model.joblib')
metadata = {'features': FEATURES, 'accuracy': round(float(score), 4), 'samples': len(df), 'dataset': 'Synthetic demonstration data', 'class_labels': {'0': 'Needs improvement', '1': 'Likely pass'}}
(ROOT / 'model_info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Model saved. Holdout accuracy:', round(score, 4))
print(classification_report(y_test, model.predict(X_test)))
