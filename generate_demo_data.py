"""Generate synthetic teaching data, not actual student observations."""
from pathlib import Path
import numpy as np
import pandas as pd
rng = np.random.default_rng(2026)
n = 1800
study = np.round(rng.uniform(0, 10, n), 1)
attendance = rng.integers(35, 101, n)
marks = rng.integers(20, 101, n)
sleep = np.round(rng.uniform(3.5, 10, n), 1)
score = 0.26*(study/10) + 0.22*(attendance/100) + 0.44*(marks/100) + 0.08*(1-np.minimum(np.abs(sleep-7.5)/7.5,1))
score += rng.normal(0, .065, n)
result = (score >= .56).astype(int)
df = pd.DataFrame({'study_hours':study,'attendance':attendance,'previous_marks':marks,'sleep_hours':sleep,'result':result})
out = Path(__file__).resolve().parent / 'data' / 'students.csv'
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f'Generated {len(df)} SYNTHETIC records at {out}')
