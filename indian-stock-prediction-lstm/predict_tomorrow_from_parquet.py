# predict_tomorrow_from_parquet.py
# Uses saved model + scaler + combined parquet to predict which stocks will go up tomorrow.

import numpy as np, pandas as pd, joblib, os
import tensorflow as tf
from tensorflow.keras.models import load_model

COMBINED = "nse_all_features.parquet"
SCALER = "sequences/scaler_90d.npy"   # contains {'mean':..., 'std':...}
MODEL = "models/lstm_best.h5"
SEQ_LEN = 90
FEATURE_COLS = [
    "ret1", "ret5", "logret",
    "sma10", "sma50", "ema12",
    "rsi14", "macd", "macd_signal",
    "bb_upper", "bb_lower",
    "obv"
]

# load resources
print("Loading model:", MODEL)
model = load_model(MODEL)
sc = np.load(SCALER, allow_pickle=True).item()
mean = sc["mean"].astype(np.float32)
std = sc["std"].astype(np.float32)

print("Loading combined parquet (may be large)...")
df = pd.read_parquet(COMBINED)

# iterate symbols and predict last SEQ_LEN rows
results = []
for sym, sub in df.groupby("symbol"):
    sub = sub.sort_values("date").reset_index(drop=True)
    if len(sub) < SEQ_LEN:
        continue
    last = sub.tail(SEQ_LEN)
    X = last[FEATURE_COLS].values.astype(np.float32)  # shape (SEQ_LEN, n_features)
    # scale using saved mean/std (broadcast)
    X_scaled = (X - mean.reshape((1,-1))) / std.reshape((1,-1))
    X_scaled = X_scaled.reshape((1, SEQ_LEN, len(FEATURE_COLS)))
    p = model.predict(X_scaled, verbose=0)[0,0]
    results.append((sym, float(p)))
# sort by probability descending
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
# show top candidates with prob > 0.5
up_preds = [(s,p) for s,p in results_sorted if p > 0.5]
print("Predicted UP candidates (probability > 0.5):")
for s,p in up_preds[:100]:
    print(f"{s}: {p:.3f}")
print("Total predicted UP:", len(up_preds))
