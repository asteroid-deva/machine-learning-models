# build_sequences_and_save.py
# Builds 90-day sequences for ML and saves train/val/test + scaler.
# Run this after download_and_prepare_data.py has produced nse_all_features.parquet.
#
# Usage:
#   source ~/nse_massive_project/venv311/bin/activate
#   python build_sequences_and_save.py

import os                           # file path ops
import numpy as np                  # numeric arrays
import pandas as pd                 # dataframes
from tqdm import tqdm               # progress bar
from sklearn.preprocessing import StandardScaler
import joblib                       # save scaler

# ---------------------
# Config
# ---------------------
COMBINED_PATH = "nse_all_features.parquet"   # input combined dataset
OUT_DIR = "sequences"                        # folder to save outputs
SEQ_LEN = 90                                 # number of past days in each sequence (you chose 90)
FEATURE_COLS = [                             
    "ret1", "ret5", "logret",
    "sma10", "sma50", "ema12",
    "rsi14", "macd", "macd_signal",
    "bb_upper", "bb_lower",
    "obv"
]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15   # test will be 1 - (train+val) = 0.15

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------
# Load dataset
# ---------------------
print("Loading combined dataset from:", COMBINED_PATH)
df = pd.read_parquet(COMBINED_PATH)

# ensure ordering by symbol + date
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

# ---------------------
# Create the binary target:
# target at row i = 1 if AdjClose at i+1 > AdjClose at i (i.e., tomorrow > today)
# We compute per symbol to avoid leaking across symbols.
# ---------------------
print("Creating binary target (tomorrow_up: 1 if next day's close > today's close)...")
df["tomorrow_close"] = df.groupby("symbol")["AdjClose"].shift(-1)   # next day's close (per symbol)
df["target"] = (df["tomorrow_close"] > df["AdjClose"]).astype(int)  # 1 if tomorrow is higher
# drop rows where tomorrow_close is NaN (last row per symbol cannot be labeled)
df = df.dropna(subset=["target"]).reset_index(drop=True)
df["target"] = df["target"].astype(np.int8)

# ---------------------
# Build sequences per symbol
# For each symbol, for i in [0 .. len(symbol)-SEQ_LEN-1]:
#   X = features[i : i+SEQ_LEN]
#   y = target at index i+SEQ_LEN-1 (target that corresponds to "day after last day of sequence")
# ---------------------
X_list = []
y_list = []
meta_rows = []   # optional metadata (symbol, date_of_label)

print("Building sequences with SEQ_LEN =", SEQ_LEN)
symbols = df["symbol"].unique()
for sym in tqdm(symbols, desc="Symbols", ncols=120):
    sub = df[df["symbol"] == sym].reset_index(drop=True)
    if len(sub) <= SEQ_LEN:
        # not enough rows to build a single sequence
        continue

    # extract feature matrix and labels for this symbol
    feats = sub[FEATURE_COLS].values.astype(np.float32)
    labels = sub["target"].values.astype(np.int8)
    dates = sub["date"].values  # for meta: date of each sample

    # iterate possible sequence start indices
    # For start i, sequence covers rows [i .. i+SEQ_LEN-1] and label we want is labels[i+SEQ_LEN-1]
    for i in range(0, len(sub) - SEQ_LEN):
        seq = feats[i : i + SEQ_LEN]                # shape = (SEQ_LEN, n_features)
        lab = labels[i + SEQ_LEN - 1]               # label for next-day after sequence end
        label_date = dates[i + SEQ_LEN - 1]         # date corresponding to that label
        X_list.append(seq)
        y_list.append(lab)
        meta_rows.append((sym, str(label_date)))

# convert to numpy arrays
X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int8)

print("Total sequences built:", X.shape, y.shape)
if X.size == 0:
    raise SystemExit("No sequences were built. Check SEQ_LEN and your combined dataset.")

# ---------------------
# Train / Val / Test split (time-ordered: we do a simple global split by index)
# ---------------------
N = len(X)
train_end = int(N * TRAIN_RATIO)
val_end = int(N * (TRAIN_RATIO + VAL_RATIO))

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

print("Train / Val / Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

# ---------------------
# Scale features using StandardScaler fit only on training data
# We fit scaler on flattened training data: (samples*SEQ_LEN, n_features)
# Then we transform each split and restore sequence shape.
# ---------------------
n_features = X.shape[2]
scaler = StandardScaler()

# flatten training for fit
X_train_flat = X_train.reshape(-1, n_features)
scaler.fit(X_train_flat)

def scale_sequences(X_block):
    """
    X_block: (num_samples, SEQ_LEN, n_features)
    returns scaled block with same shape
    """
    orig_shape = X_block.shape
    flat = X_block.reshape(-1, orig_shape[2])
    scaled_flat = scaler.transform(flat)
    return scaled_flat.reshape(orig_shape)

X_train_scaled = scale_sequences(X_train)
X_val_scaled = scale_sequences(X_val)
X_test_scaled = scale_sequences(X_test)

# ---------------------
# Save outputs: compressed NPZ + scaler object
# ---------------------
out_npz = os.path.join(OUT_DIR, "data_sequences_90d.npz")
out_scaler = os.path.join(OUT_DIR, "scaler_90d.joblib")

print("Saving sequences to:", out_npz)
np.savez_compressed(
    out_npz,
    X_train=X_train_scaled,
    y_train=y_train,
    X_val=X_val_scaled,
    y_val=y_val,
    X_test=X_test_scaled,
    y_test=y_test
)

# save scaler for later use at training/inference time
joblib.dump(scaler, out_scaler)

# save simple metadata CSV for reference (optional)
meta_df = pd.DataFrame(meta_rows, columns=["symbol", "label_date"])
meta_df.to_csv(os.path.join(OUT_DIR, "meta_index_90d.csv"), index=False)

print("✅ Sequences saved in:", OUT_DIR)
print(" -", out_npz)
print(" -", out_scaler)
print(" -", os.path.join(OUT_DIR, "meta_index_90d.csv"))
print("Finished.")

