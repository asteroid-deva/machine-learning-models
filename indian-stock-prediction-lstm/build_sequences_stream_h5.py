"""
build_sequences_stream_h5.py

Stream-build 90-day sequences from nse_all_features.parquet into a disk-backed HDF5 file
to avoid memory allocation errors. Then compute scaler from the training split and write
scaled train/val/test datasets back to HDF5 in chunked form (float16).

Outputs:
  - sequences/sequences_raw.h5        (datasets: X (float32), y (int8), meta (bytes))
  - sequences/sequences_scaled.h5     (datasets: X_train, y_train, X_val, y_val, X_test, y_test)
  - sequences/scaler_90d.npy          (mean & std arrays for features)
  - sequences/meta_index_90d.csv      (symbol, label_date rows in order)
"""

import os
import math
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

# Config
COMBINED_PATH = "nse_all_features.parquet"
OUT_DIR = "sequences"
SEQ_LEN = 90
FEATURE_COLS = [
    "ret1", "ret5", "logret",
    "sma10", "sma50", "ema12",
    "rsi14", "macd", "macd_signal",
    "bb_upper", "bb_lower",
    "obv"
]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
DTYPE_X = np.float32   # stored raw sequences
DTYPE_X_SCALED = np.float16  # scaled for training files (saves disk)
BATCH_RESIZE = 1024    # chunk size used when appending to HDF5

os.makedirs(OUT_DIR, exist_ok=True)
RAW_H5 = os.path.join(OUT_DIR, "sequences_raw.h5")
SCALED_H5 = os.path.join(OUT_DIR, "sequences_scaled.h5")
SCALER_FILE = os.path.join(OUT_DIR, "scaler_90d.npy")
META_CSV = os.path.join(OUT_DIR, "meta_index_90d.csv")

# 1) Load combined parquet
print("Loading combined dataset...")
df = pd.read_parquet(COMBINED_PATH)
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
print("Loaded rows:", len(df))

# 2) Build binary target (tomorrow up): same as before
df["tomorrow_close"] = df.groupby("symbol")["AdjClose"].shift(-1)
df["target"] = (df["tomorrow_close"] > df["AdjClose"]).astype(int)
df = df.dropna(subset=["target"]).reset_index(drop=True)
df["target"] = df["target"].astype(np.int8)

# 3) Prepare HDF5 for raw sequences (we will append)
print("Preparing raw HDF5:", RAW_H5)
if os.path.exists(RAW_H5):
    os.remove(RAW_H5)
h5f = h5py.File(RAW_H5, "w")

# create extendable datasets
# X: shape (0, SEQ_LEN, n_features), maxshape (None, SEQ_LEN, n_features)
n_features = len(FEATURE_COLS)
h5_X = h5f.create_dataset(
    "X",
    shape=(0, SEQ_LEN, n_features),
    maxshape=(None, SEQ_LEN, n_features),
    dtype=DTYPE_X,
    chunks=(min(BATCH_RESIZE, 1024), SEQ_LEN, n_features),
    compression="gzip"
)
h5_y = h5f.create_dataset("y", shape=(0,), maxshape=(None,), dtype=np.int8, chunks=(BATCH_RESIZE,), compression="gzip")

# meta: store symbol and label_date as fixed-length bytes (or variable-length)
dt = h5py.string_dtype(encoding="utf-8")
h5_meta = h5f.create_dataset("meta", shape=(0,2), maxshape=(None,2), dtype=dt, chunks=(BATCH_RESIZE,2), compression="gzip")

# helper to append arrays to h5 dataset
def append_to_dataset(ds, values):
    """Append values along first axis to dataset ds (h5py)."""
    cur = ds.shape[0]
    add = values.shape[0]
    ds.resize(cur + add, axis=0)
    ds[cur:cur+add] = values

# 4) Stream sequences per symbol into HDF5
print("Streaming sequences (SEQ_LEN={}):".format(SEQ_LEN))
symbols = df["symbol"].unique()
meta_rows = []
append_buffer_X = []
append_buffer_y = []
append_buffer_meta = []

total = 0
for sym in tqdm(symbols, desc="Symbols", ncols=120):
    sub = df[df["symbol"] == sym].reset_index(drop=True)
    if len(sub) <= SEQ_LEN:
        continue
    feats = sub[FEATURE_COLS].values.astype(DTYPE_X)
    labels = sub["target"].values.astype(np.int8)
    dates = sub["date"].values
    # create sequences for this symbol
    for i in range(0, len(sub) - SEQ_LEN):
        seq = feats[i : i + SEQ_LEN]  # (SEQ_LEN, n_features)
        lab = labels[i + SEQ_LEN - 1]  # label for the day after last day in seq
        label_date = str(dates[i + SEQ_LEN - 1])
        append_buffer_X.append(seq)
        append_buffer_y.append(lab)
        append_buffer_meta.append((sym, label_date))
        total += 1
        # flush buffer in batches
        if len(append_buffer_X) >= BATCH_RESIZE:
            arrX = np.stack(append_buffer_X, axis=0)
            arry = np.array(append_buffer_y, dtype=np.int8)
            arrmeta = np.array(append_buffer_meta, dtype=object)
            # meta must be shape (N,2)
            arrmeta2 = np.empty((arrmeta.shape[0], 2), dtype=object)
            arrmeta2[:,0] = arrmeta[:,0]
            arrmeta2[:,1] = arrmeta[:,1]
            append_to_dataset(h5_X, arrX)
            append_to_dataset(h5_y, arry)
            append_to_dataset(h5_meta, arrmeta2)
            append_buffer_X.clear(); append_buffer_y.clear(); append_buffer_meta.clear()
# flush remaining
if len(append_buffer_X) > 0:
    arrX = np.stack(append_buffer_X, axis=0)
    arry = np.array(append_buffer_y, dtype=np.int8)
    arrmeta = np.array(append_buffer_meta, dtype=object)
    arrmeta2 = np.empty((arrmeta.shape[0], 2), dtype=object)
    arrmeta2[:,0] = arrmeta[:,0]
    arrmeta2[:,1] = arrmeta[:,1]
    append_to_dataset(h5_X, arrX)
    append_to_dataset(h5_y, arry)
    append_to_dataset(h5_meta, arrmeta2)
    append_buffer_X.clear(); append_buffer_y.clear(); append_buffer_meta.clear()

h5f.flush()
N = h5_X.shape[0]
print(f"Total sequences written: {N:,}")
# save meta CSV (read back h5_meta)
meta_arr = np.array(h5_meta[:], dtype=object)
pd.DataFrame(meta_arr, columns=["symbol","label_date"]).to_csv(META_CSV, index=False)

# 5) Split indices for train/val/test
train_end = int(N * TRAIN_RATIO)
val_end = int(N * (TRAIN_RATIO + VAL_RATIO))
print("Train/Val/Test indices:", 0, train_end, val_end, N)

# 6) Compute scaler (mean & std) over flattened training data in chunks
# We'll compute mean and variance using two-pass or Welford-style. Here we do incremental sums.
print("Computing mean & std on training data (streaming)...")
sum_vec = np.zeros(n_features, dtype=np.float64)
sum_sq_vec = np.zeros(n_features, dtype=np.float64)
count = 0
CHUNK = 4096
start = 0
while start < train_end:
    end = min(start + CHUNK, train_end)
    block = h5_X[start:end]  # shape (block_N, SEQ_LEN, n_features)
    flat = block.reshape(-1, n_features)  # (block_N*SEQ_LEN, n_features)
    sum_vec += flat.sum(axis=0)
    sum_sq_vec += (flat ** 2).sum(axis=0)
    count += flat.shape[0]
    start = end
mean = sum_vec / count
var = (sum_sq_vec / count) - (mean ** 2)
std = np.sqrt(np.maximum(var, 1e-12))
print("Training samples (flattened):", count)
print("Mean shape:", mean.shape, "Std shape:", std.shape)
# save scaler
np.save(SCALER_FILE, {"mean": mean.astype(np.float32), "std": std.astype(np.float32)})
print("Saved scaler to", SCALER_FILE)

# 7) Create scaled datasets file and write scaled splits in chunks (to save RAM we stream)
# Remove existing scaled file if any
if os.path.exists(SCALED_H5):
    os.remove(SCALED_H5)
sf = h5py.File(SCALED_H5, "w")
# shapes for splits
n_train = train_end
n_val = max(0, val_end - train_end)
n_test = max(0, N - val_end)
# datasets: X_train, X_val, X_test (float16), y_train/y_val/y_test (int8)
sf.create_dataset("X_train", shape=(n_train, SEQ_LEN, n_features), dtype=DTYPE_X_SCALED,
                  chunks=(min(512, n_train) or 1, SEQ_LEN, n_features), compression="gzip")
sf.create_dataset("y_train", shape=(n_train,), dtype=np.int8, compression="gzip")
sf.create_dataset("X_val", shape=(n_val, SEQ_LEN, n_features), dtype=DTYPE_X_SCALED,
                  chunks=(min(512, max(1,n_val)), SEQ_LEN, n_features), compression="gzip")
sf.create_dataset("y_val", shape=(n_val,), dtype=np.int8, compression="gzip")
sf.create_dataset("X_test", shape=(n_test, SEQ_LEN, n_features), dtype=DTYPE_X_SCALED,
                  chunks=(min(512, max(1,n_test)), SEQ_LEN, n_features), compression="gzip")
sf.create_dataset("y_test", shape=(n_test,), dtype=np.int8, compression="gzip")

# helper to scale blocks and write
def scale_and_write(src_h5, dst_ds, dst_y, idx_src_start, idx_src_end, dst_idx_start, mean, std):
    """
    Read from src_h5['X'][idx_src_start:idx_src_end], scale using (mean,std), write to dst_ds[dst_idx_start:...],
    and write labels to dst_y similarly.
    """
    block = src_h5['X'][idx_src_start:idx_src_end]  # float32
    # scale: (x - mean) / std  ; mean/std are per-feature for flattened features -> we broadcast across seq/time
    # shape block: (B, SEQ_LEN, n_features) ; mean shape (n_features,)
    scaled = (block - mean.reshape((1,1,-1))) / std.reshape((1,1,-1))
    # cast to float16 to save disk
    scaled16 = scaled.astype(DTYPE_X_SCALED)
    dst_ds[dst_idx_start:dst_idx_start + scaled16.shape[0], :, :] = scaled16
    # labels
    dst_y[dst_idx_start:dst_idx_start + scaled16.shape[0]] = src_h5['y'][idx_src_start:idx_src_end]

# Write training split in chunks
print("Writing scaled train/val/test splits to:", SCALED_H5)
CH = 2048
dst_i = 0
for s in range(0, train_end, CH):
    e = min(train_end, s + CH)
    scale_and_write(h5f, sf['X_train'], sf['y_train'], s, e, dst_i, mean, std)
    dst_i += (e - s)

# validation
dst_i = 0
for s in range(train_end, val_end, CH):
    e = min(val_end, s + CH)
    scale_and_write(h5f, sf['X_val'], sf['y_val'], s, e, dst_i, mean, std)
    dst_i += (e - s)

# test
dst_i = 0
for s in range(val_end, N, CH):
    e = min(N, s + CH)
    scale_and_write(h5f, sf['X_test'], sf['y_test'], s, e, dst_i, mean, std)
    dst_i += (e - s)

sf.flush()
sf.close()
h5f.close()

print("Finished. Scaled datasets saved to:", SCALED_H5)
print("Meta CSV:", META_CSV)
print("Scaler file:", SCALER_FILE)
