#!/usr/bin/env python3
# predict_and_save_filtered.py
# ---------------------------------------
# Predict tomorrow-up probabilities using your saved LSTM model,
# then filter results by liquidity (30-day avg volume / turnover),
# apply a probability threshold, and select top-K picks.
#
# Outputs:
#  - predictions_all.csv         (all symbols with probs + liquidity metrics)
#  - predictions_filtered.csv    (symbols passing filters: threshold & liquidity)
#  - predictions_topk.csv        (top-K by probability after filtering)
#
# Usage:
#   source venv311/bin/activate
#   python predict_and_save_filtered.py
# ---------------------------------------

# --- standard imports
import os                                          # file / path utilities
import numpy as np                                 # numeric arrays
import pandas as pd                                # dataframes and csv/parquet IO
from tensorflow.keras.models import load_model     # load saved Keras model

# --- Config (tweak these values as you like)
COMBINED_PQ = "nse_all_features.parquet"           # combined dataset produced earlier
SCALER_FILE = "sequences/scaler_90d.npy"           # saved scaler (dict with mean/std)
MODEL_CAND1 = "models/lstm_best_finetuned.h5"      # prefer fine-tuned model if present
MODEL_CAND2 = "models/lstm_best.h5"                # fallback
SEQ_LEN = 90                                       # must match training SEQ_LEN
FEATURE_COLS = [                                   # same features used in training
    "ret1", "ret5", "logret",
    "sma10", "sma50", "ema12",
    "rsi14", "macd", "macd_signal",
    "bb_upper", "bb_lower",
    "obv"
]

# Prediction filter hyperparameters (change these to be more/less aggressive)
PROB_THRESHOLD = 0.75         # require predicted prob >= this to keep
TOP_K = 20                    # after filtering, take top K picks (set to None to keep all)
MIN_AVG_VOLUME = 200000       # require 30-day average volume >= this (shares)
MIN_30D_TURNOVER = None       # in INR (turnover = price * volume). Set to e.g. 2e7 for 2 Crore. None = disable
MIN_LAST_CLOSE = 1.0          # filter tiny price stocks (< ₹1) if desired (set to 0 to disable)

OUT_ALL = "predictions_all.csv"
OUT_FILTERED = "predictions_filtered.csv"
OUT_TOPK = "predictions_topk.csv"

# --- basic sanity checks: files exist
if not os.path.exists(COMBINED_PQ):
    raise SystemExit(f"ERROR: combined parquet not found: {COMBINED_PQ}")

if not os.path.exists(SCALER_FILE):
    raise SystemExit(f"ERROR: scaler file not found: {SCALER_FILE}")

# choose model: prefer fine-tuned model if exists, else fallback
if os.path.exists(MODEL_CAND1):
    MODEL_PATH = MODEL_CAND1
elif os.path.exists(MODEL_CAND2):
    MODEL_PATH = MODEL_CAND2
else:
    raise SystemExit(f"ERROR: no model found. Expected {MODEL_CAND1} or {MODEL_CAND2}")

# --- load resources: model + scaler + combined parquet
model = load_model(MODEL_PATH)                      # load Keras model into memory
scaler = np.load(SCALER_FILE, allow_pickle=True).item()  # load dict {'mean':..., 'std':...}
mean = scaler["mean"].astype(np.float32)            # per-feature mean (float32)
std = scaler["std"].astype(np.float32)              # per-feature std (float32)

df = pd.read_parquet(COMBINED_PQ)                    # load combined dataset (may be large)
# ensure date is datetime and rows sorted for each symbol
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

# --- helper: compute 30-day liquidity stats per symbol (avg volume and avg turnover)
# compute last 30 trading days per symbol by grouping and taking tail(30), then aggregate
def compute_liquidity(df):
    """
    Returns a dataframe with columns: symbol, avg_vol_30, avg_turnover_30, last_close
    avg_vol_30 = mean of Volume over last 30 rows per symbol
    avg_turnover_30 = mean of (AdjClose * Volume) over last 30 rows per symbol
    last_close = last AdjClose for symbol
    """
    rows = []
    # iterate symbols (fast enough); if you want faster, use groupby + apply
    for sym, sub in df.groupby("symbol"):
        sub = sub.sort_values("date").tail(30)   # last 30 trading rows for symbol
        if sub.empty:
            continue
        avg_vol = float(sub["Volume"].mean()) if "Volume" in sub.columns else 0.0
        avg_turn = float((sub["AdjClose"] * sub["Volume"]).mean()) if ("AdjClose" in sub.columns and "Volume" in sub.columns) else 0.0
        last_close = float(sub["AdjClose"].iloc[-1]) if "AdjClose" in sub.columns else 0.0
        rows.append((sym, avg_vol, avg_turn, last_close))
    return pd.DataFrame(rows, columns=["symbol","avg_vol_30","avg_turnover_30","last_close"])

liq_df = compute_liquidity(df)                       # run liquidity computation
liq_df = liq_df.set_index("symbol")

# --- build predictions list (symbol, prob, last_date, liquidity metrics)
results = []
# iterate symbols and predict using the last SEQ_LEN rows for each
for sym, sub in df.groupby("symbol"):
    sub = sub.sort_values("date").reset_index(drop=True)     # sorted by date
    # skip if not enough history for sequence
    if len(sub) < SEQ_LEN:
        continue
    # extract last SEQ_LEN rows (most recent)
    last_block = sub.tail(SEQ_LEN)
    # pick features to feed model; if any feature missing, skip symbol
    missing = [c for c in FEATURE_COLS if c not in last_block.columns]
    if missing:
        # if features missing, skip symbol gracefully
        continue
    X = last_block[FEATURE_COLS].values.astype(np.float32)   # shape (SEQ_LEN, n_features)
    # scale using saved mean/std (broadcast across sequence/time)
    Xs = (X - mean.reshape((1,-1))) / std.reshape((1,-1))
    # reshape to model input: (1, SEQ_LEN, n_features)
    Xs = Xs.reshape((1, SEQ_LEN, len(FEATURE_COLS)))
    # predict probability that tomorrow_up == 1
    try:
        p = float(model.predict(Xs, verbose=0)[0,0])
    except Exception as e:
        # prediction failed for symbol (rare); skip
        continue
    # get liquidity metrics (if present), else set defaults
    if sym in liq_df.index:
        avg_vol = float(liq_df.loc[sym, "avg_vol_30"])
        avg_turn = float(liq_df.loc[sym, "avg_turnover_30"])
        last_close = float(liq_df.loc[sym, "last_close"])
    else:
        avg_vol = 0.0
        avg_turn = 0.0
        last_close = float(last_block["AdjClose"].iloc[-1])
    # append result tuple
    last_date = str(last_block["date"].iloc[-1].date())
    results.append((sym, p, last_date, avg_vol, avg_turn, last_close))

# convert to DataFrame and sort by probability desc
res_df = pd.DataFrame(results, columns=["symbol","prob","last_date","avg_vol_30","avg_turnover_30","last_close"])
res_df = res_df.sort_values("prob", ascending=False).reset_index(drop=True)

# write the unfiltered full prediction table
res_df.to_csv(OUT_ALL, index=False)
print("Wrote", OUT_ALL, "with", len(res_df), "rows")

# --- apply filters: liquidity + price + probability threshold
# boolean mask: start True, then apply conditions to filter in/out
mask = np.ones(len(res_df), dtype=bool)

# filter by 30-day average volume
if MIN_AVG_VOLUME is not None:
    mask = mask & (res_df["avg_vol_30"] >= float(MIN_AVG_VOLUME))

# filter by 30-day average turnover if configured
if MIN_30D_TURNOVER is not None:
    mask = mask & (res_df["avg_turnover_30"] >= float(MIN_30D_TURNOVER))

# filter by last close price (avoid sub-₹1 tickers if desired)
if MIN_LAST_CLOSE is not None and MIN_LAST_CLOSE > 0:
    mask = mask & (res_df["last_close"] >= float(MIN_LAST_CLOSE))

# filter by probability threshold
mask = mask & (res_df["prob"] >= float(PROB_THRESHOLD))

# filtered DataFrame
filtered_df = res_df[mask].reset_index(drop=True)
filtered_df.to_csv(OUT_FILTERED, index=False)
print("Wrote", OUT_FILTERED, "with", len(filtered_df), "rows (after applying filters)")

# --- take top-K after filtering (if TOP_K is set), else keep all filtered rows
if TOP_K is not None:
    topk_df = filtered_df.head(TOP_K).reset_index(drop=True)
else:
    topk_df = filtered_df.copy()

topk_df.to_csv(OUT_TOPK, index=False)
print("Wrote", OUT_TOPK, "with", len(topk_df), "rows (TOP_K selection)")

# print short summary to console (first 40 rows)
print("\nTop picks (symbol, prob, avg_vol_30, last_close):")
print(topk_df[["symbol","prob","avg_vol_30","last_close"]].head(40).to_string(index=False))

# done
print("\nFinished. Tune PROB_THRESHOLD / MIN_AVG_VOLUME / TOP_K in the script to be more/less selective.")
