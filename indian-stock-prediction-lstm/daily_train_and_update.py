#!/usr/bin/env python3
# daily_train_and_update.py
# Safe daily update + fine-tune pipeline.
#
# What it does (high level):
# 1) downloads last N days (default 7) for each symbol (fast)
# 2) appends truly new rows (dates > existing max) and computes indicators for them
# 3) builds sequences that have newly available labels (i.e., rows where tomorrow exists)
# 4) samples a small historical set from sequences/sequences_scaled.h5 as "replay" to avoid forgetting
# 5) creates a mixed train set (mostly new sequences + small historical sample)
# 6) backs up current model, fine-tunes gently (small LR, 1-3 epochs)
# 7) evaluates validation AUC before/after and rolls back if val AUC drops
# 8) saves timestamped model backup and overwrites main model only if improved (or forced)
#
# Usage:
#   source venv311/bin/activate
#   python daily_train_and_update.py
#
# Notes:
# - This script is conservative (keeps backups). Check printed messages.
# - Tune HYPERPARAMS below if you want different behaviour.
# -----------------------------------------

import os, sys, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import shutil
import random

# TensorFlow import will be done later to keep early checks fast
# ------------------- HYPERPARAMS -------------------
COMBINED = "nse_all_features.parquet"           # combined parquet
SCALER = "sequences/scaler_90d.npy"             # scaler (dict {"mean":..., "std":...})
MODEL_MAIN = "models/lstm_best.h5"              # main model to fine-tune (will be backed-up)
MODEL_DIR = "models"                             # where to write backups
SEQ_LEN = 90
FEATURE_COLS = [
    "ret1","ret5","logret",
    "sma10","sma50","ema12",
    "rsi14","macd","macd_signal",
    "bb_upper","bb_lower","obv"
]
DOWNLOAD_DAYS = 7           # download last 7 calendar days per symbol
EPOCHS = 2                  # small number of epochs for fine-tuning
BATCH_SIZE = 256            # adjust if OOM
LR = 1e-4                   # low LR for safe fine-tuning
REPLAY_RATIO = 0.2          # fraction of training samples that should be historical replay (0.2 = 20%)
HIST_SAMPLE_MAX = 50_000    # max historical samples to pull for replay (caps memory)
MIN_NEW_SEQ = 50            # require at least this many new sequences to proceed fine-tune
VERBOSE = True
# -------------------------------------------------

# quick existence checks
if not os.path.exists(COMBINED):
    sys.exit(f"ERROR: combined file not found: {COMBINED}")

if not os.path.exists(MODEL_MAIN):
    sys.exit(f"ERROR: model file not found: {MODEL_MAIN} — train a model first")

# load combined dataset
if VERBOSE: print("Loading combined parquet (may be large)...")
df_comb = pd.read_parquet(COMBINED)
df_comb["date"] = pd.to_datetime(df_comb["date"])
df_comb = df_comb.sort_values(["symbol","date"]).reset_index(drop=True)

# get symbol list and last date per symbol
symbols = sorted(df_comb["symbol"].unique().tolist())
last_date = df_comb.groupby("symbol")["date"].max().to_dict()

# helper to flatten yfinance MultiIndex columns if needed
def flatten_yf(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(p) for p in col if p]) for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

collected = []
SLEEP = 0.12

# download short ranges for each symbol and compute indicators for appended window
if VERBOSE: print("Downloading last", DOWNLOAD_DAYS, "days for", len(symbols), "symbols...")
for i, sym in enumerate(symbols):
    ticker = sym + ".NS"
    try:
        period_str = f"{DOWNLOAD_DAYS}d"
        raw = yf.download(ticker, period=period_str, interval="1d", progress=False, threads=False)
    except Exception as e:
        if VERBOSE: print(f"[{i+1}/{len(symbols)}] Download error for {sym}: {e}")
        time.sleep(SLEEP)
        continue
    if raw is None or raw.empty:
        time.sleep(SLEEP)
        continue
    raw = flatten_yf(raw)
    # find close/vol
    close_col = next((c for c in raw.columns if "close" in c.lower()), None)
    vol_col = next((c for c in raw.columns if "volume" in c.lower()), None)
    if close_col is None:
        time.sleep(SLEEP)
        continue
    raw = raw.reset_index()
    raw = raw.rename(columns={close_col: "AdjClose"})
    if vol_col is None:
        raw["Volume"] = 0.0
    else:
        raw = raw.rename(columns={vol_col: "Volume"})
    # unify date column
    if "Date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["Date"])
    else:
        raw["date"] = pd.to_datetime(raw.index)
    raw["symbol"] = sym
    # select only rows newer than we have
    last = last_date.get(sym, pd.Timestamp("1900-01-01"))
    new_rows = raw[raw["date"] > pd.to_datetime(last)].copy()
    if new_rows.empty:
        time.sleep(SLEEP)
        continue
    # compute features by concatenating a tail of historic rows + new_rows to compute indicators cleanly
    tail_need = max(SEQ_LEN, 60) + 10
    hist_tail = df_comb[df_comb["symbol"]==sym].sort_values("date").tail(tail_need)
    tmp = pd.concat([hist_tail, new_rows], ignore_index=True, sort=False).sort_values("date").reset_index(drop=True)
    # numeric conversions
    tmp["AdjClose"] = pd.to_numeric(tmp["AdjClose"], errors="coerce")
    tmp["Volume"] = pd.to_numeric(tmp["Volume"], errors="coerce").fillna(0.0)
    tmp["ret1"] = tmp["AdjClose"].pct_change(1).fillna(0)
    tmp["ret5"] = tmp["AdjClose"].pct_change(5).fillna(0)
    tmp["logret"] = np.log(tmp["AdjClose"]).diff().fillna(0)
    # compute TA indicators (local import)
    try:
        from ta.trend import SMAIndicator, EMAIndicator, MACD
        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands
        from ta.volume import OnBalanceVolumeIndicator

        tmp["sma10"] = SMAIndicator(tmp["AdjClose"], window=10).sma_indicator()
        tmp["sma50"] = SMAIndicator(tmp["AdjClose"], window=50).sma_indicator()
        tmp["ema12"] = EMAIndicator(tmp["AdjClose"], window=12).ema_indicator()
        tmp["rsi14"] = RSIIndicator(tmp["AdjClose"], window=14).rsi()
        macd = MACD(tmp["AdjClose"]); tmp["macd"] = macd.macd(); tmp["macd_signal"] = macd.macd_signal()
        bb = BollingerBands(tmp["AdjClose"], window=20, window_dev=2)
        tmp["bb_upper"] = bb.bollinger_hband(); tmp["bb_lower"] = bb.bollinger_lband()
        tmp["obv"] = OnBalanceVolumeIndicator(tmp["AdjClose"], tmp["Volume"]).on_balance_volume()
    except Exception as e:
        if VERBOSE: print(f"[{sym}] TA error: {e}")
        for c in ("sma10","sma50","ema12","rsi14","macd","macd_signal","bb_upper","bb_lower","obv"):
            tmp[c] = tmp.get(c, np.nan)
    # fill small gaps as pipeline does
    tmp = tmp.fillna(method="bfill").fillna(method="ffill").fillna(0.0)
    # pick only new rows
    new_ready = tmp[tmp["date"] > pd.to_datetime(last)].copy()
    if not new_ready.empty:
        collected.append(new_ready)
        if VERBOSE: print(f"[{i+1}/{len(symbols)}] {sym} new rows:", len(new_ready))
    time.sleep(SLEEP)

if len(collected) == 0:
    print("No new rows found. Exiting.")
    sys.exit(0)

df_new = pd.concat(collected, ignore_index=True, sort=False)
df_new = df_new.sort_values(["symbol","date"]).reset_index(drop=True)
# compute tomorrow label for new rows
df_new["tomorrow_close"] = df_new.groupby("symbol")["AdjClose"].shift(-1)
df_new["target"] = (df_new["tomorrow_close"] > df_new["AdjClose"]).astype(int)

# Build sequences ending on new rows that have labels
seqs_X = []
seqs_y = []
meta = []
for sym in df_new["symbol"].unique():
    # build a combined historic+new series for the symbol
    hist = pd.concat([df_comb[df_comb["symbol"]==sym], df_new[df_new["symbol"]==sym]], ignore_index=True, sort=False)
    hist = hist.sort_values("date").reset_index(drop=True)
    # find rows corresponding to new region with valid label
    new_dates = set(df_new[df_new["symbol"]==sym]["date"].dt.normalize().tolist())
    for idx in range(len(hist)):
        if hist.loc[idx, "date"].normalize() in new_dates:
            # ensure label exists (tomorrow present)
            if idx+1 >= len(hist):
                continue
            label = 1 if hist.loc[idx+1,"AdjClose"] > hist.loc[idx,"AdjClose"] else 0
            start = idx - (SEQ_LEN - 1)
            if start < 0:
                continue
            seq = hist.loc[start:idx, FEATURE_COLS].values
            if seq.shape != (SEQ_LEN, len(FEATURE_COLS)):
                continue
            seqs_X.append(seq.astype(np.float32))
            seqs_y.append(int(label))
            meta.append((sym, str(hist.loc[idx,"date"])))
# require minimal number of sequences to continue
if len(seqs_X) < MIN_NEW_SEQ:
    print(f"Only {len(seqs_X)} new sequences found (< {MIN_NEW_SEQ}). Skipping fine-tune to avoid noisy updates.")
    # still append new rows to parquet but do not fine-tune
    to_add = df_new.drop(columns=["tomorrow_close","target"], errors="ignore")
    df_full = pd.concat([df_comb, to_add], ignore_index=True, sort=False).drop_duplicates(subset=["symbol","date"], keep="last")
    df_full = df_full.sort_values(["symbol","date"]).reset_index(drop=True)
    df_full.to_parquet(COMBINED)
    print("Appended new rows to combined parquet. Exiting.")
    sys.exit(0)

X_new = np.stack(seqs_X, axis=0)   # shape (N_new, SEQ_LEN, F)
y_new = np.array(seqs_y, dtype=np.int8)

if VERBOSE:
    print("New sequences:", X_new.shape, "labels:", y_new.shape)
    print("Meta sample:", meta[:3])

# Load scaler and scale new sequences
if os.path.exists(SCALER):
    sc = np.load(SCALER, allow_pickle=True).item()
    mean = sc["mean"].astype(np.float32)
    std = sc["std"].astype(np.float32)
    X_new_scaled = (X_new - mean.reshape((1,1,-1))) / std.reshape((1,1,-1))
else:
    print("Warning: scaler not found. Using raw features (NOT recommended).")
    X_new_scaled = X_new

# Try to sample historical sequences for replay from sequences/sequences_scaled.h5 (if present)
hist_X = None
hist_y = None
h5_path = "sequences/sequences_scaled.h5"
if os.path.exists(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as fh:
        # check for dataset names; pick 'train' or 'x_train' common keys
        if "train" in fh:
            dset = fh["train"]
            n_hist = dset.shape[0]
            # we will sample min(HIST_SAMPLE_MAX, n_hist) indices
            sample_n = min(int(len(X_new_scaled)*REPLAY_RATIO), HIST_SAMPLE_MAX, n_hist)
            if sample_n > 0:
                idxs = np.random.choice(n_hist, sample_n, replace=False)
                hist_X = dset[idxs]        # shape (sample_n, SEQ_LEN, F)
                # attempt to find labels in companion dataset 'y_train' or under 'train_labels' etc.
                if "train_labels" in fh:
                    hist_y = fh["train_labels"][idxs]
                elif "y_train" in fh:
                    hist_y = fh["y_train"][idxs]
                elif "labels" in fh:
                    hist_y = fh["labels"][idxs]
                else:
                    hist_y = None
                if VERBOSE:
                    print("Sampled", sample_n, "historical sequences for replay from", h5_path)
        elif "x_train" in fh:
            dset = fh["x_train"]
            n_hist = dset.shape[0]
            sample_n = min(int(len(X_new_scaled)*REPLAY_RATIO), HIST_SAMPLE_MAX, n_hist)
            if sample_n > 0:
                idxs = np.random.choice(n_hist, sample_n, replace=False)
                hist_X = dset[idxs]
                hist_y = fh["y_train"][idxs] if "y_train" in fh else None
                if VERBOSE:
                    print("Sampled", sample_n, "historical sequences for replay from", h5_path)
        else:
            if VERBOSE: print("No recognized dataset keys in", h5_path, " — skipping historical replay.")
else:
    if VERBOSE: print("No sequences HDF5 found at", h5_path, " — skipping historical replay.")

# combine new + historical replay (if available)
if hist_X is not None and hist_y is not None:
    # scale historical sequences may already be scaled in file; assume the sequences file is already scaled.
    # To avoid mixing different scales, we assume both X_new_scaled and hist_X are scaled the same way.
    X_for_train = np.concatenate([X_new_scaled, hist_X], axis=0)
    y_for_train = np.concatenate([y_new, hist_y], axis=0)
elif hist_X is not None and hist_y is None:
    # if hist_y not present, don't mix to avoid labelless data
    X_for_train = X_new_scaled
    y_for_train = y_new
else:
    X_for_train = X_new_scaled
    y_for_train = y_new

# shuffle dataset
perm = np.random.permutation(len(X_for_train))
X_for_train = X_for_train[perm]
y_for_train = y_for_train[perm]

# split a small validation holdout from the mix (5-10% of combined)
val_frac = 0.10
val_n = max( min(int(len(X_for_train)*val_frac), 20000), int(0.05*len(X_new_scaled)) )  # ensure some val
if val_n >= 20:
    X_val = X_for_train[:val_n]
    y_val = y_for_train[:val_n]
    X_train = X_for_train[val_n:]
    y_train = y_for_train[val_n:]
else:
    # if too small, use a smaller split
    X_train, y_train = X_for_train, y_for_train
    X_val, y_val = None, None

# backup current model with timestamp
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
backup_main = os.path.join(MODEL_DIR, f"lstm_backup_before_ft_{ts}.h5")
os.makedirs(MODEL_DIR, exist_ok=True)
shutil.copy2(MODEL_MAIN, backup_main)
if VERBOSE: print("Backed up current model to:", backup_main)

# import tensorflow and load model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

model = load_model(MODEL_MAIN)
model.compile(optimizer=Adam(LR), loss="binary_crossentropy", metrics=["AUC","accuracy"])

# evaluate pre-finetune validation metric if available (optional)
if X_val is not None:
    pre_eval = model.evaluate(X_val, y_val, verbose=0)
    pre_val_auc = None
    # try to find AUC value in metrics (metrics_order: loss, auc, accuracy)
    if len(pre_eval) >= 2:
        pre_val_auc = float(pre_eval[1])
    if VERBOSE: print(f"Pre-finetune val metrics (loss, auc, acc): {pre_eval}")

# fine-tune (small epochs)
if VERBOSE: print("Starting fine-tune: epochs", EPOCHS, "batch", BATCH_SIZE)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val) if X_val is not None else None,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# evaluate post-finetune on validation
if X_val is not None:
    post_eval = model.evaluate(X_val, y_val, verbose=0)
    post_val_auc = float(post_eval[1]) if len(post_eval)>=2 else None
    if VERBOSE: print(f"Post-finetune val metrics (loss, auc, acc): {post_eval}")
    # decision: if val AUC dropped by more than a small epsilon, rollback
    if pre_val_auc is not None and post_val_auc is not None:
        if post_val_auc + 1e-4 < pre_val_auc:   # allow tiny numerical noise
            print("Validation AUC decreased after fine-tune. Rolling back to backup model.")
            # restore backup to main
            shutil.copy2(backup_main, MODEL_MAIN)
            print("Model restored from:", backup_main)
            # still append rows to combined parquet (we didn't change model)
            to_add = df_new.drop(columns=["tomorrow_close","target"], errors="ignore")
            df_full = pd.concat([df_comb, to_add], ignore_index=True, sort=False).drop_duplicates(subset=["symbol","date"], keep="last")
            df_full = df_full.sort_values(["symbol","date"]).reset_index(drop=True)
            df_full.to_parquet(COMBINED)
            print("Appended new rows; fine-tune rolled back. Exiting.")
            sys.exit(0)
        else:
            # keep the fine-tuned model, and save versioned copy
            saved_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            model_savepath = os.path.join(MODEL_DIR, f"lstm_best_{saved_ts}.h5")
            model.save(model_savepath)
            # copy to main model path
            shutil.copy2(model_savepath, MODEL_MAIN)
            if VERBOSE: print("Fine-tuned model saved to:", model_savepath, "and copied to", MODEL_MAIN)
    else:
        # no pre/post AUC available; still save fine-tuned model with timestamp
        saved_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_savepath = os.path.join(MODEL_DIR, f"lstm_best_{saved_ts}.h5")
        model.save(model_savepath)
        shutil.copy2(model_savepath, MODEL_MAIN)
        if VERBOSE: print("Fine-tuned model saved to:", model_savepath, "and copied to", MODEL_MAIN)
else:
    # no validation available — still save versioned model but warn
    saved_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_savepath = os.path.join(MODEL_DIR, f"lstm_best_{saved_ts}.h5")
    model.save(model_savepath)
    shutil.copy2(model_savepath, MODEL_MAIN)
    if VERBOSE: print("No validation present; fine-tuned model saved to:", model_savepath, "and copied to", MODEL_MAIN)

# Append new rows to the combined parquet (drop helper columns)
to_add = df_new.drop(columns=["tomorrow_close","target"], errors="ignore")
df_full = pd.concat([df_comb, to_add], ignore_index=True, sort=False).drop_duplicates(subset=["symbol","date"], keep="last")
df_full = df_full.sort_values(["symbol","date"]).reset_index(drop=True)
df_full.to_parquet(COMBINED)
if VERBOSE: print("Appended new rows to combined parquet. Total rows now:", len(df_full))

print("Daily update + fine-tune complete. Model backed up and updated in", MODEL_DIR)
