# 📈 Indian Stock Market Prediction (LSTM Pipeline)

An end-to-end, out-of-core Deep Learning pipeline designed to predict upward price movements in the Indian Stock Market (NSE). 

This system handles automated daily data ingestion, feature engineering, memory-safe sequence generation for datasets larger than RAM, LSTM training and continuous daily fine-tuning with rollback protections.

**Maintained by:** [asteroid-deva](https://github.com/asteroid-deva)

---

## 🏗️ System Architecture & Codebase Deep Dive

The pipeline is modularized into five distinct execution phases. Below is the detailed documentation for every script driving the architecture.

### Phase 1: Data Ingestion & ETL
**`download_and_prepare_data.py`**
* **Function:** The foundational data engineering script. It downloads the master equities list directly from the NSE and fetches up to 10 years of historical OHLCV data for every ticker via Yahoo Finance.
* **Technical Details:**
  * Implements defensive fetching with retries and timeout handling to bypass rate limits.
  * Flattens complex `MultiIndex` columns and handles missing price data robustly.
  * Computes core technical indicators using the `ta` library, including Simple/Exponential Moving Averages (10, 12, 50), RSI (14), MACD, Bollinger Bands and On-Balance Volume.
  * Winsorizes extreme returns (clipping outliers) to stabilize neural network training.
  * Exports the massive unified dataset to a highly optimized `nse_all_features.parquet` file.

### Phase 2: Sequence Engineering (Time-Series Tensors)
Neural networks require data formatted as 3D tensors `(samples, time_steps, features)`. This project utilizes a 90-day lookback window (`SEQ_LEN = 90`). To handle massive data, it offers two generation methods:

**`build_sequences_stream_h5.py` (Primary - Out-of-Core)**
* **Function:** Memory-safe sequence builder for datasets larger than available RAM.
* **Technical Details:**
  * Streams sequences directly to a disk-backed `.h5` (HDF5) file in chunks (`BATCH_RESIZE = 1024`).
  * Computes scaling parameters (Mean/Std) incrementally using Welford-style chunked streaming.
  * Casts scaled features to `float16` to drastically reduce disk footprint without losing model precision.
  * Automatically handles Train (70%) / Val (15%) / Test (15%) chronological splits.

**`build_sequences_and_save.py` (Alternative - In-Memory)**
* **Function:** A lighter, in-memory sequence builder that uses `sklearn.preprocessing.StandardScaler`.
* **Technical Details:** Saves standard compressed `.npz` files for smaller dataset testing.

### Phase 3: Deep Learning Model Training
**`train_lstm_from_h5.py`**
* **Function:** The core model training engine.
* **Technical Details:**
  * Utilizes a custom `tf.keras.utils.Sequence` generator (`H5Sequence`) to stream batches from the `.h5` file directly to the CPU/GPU, completely avoiding Out-Of-Memory (OOM) crashes.
  * **Architecture:** A stacked architecture featuring a 256-unit LSTM layer (returning sequences), followed by a 128-unit LSTM layer, Batch Normalization and aggressive Dropout (0.25) to prevent overfitting.
  * **Callbacks:** Implements `ModelCheckpoint` (saving best `val_auc`), `EarlyStopping` and `ReduceLROnPlateau`.
  * Saves the final tuned model to `models/lstm_best.h5`.

### Phase 4: Inference & Actionable Trading Output
**`predict_and_save_filtered.py` (Primary)**
* **Function:** Generates tomorrow's predictions and applies strict business logic to output actionable trading candidates.
* **Technical Details:**
  * Restores the saved model and normalizes the last 90 days of data using the saved `.npy` scaler.
  * **Noise Filtering:** Drops low-liquidity stocks by strictly requiring a minimum 30-day average volume (e.g., > 200,000 shares) and filtering out penny stocks (< ₹1).
  * **Probability Threshold:** Only accepts predictions where the model's confidence is highly elevated (e.g., `PROB_THRESHOLD >= 0.75`).
  * Outputs the final, sorted candidates to `predictions_topk.csv`.

**`predict_tomorrow_from_parquet.py` (Utility)**
* **Function:** A lightweight console script for rapid probability checks without generating the full filtered CSV suite.

### Phase 5: Automated MLOps (Daily Operations)
**`daily_train_and_update.py`**
* **Function:** The daily heartbeat of the system. Safely updates the dataset and gently fine-tunes the model to adapt to new market regimes.
* **Technical Details:**
  * **Delta Downloads:** Only fetches the last 7 calendar days to save bandwidth and compute time.
  * **Historical Replay:** To prevent "catastrophic forgetting" (where the model overfits to the last 7 days and forgets long-term patterns), it samples up to 50,000 historical sequences and mixes them with the new data.
  * **Safe Rollback:** Evaluates the Validation AUC before and after fine-tuning (1-2 epochs). If the model's performance degrades, it automatically rolls back to yesterday's backup weights.

**`archive_predictions_daily.py`**
* **Function:** Auditing and logging utility.
* **Technical Details:** Renames the daily prediction CSV outputs with a UTC timestamp and securely moves them to `predictions_archive/` to allow for long-term historical backtesting.

---

## 🚀 Execution Guide

To initialize and run this pipeline from scratch:

**1. Initialize Data & Tensors:**
```bash
python download_and_prepare_data.py
python build_sequences_stream_h5.py
