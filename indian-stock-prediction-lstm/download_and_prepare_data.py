# download_and_prepare_data.py
# ----------------------------------------------------------------------------------------------------------------------
# Purpose:
#   - Download NSE master symbol list (EQUITY_L.csv)
#   - For each symbol, download ~10 years of daily OHLCV from Yahoo Finance (ticker + ".NS")
#   - Robustly find Close & Volume columns (yfinance sometimes returns MultiIndex column names)
#   - Compute basic returns and a set of technical indicators using the 'ta' package
#   - Save a per-symbol CSV and a combined Parquet dataset with features for model building
#
# Notes:
#   - Run this inside the Python venv you created (venv311).
#   - The script is defensive: it skips tickers with insufficient data or missing columns.
#   - This is the full dataset mode: it iterates all symbols found in the NSE master list.
# ----------------------------------------------------------------------------------------------------------------------

# standard library imports
import os                                       # file system path ops
import time                                     # small sleeps to avoid throttling
import io                                       # handle in-memory bytes/streams
from typing import Optional                      # type hint for helper functions

# third-party imports
import requests                                  # download NSE master CSV reliably
import pandas as pd                              # dataframes for processing
import numpy as np                               # numeric ops
import yfinance as yf                            # Yahoo Finance downloader
from tqdm import tqdm                            # progress bar for long loops

# technical-analysis indicators from 'ta' package
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# -----------------------------
# Configuration (changeable)
# -----------------------------
OUT_DIR = "data_raw"                              # folder for per-symbol CSVs
COMBINED_PATH = "nse_all_features.parquet"        # final combined parquet file
YEARS = 10                                        # how many years to download
PERIOD = f"{YEARS}y"                              # yfinance period string
SLEEP_BETWEEN = 0.3                               # polite pause between requests (seconds)
MAX_WARNINGS_PRINT = 20                           # limit number of printed warnings to avoid spam
REQUEST_TIMEOUT = 20                              # seconds timeout for NSE master download

# ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def download_nse_master() -> Optional[pd.DataFrame]:
    """
    Download the NSE equities master CSV (EQUITY_L.csv) using common mirror URLs.
    Returns a pandas DataFrame or None on failure.
    """
    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
    ]
    headers = {"User-Agent": "Mozilla/5.0"}   # pretend to be a browser to avoid simple blocking
    for url in urls:
        try:
            # request the CSV bytes from NSE with a timeout
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            # ensure we got a 200 OK and content
            if r.status_code == 200 and r.content:
                # parse CSV bytes into pandas DataFrame
                df = pd.read_csv(io.BytesIO(r.content))
                # normalize column names by stripping whitespace
                df.columns = [c.strip() for c in df.columns]
                return df
        except Exception as e:
            # print a compact error message and continue to the next mirror
            print(f"Error fetching {url} → {e}")
    # if both mirrors fail, return None
    return None

def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If yfinance returned a DataFrame with MultiIndex columns, flatten them into single strings.
    Examples:
      - MultiIndex ('Close', 'RELIANCE.NS') -> 'Close_RELIANCE.NS'
      - Single-level names are preserved and stripped
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            # join non-empty parts with underscore
            parts = [str(c).strip() for c in col if c is not None and str(c).strip() != ""]
            new_cols.append("_".join(parts))
        df.columns = new_cols
    else:
        # make sure all column names are plain strings with no leading/trailing spaces
        df.columns = [str(c).strip() for c in df.columns]
    return df

def detect_price_and_volume_cols(df: pd.DataFrame):
    """
    Robustly detect which column in df looks like a Close (price) and which looks like Volume.
    It scans all column names and picks the first that contains the substring 'close' (case-insensitive)
    and first that contains 'volume' (case-insensitive).
    Returns (price_col_name_or_None, volume_col_name_or_None).
    """
    price_col = None
    volume_col = None
    for c in df.columns:
        low = str(c).lower()
        if price_col is None and "close" in low:
            price_col = c
        if volume_col is None and "volume" in low:
            volume_col = c
        # break early if both found
        if price_col is not None and volume_col is not None:
            break
    return price_col, volume_col

def safe_numeric(series):
    """
    Convert series to numeric (float), coerce errors to NaN, then return the series.
    This avoids problems when 'AdjClose' contains non-numeric junk.
    """
    return pd.to_numeric(series, errors="coerce")

# -----------------------------
# Main execution
# -----------------------------

# 1) Download NSE master list
master_df = download_nse_master()
if master_df is None:
    # if master list could not be downloaded, exit with an explanatory message
    raise SystemExit("❌ Could not download NSE master list. Please save EQUITY_L.csv into this folder and re-run.")

# ensure master_df has SYMBOL column
if "SYMBOL" not in master_df.columns:
    raise SystemExit("❌ Master list is missing the SYMBOL column. Inspect EQUITY_L.csv manually.")

# build a cleaned list of unique tickers (uppercase, stripped)
symbols = master_df["SYMBOL"].astype(str).str.strip().str.upper().unique().tolist()
print(f"Found {len(symbols)} symbols in NSE master list.")

# prepare container for per-symbol feature DataFrames
records = []

# small warnings counter so we don't spam the terminal too much
warnings_shown = 0

# 2) Loop through symbols and download data
for s in tqdm(symbols, desc="Downloading stocks", ncols=120):
    # construct Yahoo ticker by adding .NS suffix for NSE symbols
    yf_ticker = s + ".NS"

    # try downloading with a couple of retries inside a small loop
    df_raw = None
    for attempt in range(3):
        try:
            # use threads=False to avoid unexpected parallel behavior and keep deterministic ordering
            df_raw = yf.download(yf_ticker, period=PERIOD, interval="1d", progress=False, threads=False)
            # stop retrying if data was returned
            if df_raw is not None and not df_raw.empty:
                break
        except Exception as e:
            # short backoff and retry
            time.sleep(1 + attempt)
    # if nothing obtained, skip this ticker with a short message
    if df_raw is None or df_raw.empty:
        if warnings_shown < MAX_WARNINGS_PRINT:
            print(f"  Skipped {s}: yfinance returned no data.")
            warnings_shown += 1
        continue

    # flatten yfinance columns (handle MultiIndex like ('Close','RELIANCE.NS'))
    try:
        df_flat = flatten_yf_columns(df_raw.copy())
    except Exception as e:
        if warnings_shown < MAX_WARNINGS_PRINT:
            print(f"  Skipped {s}: error flattening columns → {e}")
            warnings_shown += 1
        continue

    # robustly detect price & volume column names
    price_col, volume_col = detect_price_and_volume_cols(df_flat)

    # if no close-like column, skip
    if price_col is None:
        if warnings_shown < MAX_WARNINGS_PRINT:
            print(f"  Skipped {s}: no 'close' like column found. Columns sample: {list(df_flat.columns)[:8]}")
            warnings_shown += 1
        continue

    # select only relevant columns that exist in this DataFrame
    available_cols = []
    for c in ["Open", "High", "Low", "Close", price_col, "Volume", volume_col]:
        if c in df_flat.columns and c not in available_cols:
            available_cols.append(c)

    # ensure at minimum we have price column and at least one OHLC or Volume
    if price_col not in available_cols:
        if warnings_shown < MAX_WARNINGS_PRINT:
            print(f"  Skipped {s}: price column not present after filtering.")
            warnings_shown += 1
        continue

    # build a working DataFrame with the available columns
    df_work = df_flat.loc[:, available_cols].copy()

    # rename the detected price column to 'AdjClose' for downstream consistency
    if price_col in df_work.columns:
        df_work = df_work.rename(columns={price_col: "AdjClose"})

    # rename the detected volume column to 'Volume' if present
    if volume_col in df_work.columns:
        df_work = df_work.rename(columns={volume_col: "Volume"})

    # reset index to expose Date as a column (index may be DatetimeIndex)
    df_work = df_work.reset_index()

    # coerce numeric columns to numbers, keep symbol and date as-is
    for col in df_work.columns:
        if col not in ("symbol", "date", "Date"):  # avoid touching date-like columns here
            try:
                df_work[col] = safe_numeric(df_work[col])
            except Exception:
                # ignore coercion errors - will handle missing values later
                pass

    # if Volume not present, create a zero-filled Volume column (safe fallback for indicators)
    if "Volume" not in df_work.columns:
        df_work["Volume"] = 0.0

    # drop rows with missing AdjClose (price) as they are unusable
    if "AdjClose" not in df_work.columns or df_work["AdjClose"].isna().all():
        if warnings_shown < MAX_WARNINGS_PRINT:
            print(f"  Skipped {s}: AdjClose empty or missing after processing.")
            warnings_shown += 1
        continue
    df_work = df_work.dropna(subset=["AdjClose"])

    # if there are too few rows, skip (not enough history)
    if df_work.shape[0] < 10:
        if warnings_shown < MAX_WARNINGS_PRINT:
            print(f"  Skipped {s}: insufficient rows ({df_work.shape[0]}).")
            warnings_shown += 1
        continue

    # canonicalize column names: add 'symbol' and 'date' columns for later grouping
    df_work["symbol"] = s
    # if 'Date' exists (from reset_index) use it, otherwise use the index as date
    if "Date" in df_work.columns:
        df_work["date"] = df_work["Date"]
    else:
        df_work["date"] = df_work.index

    # compute basic returns (safe numeric ops)
    try:
        df_work["ret1"] = df_work["AdjClose"].pct_change(1).fillna(0)
        df_work["ret5"] = df_work["AdjClose"].pct_change(5).fillna(0)
        df_work["logret"] = np.log(df_work["AdjClose"]).diff().fillna(0)
    except Exception as e:
        if warnings_shown < MAX_WARNINGS_PRINT:
            print(f"  Skipped {s}: error computing basic returns → {e}")
            warnings_shown += 1
        continue

    # compute technical indicators using 'ta', using safe fallbacks when needed
    try:
        # Simple and Exponential Moving Averages
        df_work["sma10"] = SMAIndicator(df_work["AdjClose"], window=10).sma_indicator()
        df_work["sma50"] = SMAIndicator(df_work["AdjClose"], window=50).sma_indicator()
        df_work["ema12"] = EMAIndicator(df_work["AdjClose"], window=12).ema_indicator()

        # RSI
        df_work["rsi14"] = RSIIndicator(df_work["AdjClose"], window=14).rsi()

        # MACD and signal
        macd_obj = MACD(df_work["AdjClose"])
        df_work["macd"] = macd_obj.macd()
        df_work["macd_signal"] = macd_obj.macd_signal()

        # Bollinger Bands
        bb = BollingerBands(df_work["AdjClose"], window=20, window_dev=2)
        df_work["bb_upper"] = bb.bollinger_hband()
        df_work["bb_lower"] = bb.bollinger_lband()

        # On-Balance Volume (safe because we ensured a numeric Volume column exists)
        obv = OnBalanceVolumeIndicator(close=df_work["AdjClose"], volume=df_work["Volume"])
        df_work["obv"] = obv.on_balance_volume()
    except Exception as e:
        # if TA computation fails for a specific ticker, print a compact message but continue
        if warnings_shown < MAX_WARNINGS_PRINT:
            print(f"  TA error for {s}: {e}")
            warnings_shown += 1
        # fill indicator columns with NaN to keep schema consistent
        for col in ("sma10","sma50","ema12","rsi14","macd","macd_signal","bb_upper","bb_lower","obv"):
            if col not in df_work.columns:
                df_work[col] = np.nan

    # forward/back fill small gaps then replace remaining NaN with zeros (safer for ML pipelines)
    df_work = df_work.fillna(method="bfill").fillna(method="ffill").fillna(0.0)

    # winsorize extreme returns to reduce outlier influence
    df_work["ret1"] = df_work["ret1"].clip(-0.25, 0.25)
    df_work["ret5"] = df_work["ret5"].clip(-0.5, 0.5)

    # attach industry/sector info if present in the master list (best-effort)
    meta = master_df[master_df["SYMBOL"].astype(str).str.strip().str.upper() == s]
    if not meta.empty:
        if "INDUSTRY" in master_df.columns:
            df_work["industry"] = meta.iloc[0].get("INDUSTRY", "")
        elif "Industry" in master_df.columns:
            df_work["industry"] = meta.iloc[0].get("Industry", "")
        else:
            df_work["industry"] = ""
    else:
        df_work["industry"] = ""

    # attempt to save per-symbol CSV (use try/except to avoid one file failing the whole run)
    try:
        csv_path = os.path.join(OUT_DIR, f"{s}.csv")
        df_work.to_csv(csv_path, index=False)
    except Exception:
        # ignore write failures for individual symbols
        pass

    # append prepared DataFrame to the list for final concatenation
    records.append(df_work)

    # polite pause to avoid being throttled by yfinance / remote servers
    time.sleep(SLEEP_BETWEEN)

# -----------------------------
# Combine all symbol frames and save to Parquet
# -----------------------------
if len(records) == 0:
    # if nothing collected, stop and show a helpful message
    raise SystemExit("❌ No data downloaded. Something went wrong during the per-symbol processing.")

# concatenate all DataFrames into one large frame
combined = pd.concat(records, ignore_index=True)

# print a quick summary of the combined dataset
print(f"Combined rows: {len(combined):,}  (columns: {list(combined.columns)})")

# write the combined dataset to a Parquet file for fast read/write and compact storage
combined.to_parquet(COMBINED_PATH, index=False)

# final confirmation print
print("✅ Saved combined dataset to:", COMBINED_PATH)
