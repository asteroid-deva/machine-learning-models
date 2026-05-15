#!/usr/bin/env python3
# archive_predictions_daily.py
# Copies / renames today's prediction CSVs to files with YYYY-MM-DD suffix.
# Usage:
#   source venv311/bin/activate
#   python archive_predictions_daily.py

import os, shutil
from datetime import datetime

# --- Config: filenames your prediction script writes
FILES = [
    "predictions_all.csv",
    "predictions_filtered.csv",
    "predictions_topk.csv"
]

OUT_DIR = "predictions_archive"  # where to store archived files
DATE = datetime.utcnow().strftime("%Y-%m-%d")  # e.g., 2025-11-29 (UTC)
os.makedirs(OUT_DIR, exist_ok=True)

# iterate all files and copy/rename them with the date suffix
for fname in FILES:
    if not os.path.exists(fname):
        print(f"Skip: {fname} not found (no file to archive).")
        continue
    base, ext = os.path.splitext(fname)
    outname = f"{base}_{DATE}{ext}"           # e.g., predictions_topk_2025-11-29.csv
    outpath = os.path.join(OUT_DIR, outname)
    # if file already exists, append a numeric suffix to avoid overwrite
    if os.path.exists(outpath):
        i = 1
        while True:
            alt = os.path.join(OUT_DIR, f"{base}_{DATE}_{i}{ext}")
            if not os.path.exists(alt):
                outpath = alt
                break
            i += 1
    shutil.copy2(fname, outpath)  # copy with metadata
    print("Archived:", fname, "->", outpath)

print("Archive complete. Files stored in:", OUT_DIR)
