# train_lstm_from_h5.py
# Train an LSTM to predict "tomorrow_up" using the scaled HDF5 created earlier.
# - Streams batches from sequences/sequences_scaled.h5 using a Keras Sequence
# - Saves best model to models/lstm_best.h5
# - Works on CPU or GPU (TF auto-chosen)

import os
import math
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import Sequence

# ----------------------
# Config
# ----------------------
H5_PATH = "sequences/sequences_scaled.h5"   # input scaled HDF5 (float16)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_best.h5")

SEQ_LEN = 90           # must match what you built
N_FEATURES = 12        # must match FEATURE_COLS length used earlier
BATCH_SIZE = 256       # adjust if GPU OOM; 256 is a reasonable starting point for 4050
EPOCHS = 20
LR = 1e-3

# ----------------------
# Data generator (streams from HDF5)
# ----------------------
class H5Sequence(Sequence):
    """Keras Sequence reading X/y pairs directly from an HDF5 dataset in chunks."""
    def __init__(self, h5file, xname, yname, start, end, batch_size):
        self.h5 = h5file
        self.xname = xname
        self.yname = yname
        self.start = int(start)
        self.end = int(end)
        self.batch_size = int(batch_size)
        self.n = self.end - self.start

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def __getitem__(self, idx):
        i0 = self.start + idx * self.batch_size
        i1 = min(self.end, i0 + self.batch_size)
        # read from disk (X stored as float16, convert to float32 for training)
        X = self.h5[self.xname][i0:i1].astype(np.float32)
        y = self.h5[self.yname][i0:i1].astype(np.float32)
        return X, y

    def on_epoch_end(self):
        # optional: you can shuffle indices by reading shuffled index list from disk.
        pass

# ----------------------
# Open HDF5 and get split sizes
# ----------------------
print("Opening HDF5:", H5_PATH)
hf = h5py.File(H5_PATH, "r")
n_train = hf["X_train"].shape[0]
n_val = hf["X_val"].shape[0]
n_test = hf["X_test"].shape[0]
print("Train/Val/Test shapes:", hf["X_train"].shape, hf["X_val"].shape, hf["X_test"].shape)

# ----------------------
# Build data sequences
# ----------------------
train_seq = H5Sequence(hf, "X_train", "y_train", 0, n_train, BATCH_SIZE)
val_seq   = H5Sequence(hf, "X_val", "y_val", 0, n_val, BATCH_SIZE)
test_seq  = H5Sequence(hf, "X_test", "y_test", 0, n_test, BATCH_SIZE)

# ----------------------
# Simple LSTM model
# ----------------------
def build_model(seq_len=SEQ_LEN, n_features=N_FEATURES):
    inp = layers.Input(shape=(seq_len, n_features), dtype=tf.float32)
    # first LSTM block (return sequences for stacked LSTM)
    x = layers.LSTM(256, return_sequences=True)(inp)
    x = layers.Dropout(0.25)(x)
    # second LSTM block (final encoding)
    x = layers.LSTM(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    # dense head for binary classification
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)   # probability that tomorrow_up == 1
    model = models.Model(inputs=inp, outputs=out)
    return model

model = build_model()
opt = optimizers.Adam(learning_rate=LR)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["AUC", "accuracy"])
model.summary()

# ----------------------
# Callbacks
# ----------------------
cb = []
cb.append(callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, verbose=1))
cb.append(callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True, verbose=1))
cb.append(callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, verbose=1))

# ----------------------
# Train
# ----------------------
print("Starting training EPOCHS=", EPOCHS)
history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    callbacks=cb,
    workers=4,
    use_multiprocessing=False,
)

# ----------------------
# Evaluate on test set
# ----------------------
print("Evaluating on test set...")
res = model.evaluate(test_seq)
print("Test results (loss, AUC, acc):", res)

# ----------------------
# Save final model if not already saved by checkpoint
# ----------------------
if not os.path.exists(MODEL_PATH):
    model.save(MODEL_PATH)
    print("Saved model to", MODEL_PATH)
else:
    print("Best model saved to", MODEL_PATH)

# Close HDF5 handle
hf.close()
