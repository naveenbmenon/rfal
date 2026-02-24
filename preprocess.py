import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# CONFIGURATION
# =========================
SAMPLES_PER_EXAMPLE = 1024
snr_db = 0     # change this
num_tx = 8     # change this
DATA_DIR = "data"
OUT_DIR = "."

# =========================
# LOAD & FRAME FUNCTION
# =========================
def load_and_frame(filepath, label):
    iq = np.fromfile(filepath, dtype=np.complex64)

    num_examples = len(iq) // SAMPLES_PER_EXAMPLE
    iq = iq[:num_examples * SAMPLES_PER_EXAMPLE]

    X = np.zeros((num_examples, 2 * SAMPLES_PER_EXAMPLE), dtype=np.float32)
    y = np.full(num_examples, label, dtype=np.int64)

    for i in range(num_examples):
        chunk = iq[i*SAMPLES_PER_EXAMPLE:(i+1)*SAMPLES_PER_EXAMPLE]
        X[i, 0::2] = np.real(chunk)
        X[i, 1::2] = np.imag(chunk)

    return X, y

# =========================
# AUTO LOAD FILES BASED ON SNR & NUM_TX
# =========================
X_list = []
y_list = []

for tx in range(1, num_tx + 1):
    filename = f"trans{tx}_snr{snr_db}.dat"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"WARNING: {filename} not found. Skipping.")
        continue

    X_tx, y_tx = load_and_frame(filepath, tx - 1)
    print(f"Loaded {filename} → {X_tx.shape}")
    
    X_list.append(X_tx)
    y_list.append(y_tx)

# =========================
# MERGE DATASETS
# =========================
X = np.vstack(X_list)
y = np.hstack(y_list)

print("Final dataset:", X.shape, y.shape)

# =========================
# SAVE DNN VERSION
# =========================
np.save(os.path.join(OUT_DIR, "X_dnn.npy"), X)
np.save(os.path.join(OUT_DIR, "y.npy"), y)

# =========================
# RNN VERSION (N, 1024, 2)
# =========================
X_rnn = X.reshape(-1, SAMPLES_PER_EXAMPLE, 2)
np.save(os.path.join(OUT_DIR, "X_rnn.npy"), X_rnn)


X_cnn = X_rnn.reshape(-1, 2, SAMPLES_PER_EXAMPLE, 1)
np.save(os.path.join(OUT_DIR, "X_cnn.npy"), X_cnn)

print("Saved:")
print("  X_dnn :", X.shape)
print("  X_rnn :", X_rnn.shape)
print("  X_cnn :", X_cnn.shape)
print("  y     :", y.shape)

def plot_random_points(X, y, num_tx, num_points=5000):
    for cls in range(num_tx):
        indices = np.where(y == cls)[0]
        if len(indices) == 0:
            continue

        chosen = np.random.choice(indices, size=min(10, len(indices)), replace=False)

        I_all = []
        Q_all = []

        for idx in chosen:
            sample = X[idx]
            I_all.extend(sample[0::2])
            Q_all.extend(sample[1::2])

        plt.figure(figsize=(5, 5))
        plt.scatter(I_all, Q_all, s=2, alpha=0.5)
        plt.axhline(0, color="gray", linewidth=0.5)
        plt.axvline(0, color="gray", linewidth=0.5)
        plt.grid(True)
        plt.axis("equal")
        plt.title(f"Constellation – TX {cls} (Random Samples)")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.show()
        
#plot_random_points(X, y, num_tx)        