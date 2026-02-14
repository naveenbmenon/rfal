# RF Transmitter Identification using Deep Learning

A deep learning pipeline for identifying RF transmitters based on hardware-induced signal distortions in raw I/Q samples, inspired by the RFAL (Radio Frequency Adversarial Learning) framework.

## Problem

Traditional RF transmitter identification relies on digital identifiers, which can be spoofed. Physical-layer fingerprinting instead exploits hardware imperfections — unique distortions introduced by the transmitter's analog components — to identify devices independent of any digital credential.

## Approach

Instead of analyzing packet headers or digital identifiers, this system operates directly on raw I/Q baseband samples. Hardware-induced impairments (IQ imbalance, DC offset) create distinct signal fingerprints that deep learning models can learn to classify.

## System Architecture

```
QPSK Signal Generation (GNU Radio)
            ↓
Transmitter Impairment Simulation
(IQ imbalance + DC offset per virtual transmitter)
            ↓
Receiver Synchronization
(Symbol Sync + Costas Loop)
            ↓
Baseband I/Q Capture (.dat)
            ↓
Preprocessing & Feature Construction (Python)
            ↓
2048-D Feature Vectors
            ↓
DNN / RNN Classification
```

## Signal Generation & Transmitter Simulation

Since multiple physical USRPs were not available, transmitter diversity is emulated in GNU Radio using controlled RF impairments:

- **IQ Imbalance Generator** — introduces amplitude and phase mismatch
- **Add Const Block** — introduces gain and DC offset

Each virtual transmitter uses distinct impairment parameters, producing unique RF fingerprints across the dataset.

Signal chain:
- Random binary source → QPSK modulation → Root Raised Cosine (RRC) filtering
- Configurable sample rate and symbol rate
- Channel modeling with noise and frequency offsets

## Data Preprocessing (RFAL Format)

Raw I/Q streams are converted into structured ML inputs:

1. Continuous stream segmented into frames of 1024 complex samples
2. I and Q components separated and interleaved
3. Output: 2048-dimensional real-valued feature vectors
4. Global normalization using training set statistics
5. Labels assigned per transmitter

This format is compatible with the RFAL framework.

## Models

### Deep Neural Network (DNN)
- Input: 2048-D feature vector
- Fully connected layers with Tanh activation
- Dropout regularization
- Softmax output
- Learns global distortion patterns across the I/Q space
- **Accuracy: 97–99%**

### Recurrent Neural Network (GRU-based)
- Input reshaped as 1024 × 2 sequence
- GRU layers for temporal signal modeling
- Dropout + Softmax output
- Models time-dependent signal characteristics
- **Accuracy: 80–85%**

Identical dataset splits used across both models for fair comparison.

## Evaluation

- Accuracy, Precision, Recall, F1-score
- Confusion matrix per model
- Side-by-side DNN vs RNN comparison on identical data splits

## Tech Stack

| Component | Technology |
|---|---|
| Signal Generation | GNU Radio |
| Preprocessing | Python, NumPy |
| Deep Learning | TensorFlow / Keras |
| Visualization | Matplotlib |
| Concepts | QPSK, IQ imbalance, symbol synchronization, RF fingerprinting |

## Current Limitations

- Dataset is simulation-based — no real USRP hardware used
- Limited number of virtual transmitters
- Frame-level dataset splitting may introduce local correlations
- Adversarial spoofing (GAN stage) not yet implemented

## Planned Work

- GAN-based adversarial signal generation for spoofing simulation
- Robustness testing under varying noise and frequency offset conditions
- Over-the-air validation using physical SDR hardware
- Signal-level dataset splitting to eliminate frame leakage

## References

- RFAL Framework: Adversarial RF Fingerprinting Research
- Deep Learning for RF Signal Classification
- GNU Radio Documentation
