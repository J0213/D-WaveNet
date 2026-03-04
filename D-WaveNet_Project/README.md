# D-WaveNet: A Physics-Informed Transformer with Cross-Scale Interaction for Significant Wave Height Forecasting

**[Notice for Peer Review]** *This repository is fully anonymized to comply with double-blind peer-review policies. The complete pre-trained weights, raw buoy datasets, and detailed data-preprocessing pipelines are pending institutional IP clearance and will be fully open-sourced upon the formal acceptance of the manuscript.*

---

## 🌊 Overview
This repository provides the official PyTorch implementation of **D-WaveNet**. 

Significant Wave Height (SWH) forecasting is often bottlenecked by the "orthogonality fallacy" in multi-scale decomposition and the lack of hydrodynamic constraints in deep learning models. D-WaveNet bridges the chasm between data-driven efficiency and physical consistency through two core innovations:
- **Wavelet-Component Fusion Transformer (WCFT):** Explicitly models the non-linear energy modulation between high-frequency wind-seas and low-frequency swells via a Cross-Scale Attention mechanism.
- **Kinematic-Dynamic Coupled Module (KDCM):** Internalizes wave acceleration as a precursor signal to eliminate hysteresis and enforces a physics-informed loss (depth-induced breaking limits & Stokes steepness) to prevent physical hallucinations.

## 📁 Repository Structure
Our codebase is meticulously structured for optimal readability and reproducibility:

```text
D-WaveNet_Project/
├── data_provider/          # Data loading & preprocessing
│   ├── __init__.py
│   └── data_loader.py      # Contains moving average & DWT (db4) causal separation
├── models/                 # Core network architectures
│   ├── __init__.py
│   └── D_WaveNet.py        # Implementation of WCFT and KDCM
├── utils/                  # Evaluation metrics & custom losses
│   ├── __init__.py
│   ├── metrics.py          # Includes NSE, Pearson's R2, MSE, and MAE
│   └── physics_loss.py     # Physics-Informed Loss (Depth & Steepness bounds)
├── run.py                  # Main execution pipeline and hyperparameter config
├── requirements.txt        # Python environment dependencies
└── README.md               # Project documentation