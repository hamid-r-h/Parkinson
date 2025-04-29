# Parkinson Voice Analysis with Deep Learning

**Detecting Parkinson’s disease from raw speech signals using a modified SincNet convolutional neural network.**

This repository contains the full code, notebooks and experimental logs for my B.Sc. capstone project (Shahid Beheshti University, 2024). The pipeline ingests raw Italian & Spanish voice recordings, performs language‑specific preprocessing, then trains a deep model that reaches **91 % accuracy** (Italian cohort) and **90 % accuracy** (Spanish cohort) on unseen speakers.

---

## 🚀 Highlights

| Feature | Description |
|---------|------------|
| **End‑to‑end voice pipeline** | From raw `.wav` files to ready‑to‑use tensors & metrics — all in Jupyter. |
| **SincNet‑based CNN** | First convolutional layer replaced by parameterised *sinc* filters → interpretable, low‑parameter front‑end optimised for non‑stationary speech. |
| **Dual‑language evaluation** | Trained & tested on two public datasets (Italian & Spanish) to verify generalisation. |
| **No hand‑crafted features** | Model learns directly from waveform chunks; no MFCC extraction required. |
| **Reproducible notebooks** | Six notebooks document every stage: data I/O, preprocessing, modelling and result visualisation. |

---

## 📁 Repository Layout

```
Parkinson/
├── data/
│   ├── italian/               # place raw .wav + metadata.csv here
│   └── spanish/
├── notebooks/
│   ├── data_io.ipynb          # download / organise datasets
│   ├── italian_data_preprocess.ipynb
│   ├── spanish_preprocess.ipynb
│   ├── dnn_models.ipynb       # SincNet architecture & training loops
│   ├── italian_res.ipynb      # metrics & plots
│   └── spanish_res.ipynb
├── requirements.txt
└── README.md                  # you are here
```

---

## 🛠 Installation

```bash
# clone
$ git clone https://github.com/hamid-r-h/Parkinson.git
$ cd Parkinson

# (optional) create virtual env
$ python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies
$ pip install -r requirements.txt  # minimal list below if file missing
```
<details>
<summary>Key Python packages</summary>

```
python>=3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
torchaudio
librosa
jupyterlab
```
</details>

---

## 📊 Datasets

| Cohort | Source | Speakers | Files | Sampling | Notes |
|--------|--------|----------|-------|----------|-------|
| **Italian Parkinson’s Voice & Speech** | IEEE DataPort [5] | 22 PD / 19 HC | 97 | 16 kHz | Recorded in studio‑quality, two read passages per subject |
| **KCL‑MDVR (Spanish)** | Zenodo [6] | 21 PD / 16 HC | 37 | 44.1 kHz | Captured on Motorola G4 smartphone |

> ✋ **Data not redistributed** — download links are in `data_io.ipynb`, then place files under `data/` as shown above.

### Pre‑processing
* Butterworth low‑pass filtering, silence trimming, amplitude normalisation.
* Long recordings chunked into **500 ms** windows with **100 ms** overlap.

---

## 🧠 Model Architecture

```
Input waveform → [SincConv (80 × 251)] → LayerNorm → LeakyReLU
              → [Conv(60 × 5) + Pool] × 2
              → Flatten → FC(2048) → FC(2048) → Softmax
```
* **SincConv** learns band‑pass filters via cutoff frequencies rather than full kernels → ~8 k parameters instead of 80 k.
* Trained with **5‑fold cross‑validation**, **SGD** optimiser, 24–40 epochs.

---

## 📈 Results

| Dataset | Accuracy | Val F1 | Test Specificity | Ref. paper |
|---------|---------:|-------:|-----------------:|------------|
| Italian | **91 %** | 0.92 | 0.95 | Appakaya et al. (2021): 85 % |
| Spanish | **90 %** | 0.88 | 0.83 | Orozco‑Arroyave et al. (2014): 88 % |

All training logs, ROC curves and loss trajectories are stored in `results/<language>/` and visualised in the *res* notebooks.

---

## 🔄 Reproducing the Experiments

1. Download datasets via `data_io.ipynb`.
2. Run the corresponding `*_preprocess.ipynb` to generate feature tensors.
3. Open `dnn_models.ipynb`, set `language = 'italian'` or `'spanish'`, and run all cells.
4. Evaluate with `*_res.ipynb`.

A single‑GPU machine (e.g. RTX 2060) trains each fold in ≈ 1.5 h.

---

## 🤝 Contributing

Bug reports and PRs for new languages or alternative architectures are welcome. Please open an issue first to discuss major changes.

---

## 📄 Licence & Citation

Specify your licence in `LICENSE` (MIT by default). If you use this code in academic research, please cite:

```
@bachelorthesis{heidari2024parkinson,
  author = {Hamidreza Heidari},
  title  = {Using a Deep‑Learning Approach to Diagnose Parkinson’s Disease from Audio},
  school = {Shahid Beheshti University},
  year   = {2024}
}
```

---

*Last updated 2025‑04‑29.*

