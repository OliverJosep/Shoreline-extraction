This repository was developed as part of the Master's Thesis:  
**"Shoreline Detection Using Deep Learning Techniques on RGB Imagery"**  
by *Josep Oliver Sansó* — MSc in Intelligent Systems, Universitat de les Illes Balears (2025).  

---

## Reproducibility

This repository contains all code and configuration files used in the master's thesis.  
The dataset version used can be found at [Zenodo](https://doi.org/10.5281/zenodo.10159978).  

To reproduce the experiments, please start from the notebooks in [`/notebooks/`](notebooks/) or the scripts in [`/src/`](src/).  
All predictions and dataset splits used for evaluation are stored in [`/predictions/`](predictions/).

---

## Directory Structure

```
/shoreline-extraction
├── /data/
│   ├── /raw/                  # Original (unprocessed) dataset
│   ├── /splits/               # Dataset splits (train / val / test)
│   └── /patchified/           # Dataset with patches and splits (train / val / test)
│
├── /predictions/              # Model predictions and train/val/test splits
│   ├── /station/              # Visual predictions by coastal station
│   └── /splits/               # Text files listing the dataset split
│
├── /src/                      # Source code: data processing, training, evaluation
│   ├── /data_processing/      # Scripts for generating patches, augmentation, etc.
│   ├── /training/             # Model training scripts and configuration
│   ├── /evaluation/           # Evaluation routines and metrics
│   └── /utils/                # General-purpose utilities
│
├── /models/                   # Models and architectures
│   ├── /unet/                 # U-Net implementation
│   ├── /bilstm/               # Bi-LSTM implementation
│   └── /other-models/         # Other models, if used
│
├── /weights/                  # Pre-trained weights and saved models
│   ├── /unet/                 # U-Net weights
│   ├── /bilstm/               # Bi-LSTM weights
│   └── /other-models/         # Weights for other models
│
├── /notebooks/                # Jupyter notebooks for experimentation and analysis
│   ├── /data-exploration/     # Data exploration analysis
│   ├── /model-training/       # Model training notebooks
│   └── /evaluation/           # Model evaluation notebooks
│
├── .env.example             # Example environment variables file
├── README.md                  # Project overview and documentation
└── requirements.txt           # Project dependencies
```

---

## .env configuration

The project includes a `.env.example` file used to configure MLflow for experiment tracking.  
To enable tracking (locally or via a remote server), rename the file to `.env` and update the variables with your own configuration.
