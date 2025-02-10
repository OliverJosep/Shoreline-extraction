## Directory Structure

```
/shoreline-extraction
├── /data/
│   ├── /raw/                  # Original (unprocessed) dataset
│   ├── /splits/               # Dataset splits (train / val / test)
│   └── /patchified/           # Dataset with patches and splits (train / val / test)
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
├── /scripts/                  # Scripts for data processing, training, and evaluation
│   ├── /data_processing/      # Scripts for generating patches, preprocessing, etc.
│   ├── /training/             # Scripts for training models
│   ├── /evaluation/           # Scripts for model evaluation
│   └── /utils/                # General utilities (e.g., visualization functions, metrics)
│
├── /notebooks/                # Jupyter notebooks for experimentation and analysis
│   ├── /data-exploration/     # Data exploration analysis
│   ├── /model-training/       # Model training notebooks
│   └── /evaluation/           # Model evaluation notebooks
│
├── /logs/                     # Log files, if generated during training
│
├── README.md                  # Project overview and documentation
├── requirements.txt           # Project dependencies
└── LICENSE                    # Project license (optional)
```
