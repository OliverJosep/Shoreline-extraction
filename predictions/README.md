# Visual Predictions – Best Model

This folder contains qualitative results from the **test set**, generated using the best-performing model (DeepLabv3 with three-class configuration). Each image includes:

- Input RGB image (background)
- Predicted shoreline (red line)
- Ground truth shoreline (green line)
- Agreement zone (yellow line): where prediction and ground truth overlap

These images allow direct visual comparison between model predictions and manual annotations.

## Legend

All visual results follow the same color convention:

- 🟥 **Red line**: Predicted shoreline
- 🟩 **Green line**: Ground truth shoreline
- 🟨 **Yellow line**: Agreement zone (prediction ≈ ground truth)
- 🖼️ **Background**: Original RGB image

## Dataset

The images used for these visualisations correspond to the **test split**, defined by the authors following a 70/20/10 division strategy (train/validation/test) as explained in the master's thesis.

The original data was sourced from the [SCLabels dataset](https://zenodo.org/records/10159978), which contains rectified RGB shoreline imagery collected from the Spanish CoastSnap network.

A separate `splits/` folder contains three `.txt` files listing the image identifiers used in each subset:

```
splits/
├── train.txt
├── val.txt
└── test.txt
```

Each file contains one image identifier per line. These identifiers match the naming convention used in the dataset.

## Folder Structure

The prediction images are organised by coastal station inside the `station/` folder:

Each `.png` file shows a composite visualisation including the original image, the model prediction, and the ground truth shoreline.

---

These results support the qualitative evaluation presented in the associated master's thesis.
