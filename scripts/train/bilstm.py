import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the path. TEMPORARY FIX
sys.path.append(os.getcwd())

from models.bilstm.bilstm import BiLSTM
from src.models.data_management.bilstm_formes import BiLSTMFormesDataset
from src.data_processing.dataset_loader import CoastData
from models.base_model import BaseModel

from typing import Type

import torch
import gc

MLFLOW_EXPERIMENT_NAME = "shoreline_bilstm"

image_type_paths = {
    "rectified": {
        "path": os.path.abspath(os.path.join(os.getcwd(), "data/processed_bilstm_global/")),
        "num_classes": 1
    },
    "oblique": {
        "path": os.path.abspath(os.path.join(os.getcwd(), "data/processed_obliques_2_classes/")),
        "num_classes": 1
    }
}

networks: dict[str, Type[BaseModel]] = {
    "BiLSTM": BiLSTM
}

def main():

    for data_type in image_type_paths:
        print(f"\n{'#'*30}\nProcessing {data_type} images\n{'#'*30}")
        data_path = image_type_paths[data_type]["path"]
        num_classes = image_type_paths[data_type]["num_classes"]

        # Load data
        artifact_path = os.path.abspath(os.path.join(os.getcwd(), "artifacts/"))

        data = CoastData(data_path, name="global")

        filtered_data = data.split_data()

        for network in networks:
            print(f"\n{'='*30}\nStarting training for {network}\n{'='*30}")
    
            is_oblique = True if data_type == "oblique" else False

            model = networks[network](num_classes, experiment_name=MLFLOW_EXPERIMENT_NAME, use_mlflow=True, oblique=is_oblique)

            model.load_data(filtered_data, BiLSTMFormesDataset, batch_size=1)

            # Training
            print(f"\nTraining model for {data_type}")
            run_name = f"{data_type}_{network}"
            description = f"Dataset type: {data_type}, Training {network}"
            early_stopping = 10
            model.train(epochs=100, loss_function_name="BCELoss", artifact_path=artifact_path, run_name=run_name, run_description=description, early_stopping=early_stopping)
            print(f"\tModel trained for {data_type}")

            # Clear memory
            del model
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()