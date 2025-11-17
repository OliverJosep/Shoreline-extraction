import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the path. TEMPORARY FIX
sys.path.append(os.getcwd())

from models.deeplab.deeplab import DeepLabV3
from models.duck_net.duck_net import DuckNet
from models.unet.unet import UNet
from models.attention_unet.attention_unet import Attention_UNet
from models.cnn_model import CNNModel

from src.models.data_management.cnn_formes import CNNFormes
from src.data_processing.dataset_loader import CoastData
from src.data_processing.patchify import Patchify

from typing import Type

import pandas as pd
import os

import torch
import gc

MLFLOW_EXPERIMENT_NAME = "shoreline_kfold"

image_type_paths = {
    "oblique": {
        "path": os.path.abspath(os.path.join(os.getcwd(), "data/processed_obliques_2_classes/")),
        "num_classes": 2
    },
    # "rectified": {
    #     "path": os.path.abspath(os.path.join(os.getcwd(), "data/processed_rectified_3_classes/")),
    #     "num_classes": 3
    # }
}

networks: dict[str, Type[CNNModel]] = {
    "DeepLabV3": DeepLabV3,
}

patches = {
    "256x1024": {
        "patch_size": (256, 1024),
        "stride": (128, 512)
    },
}

def generate_patches(data, output_dir, patch_size, stride):
    patchify = Patchify(patch_size, stride)
    patchify.extract_patches_and_save(data, output_dir, skip_no_shoreline=True, binary_class=True, p_keep_negative=1, padding_mode='reflect')


def export_folds_summary(data_splits, output_dir='.'):
    file_roles = {}

    for fold_name in data_splits.keys():
        for split_type in ['validation', 'train', 'test']:
            if data_splits[fold_name][split_type].get('images'):

                file_paths = data_splits[fold_name][split_type]['images']

                for file_path in file_paths:
                    filename = os.path.basename(file_path)

                    if filename not in file_roles:
                        file_roles[filename] = {}
                    
                    file_roles[filename][fold_name] = split_type

    df = pd.DataFrame.from_dict(file_roles, orient='index')

    df = df.reindex(columns=sorted(data_splits.keys()))
    df = df.reset_index().rename(columns={'index': 'filename'})

    output_filename = os.path.join(output_dir, 'folds_summary.csv')
    df.to_csv(output_filename, index=False)

    print(df.head())
    

def main():

    for data_type in image_type_paths:
        print(f"\n{'#'*30}\nProcessing {data_type} images\n{'#'*30}")
        data_path = image_type_paths[data_type]["path"]
        num_classes = image_type_paths[data_type]["num_classes"]

        # Load data
        artifact_path = os.path.abspath(os.path.join(os.getcwd(), "artifacts/"))

        data = CoastData(data_path)

        k = 5
        data_splits = data.get_kfold_splits(k, test_size=0.1)
        print(f"Generated {k}-fold splits with test size 0.1")
        for fold in data_splits.keys():
            print(f"\n{'='*20}\nStarting {fold}\n{'='*20}")
            print(f"Train size: {len(data_splits[fold]['train']['images'])}, Val size: {len(data_splits[fold]['validation']['images'])}, Test size: {len(data_splits[fold]['test']['images'])}")
            data = data_splits[fold]

            # Generate patches for each configuration if not already done
            for key in patches.keys():
                print(f"\nGenerating patches for {key}")

                output_dir = os.path.abspath(os.path.join(os.getcwd(), f"data/{k}fold/{data_type}/{fold}_patchify_{num_classes}_classes_{data_type}_{key}/"))
                if not os.path.exists(output_dir):
                    print(f"\tCreating directory {output_dir}...")
                    os.makedirs(output_dir)
                    generate_patches(data, output_dir, patches[key]["patch_size"], patches[key]["stride"])
                    print(f"\tDone generating patches for {key}")
                else:
                    print(f"\tDirectory {output_dir} already exists. Skipping...")

        output_dir = os.path.abspath(os.path.join(os.getcwd(), f"data/{k}fold/{data_type}/"))
        export_folds_summary(data_splits, output_dir=output_dir)

        for network in networks:
            print(f"\n{'='*30}\nStarting training for {network}\n{'='*30}")
    
            for key in patches.keys():
                output_dir = os.path.abspath(os.path.join(os.getcwd(), f"data/{k}fold/{data_type}/{fold}_patchify_{num_classes}_classes_{data_type}_{key}/"))

                # Load the data split
                print(f"\nLoading data for {key}")

                model = networks[network](num_classes, experiment_name=MLFLOW_EXPERIMENT_NAME, use_mlflow=True)

                model.load_data(output_dir, CNNFormes, batch_size=24, resize_shape=patches[key]["patch_size"])

                # Training
                print(f"\nTraining model for {key}")
                run_name = f"{data_type}_{network}_{key}"
                description = f"Dataset type: {data_type}, Training {network} with patch size {key}, patch_size={patches[key]['patch_size']}, stride={patches[key]['stride']}"
                early_stopping = 10
                model.train(epochs=100, artifact_path=artifact_path, run_name=run_name, run_description=description, early_stopping=early_stopping)
                print(f"\tModel trained for {key}")

                # Clear memory
                del model
                gc.collect()
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()