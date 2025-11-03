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

import torch
import gc

MLFLOW_EXPERIMENT_NAME = "shoreline_search_best_patchify"

image_type_paths = {
    # "oblique": {
    #     "path": os.path.abspath(os.path.join(os.getcwd(), "data/processed_obliques_2_classes/")),
    #     "num_classes": 2
    # },
    "rectified": {
        "path": os.path.abspath(os.path.join(os.getcwd(), "data/processed_rectified_3_classes/")),
        "num_classes": 3
    }
}

networks: dict[str, Type[CNNModel]] = {
    # "UNet": UNet,
    # "AttentionUNet": Attention_UNet,
    "DeepLabV3": DeepLabV3,
    # "DuckNet": DuckNet
}

patches = {
    # "256x256": {
    #     "patch_size": (256, 256),
    #     "stride": (128, 128)
    # },
    "256x512": {
        "patch_size": (256, 512),
        "stride": (128, 256)
    },
    "256x1024": {
        "patch_size": (256, 1024),
        "stride": (128, 512)
    }, 
    "512x512": {
        "patch_size": (512, 512),
        "stride": (256, 256)
    }
}

def generate_patches(data, output_dir, patch_size, stride):
    patchify = Patchify(patch_size, stride)
    patchify.extract_patches_and_save(data, output_dir, skip_no_shoreline=True, binary_class=True, p_keep_negative=1, padding_mode='reflect')
    

def main():

    for data_type in image_type_paths:
        print(f"\n{'#'*30}\nProcessing {data_type} images\n{'#'*30}")
        data_path = image_type_paths[data_type]["path"]
        num_classes = image_type_paths[data_type]["num_classes"]

        # Load data
        artifact_path = os.path.abspath(os.path.join(os.getcwd(), "artifacts/"))

        data = CoastData(data_path)

        filtered_data = data.get_images() 

        filtered_data = data.split_data()

        # Generate patches for each configuration if not already done
        for key in patches.keys():
            print(f"\nGenerating patches for {key}")

            output_dir = os.path.abspath(os.path.join(os.getcwd(), f"data/patchify_{num_classes}_classes_{data_type}_{key}/"))
            if not os.path.exists(output_dir):
                print(f"\tCreating directory {output_dir}...")
                os.makedirs(output_dir)
                generate_patches(filtered_data, output_dir, patches[key]["patch_size"], patches[key]["stride"])
                print(f"\tDone generating patches for {key}")
            else:
                print(f"\tDirectory {output_dir} already exists. Skipping...")

        for network in networks:
            print(f"\n{'='*30}\nStarting training for {network}\n{'='*30}")
    
            for key in patches.keys():
                output_dir = os.path.abspath(os.path.join(os.getcwd(), f"data/patchify_{num_classes}_classes_{data_type}_{key}/"))

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