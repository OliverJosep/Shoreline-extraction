import sys
import os

# Add the src directory to the path. TEMPORARY FIX
sys.path.append(os.getcwd())

from models.deeplab.deeplab import DeepLabV3
from models.duck_net.duck_net import DuckNet
from models.unet.unet import UNet
from models.attention_unet.attention_unet import Attention_UNet
from models.cnn_model import CNNModel

from src.models.data_management.cnn_formes import CNNFormes
from src.data_processing.dataset_loader import CoastData

from typing import Type

image_type_paths = {
    # "oblique": {
    #     "path": os.path.abspath(os.path.join(os.getcwd(), "data/processed_obliques_2_classes/")),
    #     "num_classes": 2
    # },
    "rectified": {
        "path": os.path.abspath(os.path.join(os.getcwd(), "data/processed_rectified_3_classes/")),
        "num_classes": 3,
        "weights_path": os.path.abspath(os.path.join(os.getcwd(), "artifacts/article/experiment2/rectified"))
    }
}

path_to_weights = os.path.abspath(os.path.join(os.getcwd(), "artifacts/"))

networks: dict[str, Type[CNNModel]] = {
    "UNet": {
        "model": UNet,
        "weights_path": "2025-10-16-19-02-57_rectified_UNet_256x256"
    },
    "AttentionUNet": {
        "model": Attention_UNet,
        "weights_path": "2025-10-16-20-56-12_rectified_AttentionUNet_256x256"
    },
    "DeepLabV3": {
        "model": DeepLabV3,
        "weights_path": "2025-10-16-22-37-47_rectified_DeepLabV3_256x256"
    },
    "DuckNet": {
        "model": DuckNet,
        "weights_path": "2025-10-17-01-46-46_rectified_DuckNet_256x256"
    }
}

patches = {
    "256x256": {
        "patch_size": (256, 256),
        "stride": (128, 128)
    }
}

def main():

    for data_type in image_type_paths:
        print(f"\n{'#'*30}\nProcessing {data_type} images\n{'#'*30}")
        data_path = image_type_paths[data_type]["path"]
        num_classes = image_type_paths[data_type]["num_classes"]

        print(f"Data path: {data_path}")

        # # Load data
        # artifact_path = os.path.abspath(os.path.join(os.getcwd(), "artifacts/"))

        data = CoastData(data_path)

        filtered_data = data.get_images() 

        filtered_data = data.split_data()


if __name__ == "__main__":
    main()