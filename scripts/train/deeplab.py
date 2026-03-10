import sys
import os

# run this script from the ./Shoreline-extraction directory to avoid import errors
sys.path.append(os.getcwd())

from models.deeplab.deeplab import DeepLabV3
from models.cnn_model import CNNModel

from src.models.data_management.cnn_formes import CNNFormes
from src.data_processing.dataset_loader import CoastData
from src.data_processing.patchify import Patchify

import torch
import gc

MLFLOW_EXPERIMENT_NAME = "experiment_name"

def generate_patches(data, output_dir, patch_size, stride):
    patchify = Patchify(patch_size, stride)
    patchify.extract_patches_and_save(data, output_dir, skip_no_shoreline=True, binary_class=True, p_keep_negative=1, padding_mode='reflect')
    
def main():
    data_path = os.path.abspath(os.path.join(os.getcwd(), "data/processed/"))
    num_classes = 3

    # Load data
    print(f"Loading data from {data_path}...")
    data = CoastData(data_path, name="samarador")
    filtered_data = data.get_images() 
    print("Data loaded successfully.")
    print(f"Splitting data into training, validation and test.")
    filtered_data = data.split_data()
    print("Data split successfully.")

    # Generate output directory for patches
    artifact_path = os.path.abspath(os.path.join(os.getcwd(), "artifacts/"))
    output_dir = os.path.join(artifact_path, MLFLOW_EXPERIMENT_NAME, "patches")
    print(f"Output directory for patches: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define patch size and stride
    patch_size = (256, 256)
    stride = (256, 256)
    print(f"Generating patches with patch size {patch_size} and stride {stride}...")
    generate_patches(filtered_data, output_dir, patch_size, stride)
    print("Patches generated successfully.")

    # Training the model
    print(f"\n{'='*30}\nStarting training for DeepLabV3\n{'='*30}")
    
    model = DeepLabV3(num_classes=num_classes, experiment_name=MLFLOW_EXPERIMENT_NAME, use_mlflow=True)
    model.load_data(output_dir, CNNFormes, batch_size=24, resize_shape=patch_size)
    print("Data loaded into model successfully.")
    print("Starting training...")

    run_name = f"DeepLabV3_{patch_size[0]}x{patch_size[1]}"
    description = f"Training DeepLabV3 with patch size {patch_size} and stride {stride}"

    early_stop = 10
    model.train(epochs=100, artifact_path=artifact_path, run_name=run_name, run_description=description, early_stopping=early_stop)
    print("Training completed successfully.")

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()