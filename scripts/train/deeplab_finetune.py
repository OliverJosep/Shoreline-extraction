import os
import sys

# run this script from the ./Shoreline-extraction directory to avoid import errors
sys.path.append(os.getcwd())

import gc
import torch

from models.deeplab.deeplab import DeepLabV3
from src.data_processing.dataset_loader import CoastData
from src.data_processing.patchify import Patchify
from src.models.data_management.cnn_formes import CNNFormes

# ---------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "DeepLabV3_Finetuning"
PRETRAINED_MODEL_PATH = "artifacts/ai4os-hub/v0.1.1/rectified_best_model.pth" 


def generate_patches(data, output_dir, patch_size, stride):
    patchify = Patchify(patch_size, stride)
    patchify.extract_patches_and_save(
        data,
        output_dir,
        skip_no_shoreline=True,
        binary_class=True,
        p_keep_negative=0.1,
        padding_mode="reflect",
    )

def merge_datasets(base_data, new_data):
    """
    Recursively merges new_data into base_data.
    If it finds dictionaries, it explores them.
    If it finds lists, it properly extends them.
    """
    if base_data is None:
        return new_data
        
    if isinstance(base_data, dict) and isinstance(new_data, dict):
        for key, value in new_data.items():
            if key in base_data:
                base_data[key] = merge_datasets(base_data[key], value)
            else:
                base_data[key] = value
        return base_data
    elif isinstance(base_data, list) and isinstance(new_data, list):
        base_data.extend(new_data)
        return base_data
    else:
        # For primitive types or None, just overwrite
        return new_data

def main():
    # 1. Define the datasets you want to use for FINE-TUNING
    # (Usually this is a new unseen dataset, or a specific local beach)
    data_paths = [
        os.path.abspath(os.path.join(os.getcwd(), "data/processed_UNSW/")),
    ]
    num_classes = 3
    
    combined_filtered_data = None

    # 2. Iterate through each directory to load and merge the data
    for idx, path in enumerate(data_paths):
        print(f"[{idx+1}/{len(data_paths)}] Loading data from {path}...")
        
        # Assign a unique name to each dataset to avoid potential conflicts
        data = CoastData(path)
        filtered_data = data.get_images() 
        print(f"Data from {path} loaded successfully.")
        
        print(f"Splitting data from {path} into training, validation and test.")
        filtered_data = data.split_data()
        
        # 3. Accumulate the split data using the recursive merge function
        combined_filtered_data = merge_datasets(combined_filtered_data, filtered_data)
        
        # PRINT TO SEE THE ACTUAL NUMBER OF IMAGES
        print(f"--> Current total accumulated images:")
        for split_key in ["train", "validation", "test"]:
            if split_key in combined_filtered_data:
                num_images = len(combined_filtered_data[split_key]["images"])
                print(f"    {split_key.capitalize()}: {num_images} images")

    # Generate output directory for patches
    artifact_path = os.path.abspath(os.path.join(os.getcwd(), "artifacts/"))
    output_dir = os.path.join("data", MLFLOW_EXPERIMENT_NAME, "patches")
    print(f"Output directory for patches: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define patch size and stride
    patch_size = (512, 512)
    stride = (256, 256)
    
    if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
        print(f"Generating patches with patch size {patch_size} and stride {stride}...")
        
        # Pass the merged data to the patch generator
        generate_patches(combined_filtered_data, output_dir, patch_size, stride)
        print("Patches generated successfully.")
    else:
        print("Patches already exist. Skipping patch generation.")

    # Initialize the model
    print(f"\n{'='*30}\nInitializing DeepLabV3 for Fine-Tuning\n{'='*30}")

    model = DeepLabV3(
        num_classes=num_classes,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        use_mlflow=True,
    )
    
    # =========================================================================
    # FINE-TUNING CONFIGURATION
    # =========================================================================
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Loading pre-trained weights from: {PRETRAINED_MODEL_PATH}")

        try:
            # Assuming the PyTorch model is stored in `model.model`
            model.model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
            print("Pre-trained weights loaded successfully.")
        except AttributeError:
            print("Warning: Could not find 'model.model'. Please update the script with the correct attribute name for the PyTorch module in your DeepLabV3 class.")
        
        # OPTIONAL: Freeze the backbone
        # If you only want to train the final classification head and keep the base features intact.
        # Uncomment the following lines if you want to freeze it:
        #
        # print("Freezing the ResNet50 backbone...")
        # for param in model.model.backbone.parameters():
        #     param.requires_grad = False
        #     print("Backbone layer frozen.")

        # for param in model.model.classifier.parameters():
        #     param.requires_grad = True
        #     print("Classifier layer set to trainable.")

        print("Freezing the entire network...")
        for param in model.model.parameters():
            param.requires_grad = False
            
        print("Unfreezing ONLY the final classification layer (classifier[4])...")
        for param in model.model.classifier[4].parameters():
            param.requires_grad = True

    else:
        print(f"WARNING: Pre-trained model not found at {PRETRAINED_MODEL_PATH}. Training from scratch!")
    # =========================================================================

    # Load data into the model structure
    model.load_data(
        output_dir,
        CNNFormes,
        batch_size=4,
        resize_shape=patch_size,
        drop_last=True,
    )
    print("Data loaded into model successfully.")
    print("Starting fine-tuning...")

    run_name = f"DeepLabV3_Finetune_{patch_size[0]}x{patch_size[1]}"
    description = f"Fine-tuning DeepLabV3 on new data. Pre-trained on SCLabels+UNSW."

    # For fine-tuning, you often need fewer epochs or a smaller learning rate
    early_stop = 15
    model.train(
        epochs=50, # Usually fewer epochs are needed for fine-tuning
        artifact_path=artifact_path,
        run_name=run_name,
        run_description=description,
        early_stopping=early_stop,
    )
    print("Fine-tuning completed successfully.")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()