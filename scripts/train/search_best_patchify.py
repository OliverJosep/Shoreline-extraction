import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the path. TEMPORARY FIX
sys.path.append(os.getcwd())

from models.deeplab.deeplab import DeepLabV3
from src.models.data_management.cnn_formes import CNNFormes
from src.data_processing.dataset_loader import CoastData
from src.data_processing.patchify import Patchify

patches = {
    "256x256": {
        "patch_size": (256, 256),
        "stride": (128, 128)
    },
    # "256x512": {
    #     "patch_size": (256, 512),
    #     "stride": (128, 256)
    # },
    # "256x1024": {
    #     "patch_size": (256, 1024),
    #     "stride": (128, 512)
    # }, 
    # "512x512": {
    #     "patch_size": (512, 512),
    #     "stride": (256, 256)
    # },
    # "512x1024": {
    #     "patch_size": (512, 1024),
    #     "stride": (256, 512)
    # },
    # "1024x1024": {
    #     "patch_size": (1024, 1024),
    #     "stride": (512, 512)
    # }
}

def generate_patches(data, output_dir, patch_size, stride):
    patchify = Patchify(patch_size, stride)
    patchify.extract_patches_and_save(data, output_dir, skip_no_shoreline=True, binary_class=True, p_keep_negative=1, padding_mode='reflect')
    

def main():
    # Load data
    data_path = os.path.abspath(os.path.join(os.getcwd(), "data/processed_obliques_2_classes/"))
    artifact_path = os.path.abspath(os.path.join(os.getcwd(), "artifacts/"))

    data = CoastData(data_path)

    filtered_data = data.get_images_and_masks() 

    filtered_data = data.split_data()

    for key in patches.keys():
        print(f"\nGenerating patches for {key}")
        
        output_dir = os.path.abspath(os.path.join(os.getcwd(), f"data/patchify_2_classes_oblique_{key}/"))
        if not os.path.exists(output_dir):
            print(f"\tCreating directory {output_dir}...")
            os.makedirs(output_dir)
            generate_patches(filtered_data, output_dir, patches[key]["patch_size"], patches[key]["stride"])
            print(f"\tDone generating patches for {key}")
        else:
            print(f"\tDirectory {output_dir} already exists. Skipping...")

        # Load the data split
        print(f"\nLoading data for {key}")
        num_classes = 2
        model = DeepLabV3(num_classes, experiment_name="shoreline_oblique_patchify", use_mlflow=False, pretrained=False)
        model.load_data(output_dir, CNNFormes, batch_size=2, resize_shape=patches[key]["patch_size"])

        # Training
        print(f"\nTraining model for {key}")
        description = f"Training DeepLabV3 with patch size {key}, patch_size={patches[key]['patch_size']}, stride={patches[key]['stride']}"
        model.train(epochs=1, artifact_path=artifact_path, run_description=description)
        print(f"\tModel trained for {key}")

if __name__ == "__main__":
    main()