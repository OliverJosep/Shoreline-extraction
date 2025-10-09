import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the path. TEMPORARY FIX
sys.path.append(os.getcwd())

from models.deeplab.deeplab import DeepLabV3
from src.models.data_management.cnn_formes import CNNFormes

from src.data_postprocessing import obtain_shoreline
from src.data_processing.crop import crop, apply_masks, merge_image_with_mask, merge_masks

import cv2
import tempfile
import numpy as np

def main():

    WEIGHTS_PATH = os.path.join(os.getcwd(), "artifacts/2025-10-02-10-54-16_2_classes_256x1024_b24/models/best_model.pth")
    folder_path_to_predict = r"" # path to the folder with oblique images to predict
    folder_path_to_save_predictions = r"" # path to the folder to save the predictions

    num_classes = 2
    model = DeepLabV3(num_classes=num_classes, experiment_name="obliques", use_mlflow=False)
    model.load_model(WEIGHTS_PATH)

    list_img = sorted([os.path.join(folder_path_to_predict, f) for f in os.listdir(folder_path_to_predict) if f.endswith(".jpg")])

    # create the folder to save predictions if it does not exist
    if not os.path.exists(folder_path_to_save_predictions):
        os.makedirs(folder_path_to_save_predictions)
    
    CROPS = {
        "sao": ((380,800), (635,1500))
    }
    

    for index in range(len(list_img)):
        # Read image
        img = cv2.imread(list_img[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save cropped image to temporary directory
            img_cropped = crop(img, CROPS["sao"][0], CROPS["sao"][1])
            temp_path = os.path.join(temp_dir, "temp.jpg")
            cv2.imwrite(temp_path, cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))

            # Predict on cropped image
            pred = model.predict_patch(temp_path, combination="avg", patch_size = (256, 1024), stride = (128, 512), padding_mode="reflect")

        # Post-process and save the prediction
        merged_img_with_pred = merge_image_with_mask(img_cropped, pred, alpha=0.7)
        pred_np = pred.cpu().numpy().astype(np.uint8)

        # Get shoreline from prediction
        mask_pred = obtain_shoreline.transform_mask_to_shoreline_from_img(pred_np, landward=0, seaward=1)

        # Apply masks to the merged image
        img_with_pred = apply_masks(merged_img_with_pred, mask_pred, shoreline_pixel_predicted_mask=1)

        # Merge with original image
        final_img = merge_masks(img, img_with_pred, CROPS["sao"][0], CROPS["sao"][1])

        # Save final image
        output_path = os.path.join(folder_path_to_save_predictions, f"pred_{os.path.basename(list_img[index])}")
        cv2.imwrite(output_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))        

if __name__ == "__main__":
    main()