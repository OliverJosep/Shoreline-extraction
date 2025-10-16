import numpy as np
import torch
import os
import cv2
import tempfile

from models.base_model import BaseModel

import matplotlib.pyplot as plt

# Model imports
from models.deeplab.deeplab import DeepLabV3
from models.unet.unet import UNet
from models.attention_unet import attention_unet
from models.duck_net import duck_net

# Data processing imports
from src.data_postprocessing import obtain_shoreline
from src.data_processing.crop import crop, apply_masks, merge_image_with_mask, merge_masks
from src.data_processing.dataset_preprocessor import DatasetPreprocessor

class ShorelinePredictor:
    def __init__(self, model: str, model_path: str = None, num_classes: int = 2):
        self.model = self._select_model(model, num_classes)

        self._load_model(model_path)

    def _select_model(self, model_name: str, num_classes: int) -> BaseModel:
        if model_name.lower() == "deeplabv3":
            return DeepLabV3(num_classes=num_classes)
        elif model_name.lower() == "unet":
            return UNet(num_classes=num_classes)
        elif model_name.lower() == "attention_unet":
            return attention_unet.Attention_UNet(num_classes=num_classes)
        elif model_name.lower() == "ducknet":
            return duck_net.DuckNet(num_classes=num_classes)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")
        
    def _load_model(self, model_path: str):
        if model_path is not None:
            self.model.load_model(model_path)

    def _predict(self, img: np.ndarray, crop_coords: tuple, patch_size: tuple, stride: tuple, landward_pixel: int, seaward_pixel: int) -> np.ndarray:

        print(crop_coords)

        # 1. Extract the ROI from the input image
        roi = crop(img, crop_coords[0], crop_coords[1])

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save cropped image to temporary directory
            temp_path = os.path.join(temp_dir, "temp.jpg")
            cv2.imwrite(temp_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

            # 2. Predict on the cropped image
            pred = self.model.predict_patch(temp_path, combination="avg", patch_size=patch_size, stride=stride, padding_mode="reflect")

        # 3. Post-process and extract the shoreline
        merged_img_with_pred = merge_image_with_mask(roi, pred, alpha=0.7)
        pred_np = pred.cpu().numpy().astype(np.uint8)

        # Get shoreline from prediction
        mask_pred = obtain_shoreline.transform_mask_to_shoreline_from_img(pred_np, landward=landward_pixel, seaward=seaward_pixel) # TODO: Analyse this code because is very slow

        # 4. Merge with original image
        final_img = apply_masks(merged_img_with_pred, mask_pred, shoreline_pixel_predicted_mask=1)
        img_with_pred = merge_masks(img, final_img, crop_coords[0], crop_coords[1])

        full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        full_mask = merge_masks(full_mask, mask_pred, crop_coords[0], crop_coords[1])

        # 5. Obtain coords of the shoreline pixels
        shoreline_coords = np.column_stack(np.where(full_mask == 1))
        shoreline_coords = self.format_coordinates(shoreline_coords)
        
        output = {
            "predicted_image": img_with_pred,
            "shoreline_mask": full_mask,
            "shoreline_coords": shoreline_coords
        }

        return output


    def predict_roi(self, image_path: str, crop_coords: tuple, patch_size: tuple, stride: tuple) -> np.ndarray:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = self._predict(img, crop_coords, patch_size, stride, landward_pixel=0, seaward_pixel=1)
        return pred

    def predict_rectified_with_mask(self, image_path: str, mask_path: str,patch_size: tuple = (256, 512), stride: tuple = (128, 256)) -> np.ndarray:
        """
        Explicit method for SCLabels dataset with rectified images and masks. Ideally is designed to extract the ROI based on the mask provided to be able to compare the results with the ground truth.
        """

        dataset_preprocessor = DatasetPreprocessor()
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mapping = {
            0: 0,    # Background → Class 0
            25: 3,   # Not classified → Class 1
            75: 1,   # Land → Class 2
            150: 2,  # Sea → Class 3
            255: 1   # Shoreline → Class 4
        }

        new_mask = dataset_preprocessor.mask_mapping(mask, mapping)

        target_classes = [1, 2] # Land and Sea
        hard_ignore = 3 # NotClassified

        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = self.extract_roi_and_bbox_strict(new_mask, target_classes, hard_ignore)

        crop = ((bbox_y_min, bbox_x_min), (bbox_y_max, bbox_x_max))

        pred = self._predict(img, crop, patch_size, stride, landward_pixel=1, seaward_pixel=2)

        return pred

    def format_coordinates(self, shoreline_coords: np.ndarray) -> dict:
        # Check if the array is empty to avoid errors
        if shoreline_coords.size == 0:
            return {"coordinates": {"u": [], "v": []}}

        # The first column (index 0) from np.where is 'y' or 'v'
        v_coords = shoreline_coords[:, 0].tolist()
        
        # The second column (index 1) is 'x' or 'u'
        u_coords = shoreline_coords[:, 1].tolist()

        return {
            "coordinates": {
                "u": u_coords,
                "v": v_coords
            }
        }
    
    def extract_roi_and_bbox_strict(self, mask, target_class_ids, hard_ignore_class_id):
        """
        Extracts a ROI by strictly excluding any row containing a 'hard ignore'
        pixel, then finds the bounding box for target classes within that ROI.

        This logic assumes that any row with a 'hard_ignore_class_id'
        (e.g., NotClassified) is completely invalid.

        Args:
            mask (np.ndarray): A 2D numpy array representing the segmentation mask.
            target_class_ids (list): Class IDs for the final bounding box
                                    (e.g., [1, 2]; Where 1 = Landwards, 2 = Seawards).
            hard_ignore_class_id (int): A class ID that invalidates any row
                                        it appears in.

        Returns:
            tuple or None: A tuple (x_min, y_min, x_max, y_max) representing the
                        bounding box coordinates, or None if no pixels with the
                        specified IDs are found.
        """
        # --- 1. Identify all fundamentally valid rows ---
        # A row is valid if it does NOT contain any 'hard_ignore_class_id' pixels.
        # np.any checks if any element in a row matches the condition.
        # The '~' inverts the result, so we get True for rows with NO hard ignores.
        is_valid_row = ~np.any(mask == hard_ignore_class_id, axis=1)

        # Find the start and end of the largest contiguous block of valid rows
        valid_row_indices = np.where(is_valid_row)[0]

        if valid_row_indices.size == 0:
            return None # No valid rows found at all

        first_valid_row = valid_row_indices.min()
        last_valid_row = valid_row_indices.max()

        # Create a view of the mask containing only this valid block of rows
        valid_data_mask = mask[first_valid_row : last_valid_row + 1, :]
        y_trim_offset = first_valid_row

        # --- 2. Find the bounding box for target classes within the valid ROI ---
        y_indices_relative, x_indices = np.where(np.isin(valid_data_mask, target_class_ids))

        if y_indices_relative.size == 0:
            return None # No target pixels found in the valid region

        x_min = x_indices.min()
        x_max = x_indices.max()
        y_min_relative = y_indices_relative.min()
        y_max_relative = y_indices_relative.max()

        # --- 3. Prepare the final output ---
        bbox_x_min = x_min
        bbox_y_min = y_min_relative + y_trim_offset
        bbox_x_max = x_max
        bbox_y_max = y_max_relative + y_trim_offset

        return (bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max)