import os
import cv2
import numpy as np
import shutil

import matplotlib.pyplot as plt

from typing import Tuple

from src.data_processing.dataset_preprocessor import DatasetPreprocessor

class DatasetPreprocessorBiLSTM(DatasetPreprocessor):
    def __init__(self):
        super().__init__()

    def remove_rows_with_only_one_class(self, img: np.array, mask: np.array) -> Tuple[np.array, np.array]:
        valid_rows = np.any(mask == 0, axis=1) & np.any(mask == 1, axis=1)
        return img[valid_rows], mask[valid_rows]
    
    def remove_cols_with_only_one_class(self, img: np.array, mask: np.array, background_class: int = 0, coastline_class: int = 2) -> Tuple[np.array, np.array]:
        unique_per_column = [np.unique(mask[:, col]) for col in range(mask.shape[1])]
    
        keep_columns = [i for i, unique_vals in enumerate(unique_per_column) 
                        if not set(unique_vals).issubset({background_class, coastline_class})]

        img_filtered = img[:, keep_columns, :] if img.ndim == 3 else img[:, keep_columns]
        mask_filtered = mask[:, keep_columns]
        
        return img_filtered, mask_filtered

    def to_binary_mask(self, mask: np.array, type_class: int = 2) -> np.array:
        rows_shape = mask.shape[0]
        # go column by column
        for i in range(mask.shape[1]):
            # Find the fixels that are the type_class
            rows = np.where(mask[:, i] == type_class)[0]
            for row in rows:
                # print(row)
                for j in range(0, rows_shape):
                    if row+j < rows_shape:
                        if mask[row+j, i] != type_class:
                            mask[row, i] = mask[row+j, i]
                            break
                    if row-j >= 0:
                        if mask[row-j, i] != type_class:
                            mask[row, i] = mask[row-j, i]
                            break
            # break
        return mask
    
    def add_padding(self, img: np.array, mask: np.array, max_width: int = 801) -> Tuple[np.array, np.array]:
        img_shape = img.shape
        mask_shape = mask.shape
        if img_shape[1] < max_width:
            img = np.pad(img, ((0, 0), (0, max_width - img_shape[1]), (0, 0)), mode='constant')
            mask = np.pad(mask, ((0, 0), (0, max_width - mask_shape[1])), mode='edge')
        return img, mask

    def process_image(self, img: np.array, mask: np.array, type_class: int = 255, background_class: int = 0, mask_mapping: dict = None) -> Tuple[np.array, np.array]:
        img, mask = self.get_rows_with_class(img, mask, type_class)
        img, mask = self.transform_class_to_background(img, mask, type_class = 25, background_class = background_class) # 25 is the class for the not classified pixels
        img, mask = self.remove_rows_with_background(img, mask, background_class)
        img, mask = self.remove_cols_with_background(img, mask, background_class)
        mask = self.mask_mappping(mask, mask_mapping) # 25 is the class for the not classified pixels
        img, mask = self.remove_rows_with_only_one_class(img, mask)
        # img, mask = self.remove_cols_with_only_one_class(img, mask)
        mask = self.to_binary_mask(mask, type_class = 2)
        # img, mask = self.add_padding(img, mask, max_width=801) # Not needed with batch size 1

        return img, mask
