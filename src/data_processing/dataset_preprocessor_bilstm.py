import os
import cv2
import numpy as np
import shutil

from typing import Tuple

from src.data_processing.dataset_preprocessor import DatasetPreprocessor

class DatasetPreprocessorBiLSTM(DatasetPreprocessor):
    def __init__(self):
        super().__init__()

    def remove_rows_with_only_one_class(self, img: np.array, mask: np.array) -> Tuple[np.array, np.array]:
        for i in range(mask.shape[0]):
            # class1 = np.where(mask[i, :] == 0)[0]
            # print(mask.shape[0])
            # print(i)
            # print(mask[i, :])
            if i >= mask.shape[0]:
                print(f"Índice {i} fuera de rango para la máscara con tamaño {mask.shape[0]}")
                continue
            class1 = np.where(mask[i, :] == 0)
            class2 = np.where(mask[i, :] == 1)
            # class2 = np.where(mask[i, :] == 1)[0]

            if len(class1) > 0:
                class1 = class1[0]
            if len(class2) > 0:
                class2 = class2[0]

            # print(f"Row {i} has {len(class1)} class 0 and {len(class2)} class 1")
            if len(class1) == 0 or len(class2) == 0:
                # print(f"Removing row {i}")
                # remove row
                img = np.delete(img, i, axis=0)
                mask = np.delete(mask, i, axis=0)
        return img, mask


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

    def process_image(self, img: np.array, mask: np.array, type_class: int = 255, background_class: int = 0, mask_mapping: dict = None) -> Tuple[np.array, np.array]:
        img, mask = self.get_rows_with_class(img, mask, type_class)
        img, mask = self.transform_class_to_background(img, mask, type_class = 25, background_class = background_class) # 25 is the class for the not classified pixels
        img, mask = self.remove_rows_with_background(img, mask, background_class)
        img, mask = self.remove_cols_with_background(img, mask, background_class)
        mask = self.mask_mappping(mask, mask_mapping) # 25 is the class for the not classified pixels
        img, mask = self.remove_rows_with_only_one_class(img, mask)
        mask = self.to_binary_mask(mask, type_class = 2)

        return img, mask
