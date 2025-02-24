import numpy as np
import torch
import torch.types as T
from typing import List, Dict

class PatchReconstructor():

    @staticmethod
    def combine_patches_avg(output: T.Tensor, n_classes: int, original_heigh: int, original_width: int, patch_size: int = 256, stride: int = 128) -> T.Tensor:
        reconstructed = np.zeros((n_classes, original_heigh, original_width), dtype=np.float32)
        count_map = np.zeros((original_heigh, original_width), dtype=np.float32)

        idx = 0
        for y in range(0, original_heigh - patch_size + 1, stride):
            for x in range(0, original_width - patch_size + 1, stride):
                output_np = output[idx].detach().cpu().numpy()  # Transform to numpy
                reconstructed[:, y:y+patch_size, x:x+patch_size] += output_np
                count_map[y:y+patch_size, x:x+patch_size] += 1
                idx += 1

        reconstructed /= count_map
        return T.Tensor(reconstructed)
    
    @staticmethod
    def combine_patches_max(output: T.Tensor, n_classes: int, original_heigh: int, original_width: int, patch_size: int = 256, stride: int = 128) -> T.Tensor:
        reconstructed = np.zeros((n_classes, original_heigh, original_width), dtype=np.float32)

        idx = 0
        for y in range(0, original_heigh - patch_size + 1, stride):
            for x in range(0, original_width - patch_size + 1, stride):
                output_np = output[idx].detach().cpu().numpy()  # Transform to numpy
                reconstructed[:, y:y+patch_size, x:x+patch_size] = np.maximum(reconstructed[:, y:y+patch_size, x:x+patch_size], output_np)
                idx += 1

        return T.Tensor(reconstructed)