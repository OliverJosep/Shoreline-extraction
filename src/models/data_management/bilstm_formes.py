import cv2
import torch
from typing import List, Optional, Tuple
from torch import Tensor
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BiLSTMFormesDataset(torch.utils.data.Dataset):

    DEFAULT_TRANSFORM = A.Compose([
        # A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # A.Normalize(mean=(0.4288, 0.4513, 0.4601), std=(0.3172, 0.3094, 0.3120)),  # Normalization adjusted for SCLabels dataset
        ToTensorV2(),
    ])

    def __init__(self, imgs_path: List[str], labels_path: List[str] = None, transform: Optional[A.Compose] = None):
        """
        Initializes the CNNFormes dataset.

        Parameters:
            imgs_path (List[str]): List of file paths for the input images.
            labels_path (List[str]): List of file paths for the corresponding masks.
            transform (Optional[A.Compose], optional): Transformation pipeline to apply. Defaults to a standard pipeline with resizing and normalization.
        """
        super().__init__()

        self.imgs_path: List[str] = imgs_path
        self.labels_path: Optional[List[str]] = labels_path or None
        self.len: int = len(self.imgs_path)

        self.transform = transform if transform else self.DEFAULT_TRANSFORM

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Returns the image and mask at the specified index, divided by rows.

        Parameters:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image rows and the corresponding mask rows.
        """

        # load the image
        img = cv2.imread(self.imgs_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load the mask
        if self.labels_path is not None:
            mask = cv2.imread(self.labels_path[index], cv2.IMREAD_GRAYSCALE)
        else:
            mask = None

        # apply the transformations
        data = self.transform(image=img, mask=mask)
        img_transformed = data['image']
        mask_transformed = data['mask']

        # Transform the image and mask into rows
        img_rows = img_transformed.permute(1, 2, 0)  # (C, H, W) --> (H, W, C)

        if mask_transformed is None:
            return img_rows
            
        return img_rows, mask

    def __len__(self):
        return self.len