import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from src.models.data_management.base_formes import BaseFormes
from typing import List, Optional, Tuple
from torch import Tensor

class UNetFormes(BaseFormes):
    """
    Dataset class for U-Net training and inference.

    This class loads images and their corresponding masks, applying the specified transformations.

    Attributes:
        imgs_path (List[str]): List of file paths for the input images.
        labels_path (List[str]): List of file paths for the corresponding masks.
        transform (A.Compose): Albumentations transformation pipeline.
        len (int): Number of samples in the dataset.
    """

    DEFAULT_TRANSFORM = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # A.Normalize(mean=(0.4288, 0.4513, 0.4601), std=(0.3172, 0.3094, 0.3120)),  # Normalization adjusted for SCLabels dataset
        ToTensorV2(),
    ])

    def __init__(self, imgs_path: List[str], labels_path: List[str] = None, transform: Optional[A.Compose] = None):
        """
        Initializes the UNetFormes dataset.

        Parameters:
            imgs_path (List[str]): List of file paths for the input images.
            labels_path (List[str]): List of file paths for the corresponding masks.
            transform (Optional[A.Compose], optional): Transformation pipeline to apply. Defaults to a standard pipeline with resizing and normalization.
        """
        super().__init__()

        self.imgs_path: List[str] = imgs_path
        if labels_path is None:
            self.labels_path: List[str] = labels_path
        self.len: int = len(self.imgs_path)

        self.transform = transform if transform else self.DEFAULT_TRANSFORM

        # self.transform: A.Compose = transform if transform else A.Compose([
        #     A.Resize(256, 256),
        #     A.Normalize(mean=(0.4288, 0.4513, 0.4601), std=(0.3172, 0.3094, 0.3120)),  # Normalization adjusted for SCLabels dataset
        #     ToTensorV2(),
        # ])


    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Returns the image and mask at the specified index.

        Parameters:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image and the mask.
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
        image_transformed = data['image']
        mask_transformed = data['mask']

        if mask_transformed is None:
            return image_transformed

        # return the path of the image and the path of the label
        return image_transformed, mask_transformed
        