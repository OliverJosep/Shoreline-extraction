import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from models.data_management.base_formes import BaseFormes
from typing import List

class UNetFormes(BaseFormes):

    def __init__(self, imgs_path: List[str], labels_path: List[str]):
        super().__init__()

        self.imgs_path = imgs_path
        self.labels_path = labels_path

        self.len = len(self.imgs_path)

    def __getitem__(self, index):
        # TODO Return the imag and the mask and apply the transformations
        transform = A.Compose([
            A.Resize(256, 256),
            # A.HorizontalFlip(),
            # A.RandomRotate90(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize the image TODO: Check the mean and std with the dataset
            ToTensorV2(),
        ])

        # load the image
        img = cv2.imread(self.imgs_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load the mask
        mask = cv2.imread(self.labels_path[index], cv2.IMREAD_GRAYSCALE)

        # apply the transformations
        data = transform(image=img, mask=mask)
        image_transformed = data['image']
        mask_transformed = data['mask']

        # return the path of the image and the path of the label
        return image_transformed, mask_transformed
        