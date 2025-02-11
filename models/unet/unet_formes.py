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

        # return the path of the image and the path of the label
        return self.imgs_path[index], self.labels_path[index]
        