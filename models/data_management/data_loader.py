import os
from typing import Dict, List, Union, Type
from torch.utils.data import Dataset, DataLoader

class DataLoaderManager:
    @staticmethod
    def load_data(data_source: Union[str, dict]) -> Dict[str, Dict[str, List[str]]]:
        if isinstance(data_source, str):
            return DataLoaderManager._load_from_path(data_source)
        elif isinstance(data_source, dict):
            return DataLoaderManager._load_from_dict(data_source)
        else:
            raise ValueError("data_source must be either a string (path) or a dictionary")

    @staticmethod
    def _load_from_path(data_path: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Load the data from the given path.
        
        The structure of the data should be as follows:

        data_path
        ├── train
        │   ├── images
        │   └── masks
        ├── validation
        │   ├── images
        │   └── masks
        └── test
            ├── images
            └── masks

        Parameters:
        data_path (str): The path to the data.

        Raises:
        FileNotFoundError: If the data is not found in the given path

        Returns:
        Dict[str, Dict[str, List[str]]]: 
            A dictionary where:
            - The keys are "train", "validation", and optionally "test".
            - The values are dictionaries with:
                - "images": List of image file paths.
                - "masks": List of mask file paths.
        """
        subsets = ["train", "validation", "test"]
        data: Dict[str, Dict[str, List[str]]] = {}

        for subset in subsets:
            img_dir = os.path.join(data_path, subset, "images")
            mask_dir = os.path.join(data_path, subset, "masks")

            if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
                if subset == "test":
                    print(f"Warning: No test set found in {data_path}. Skipping...")
                    continue
                else:
                    raise FileNotFoundError(f"Error: The {subset} set is missing in {data_path}")

            img_files = sorted(os.listdir(img_dir))
            mask_files = sorted(os.listdir(mask_dir))

            data[subset] = {
                "images": [os.path.join(img_dir, img) for img in img_files if img.endswith((".png", ".jpg", ".jpeg"))],
                "masks": [os.path.join(mask_dir, mask) for mask in mask_files if mask.endswith((".png", ".jpg", ".jpeg"))]
            }

        return data

    @staticmethod
    def _load_from_dict(data_dict: dict) -> Dict[str, Dict[str, List[str]]]:
        """
        Load the data from the given dictionary.

        The dictionary should have the following structure:
        
        {
            "train": {
                "images": List of image file paths,
                "masks": List of mask file paths
            },
            "validation": {
                "images": List of image file paths,
                "masks": List of mask file paths
            },
            "test": {
                "images": List of image file paths,
                "masks": List of mask file paths
            }
        }

        Parameters:
        data_dict (dict): A dictionary containing the data.

        Returns:
        Dict[str, Dict[str, List[str]]]: 
            A dictionary where:
            - The keys are "train", "validation", and optionally "test".
            - The values are dictionaries with:
                - "images": List of image file paths.
                - "masks": List of mask
        """
        required_keys = {"train", "validation"}
        optional_keys = {"test"}

        if not required_keys.issubset(data_dict.keys()):
            raise ValueError(f"Error: data_dict must contain at least {required_keys}")

        valid_sets = required_keys | optional_keys

        for subset, content in data_dict.items():
            if subset not in valid_sets:
                raise ValueError(f"Invalid key '{subset}' found in data_dict. Allowed keys: {valid_sets}")

            if not isinstance(content, dict) or "images" not in content or "masks" not in content:
                raise ValueError(f"Error: '{subset}' must contain 'images' and 'masks' as keys.")

            if not isinstance(content["images"], list) or not isinstance(content["masks"], list):
                raise TypeError(f"Error: '{subset}' -> 'images' and 'masks' must be lists.")

            if not all(isinstance(img, str) for img in content["images"]):
                raise TypeError(f"Error: All elements in '{subset}' -> 'images' must be strings.")
            if not all(isinstance(mask, str) for mask in content["masks"]):
                raise TypeError(f"Error: All elements in '{subset}' -> 'masks' must be strings.")

        return data_dict
    
    @staticmethod
    def generate_formes(X: list[str], y: list[str], formes_class: Type[Dataset]) -> Dataset:
        """
        Generate Formes dataset.

        Parameters:
        X (list[str]): List of paths images.
        y (list[str]): List of paths masks/labels.
        formes_class (Type[Dataset]): The class of the Formes dataset.

        Returns:
        Formes: The training dataset.
        """
        return formes_class(X, y)
    
    @staticmethod
    def generate_data_loaders(dataset: Dataset, batch_size: int = 16, shuffle: bool = False) -> DataLoader:
        """
        Generate data loaders.

        Parameters:
        batch_size (int, optional): The batch size to use. Default is 16.
        shuffle (bool, optional): Whether to shuffle the data. Default is False.

        Returns:
        DataLoader: The data loader.
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
