import json
import random
import os
import numpy as np

from sklearn.model_selection import StratifiedKFold
import numpy as np
import random

from collections import defaultdict

class CoastData:
    """
    Dataset class for coastal images and metadata. Designed for SCLabels dataset.
    """

    def __init__(self, data_path: str, name: str = "global", metadata_name: str = "metadata.json", verbose: bool = True):
        """
        Initializes the CoastData object.

        Parameters:
        data_path (str): The path to the directory where the data is stored. This is the base path used to locate the metadata file and image data. 
        name (str): The name of the dataset or a specific subset of it. It can be a global dataset or a specific coastal site. Default: global
        metadata_name (str): The name of the metadata file containing information about the images. This file includes details such as the location, conditions, and other relevant metadata for each image. Default: metadata.json
        verbose (bool): If True, prints information about the dataset. Default: True

        Returns:
        None
        """

        # Set the name of the dataset
        self.name = name

        # Load the metadata
        full_metadata = self._get_metadata(data_path, metadata_name)

        # Filter the metadata based on the name
        # self.global_data = False
        if 'global' in name:
            self.metadata = full_metadata
            # self.global_data = True
        else:
            self.metadata = [entry for entry in full_metadata if entry['image']['site']['CSname'] in name]

        # Set the data path
        self.data_path = data_path

        if verbose:
            print(f"CoastData: {name} - {len(self.metadata)} images")

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.metadata)
    
    def __getitem__(self, idx: int):
        """
        Returns the metadata for a specific image index in the dataset.

        Parameters:
        idx (int): The index of the image in the dataset.

        Returns:
        dict: The metadata for the image at the specified index.
        """
        return self.metadata[idx]
    
    def _get_metadata(self, data_path: str, metadata_name: str = "metadata.json"):
        """
        Loads the metadata from the specified file.

        Parameters:
        data_path (str): The path to the directory where the data is stored.
        metadata_name (str): The name of the metadata file containing information about the images. Default: metadata.json

        Returns:
        dict: The metadata for the dataset.
        """

        try:
            full_path = f"{data_path}/{metadata_name}"
            with open(full_path, "r") as fp:
                metadata = json.load(fp)
            return metadata
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file {metadata_name} not found in {data_path}.")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from {metadata_name}.")
    
    def get_station_names(self):
        """
        Returns a sorted list of the coastal site names in the dataset.
        """

        return sorted(list(set([entry['image']['site']['CSname'] for entry in self.metadata])))

    def get_images(self, station_name: str = None, get_mask: bool = True, get_shoreline_coords: bool = False, get_all_metadata: bool = False, get_filename: bool = False, get_original_image_path: bool = False):
        """
        Returns a list of image filenames for the specified coastal station or the entire dataset. Optionally includes masks, shoreline coordinates, original image paths, and all metadata.

        Parameters:
        station_name (str): The name of the coastal station. If None, returns all images. Default: None
        get_mask (bool): If True, includes the mask file paths in the results. Default: True
        get_shoreline_coords (bool): If True, includes the shoreline coordinates in the results. Default: False
        get_all_metadata (bool): If True, includes all metadata in the results. Default: False
        get_filename (bool): If True, includes only the image filename in the results. Default: False
        get_original_image_path (bool): If True, includes the original image path in the results. Default: False

        Returns:
        list: A list of image full paths.
        """
        if station_name is None or station_name == 'global':
            entries_to_process = self.metadata
        else:
            entries_to_process = [
                entry for entry in self.metadata 
                if entry['image']['site']['CSname'] == station_name
            ]

        results = []
        for entry in entries_to_process:
            image_data = entry['image']
        
            item = {
                'image': os.path.join(self.data_path, "images", image_data['filename'])
            }
            if get_mask:
                item['mask'] = os.path.join(self.data_path, "masks", image_data['mask']['filename'])
            if get_shoreline_coords:
                item['shoreline_coords'] = image_data['shoreline']['coordinates']
            if get_original_image_path:
                item['original_image_path'] = os.path.basename(image_data['oblique_path'])
            if get_filename:
                item['filename'] = image_data['filename']
            if get_all_metadata:
                item['metadata'] = entry
            
            results.append(item)

        return results
    
    def split_data(self, val_size: float = 0.2, test_size: float = 0.1, random_state: int = 42, get_mask: bool = True, get_coords: bool = False, get_metadata: bool = False):
        """
        Splits the dataset into training, validation, and test sets based on the specified sizes.

        Parameters:
        val_size (float): The size of the validation set as a percentage of the total dataset. Default: 0.2
        test_size (float): The size of the test set as a percentage of the total dataset. Default: 0.1
        random_state (int): The random seed to use for shuffling the dataset. Default: 42

        Returns:
        list: A list of tuples containing the image and mask filenames for the training, validation, and test sets.
        """

        # Set the random seed
        random.seed(random_state)

        # Get the list of coastal site names
        station_names = self.get_station_names()

        # Initialize the lists for the training, validation, and test sets
        data = {
            'train': self._init_empty_dict(get_coords, get_metadata),
            'validation': self._init_empty_dict(get_coords, get_metadata),
            'test': self._init_empty_dict(get_coords, get_metadata)
        }

        # Calculate the size of the training set
        train_size = 1 - (val_size + test_size)

        # Shuffle the data for each coastal site
        for station_name in station_names:

            coast_data = self.get_images(
                station_name=station_name, 
                get_mask=get_mask, 
                get_shoreline_coords=get_coords, 
                get_all_metadata=get_metadata,
                get_original_image_path=get_coords
            )

            # Shuffle the data
            random.shuffle(coast_data)

            # Calculate the total number of images
            total = len(coast_data)

            print(f"Coast: {station_name}, Total size: {len(coast_data)}")

            # Training set
            start = 0
            end = int(total * train_size)
            self._extend_data(data['train'], coast_data[start:end], get_mask, get_coords, get_metadata)

            # Validation set
            start = end
            end = int(total * (train_size + val_size))
            self._extend_data(data['validation'], coast_data[start:end], get_mask, get_coords, get_metadata)

            # Test set
            if test_size > 0:
                start = end
                self._extend_data(data['test'], coast_data[start:], get_mask, get_coords, get_metadata)
        return data

    def get_kfold_splits(self, k=5, test_size: float = 0.1, random_state=42, get_mask=True, get_coords=False, get_metadata=False):
        """
        Crea k splits (train/val) per K-Fold Cross-Validation mantenint el test fix.
        """
        # Set the random seed
        random.seed(random_state)

        # Get the list of coastal site names
        station_names = self.get_station_names()

        # Initialize the lists for the training, validation, and test sets
        data = {
            'test': self._init_empty_dict(get_coords, get_metadata)
        }

        for fold in range(k):
            data[f'fold_{fold}'] = self._init_empty_dict(get_coords, get_metadata)

        # Shuffle the data for each coastal site
        for station_name in station_names:
            coast_data = self.get_images(
                station_name=station_name,
                get_mask=get_mask,
                get_shoreline_coords=get_coords,
                get_all_metadata=get_metadata,
                get_original_image_path=get_coords
            )

            # Shuffle the data
            random.shuffle(coast_data)
            
            # Calculate the total number of images
            total = len(coast_data)
            print(f"Coast: {station_name}, Total size: {len(coast_data)}")

            # add last images to test set
            if test_size > 0:
                start = int(total * (1 - test_size))
                self._extend_data(data['test'], coast_data[start:], get_mask, get_coords, get_metadata)
                coast_data = coast_data[:start]

            total_folds = int(total * (1 - test_size))
            fold_size = total_folds // k

            for fold in range(k):
                start = fold * fold_size
                end = start + fold_size if fold != k - 1 else total_folds
                self._extend_data(data[f'fold_{fold}'], coast_data[start:end], get_mask, get_coords, get_metadata)

        new_folds = {}
        for fold in range(k):
            new_folds[f'fold_{fold}'] = {
                'train': self._init_empty_dict(get_coords, get_metadata),
                'validation': data[f'fold_{fold}'],
                'test': data['test']
            }

            for other_fold in range(k):
                if other_fold == fold:
                    continue
                for key, values in data[f'fold_{other_fold}'].items():
                    if values is None:
                        continue
                    if new_folds[f'fold_{fold}']['train'][key] is None:
                        new_folds[f'fold_{fold}']['train'][key] = []
                    new_folds[f'fold_{fold}']['train'][key].extend(values)

        return new_folds

    def _init_empty_dict(self, get_coords: bool, get_metadata: bool):
        return {
            'images': [],
            'masks': [],
            'original_image_paths': [] if get_coords else None,
            'metadata': [] if get_metadata else None
        }
    
    def _extend_data(self, data_dict, chunk, get_mask, get_coords, get_metadata):
        data_dict['images'].extend([entry['image'] for entry in chunk])
        if get_coords:
            data_dict['masks'].extend([entry['shoreline_coords'] for entry in chunk])
            data_dict['original_image_paths'].extend([entry['original_image_path'] for entry in chunk])
        if get_mask:
            data_dict['masks'].extend([entry['mask'] for entry in chunk])
        if get_metadata:
            data_dict['metadata'].extend([entry['metadata'] for entry in chunk])