import json
import random
import os

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
    
    def get_images_and_masks(self, station_name: str = None):
        """
        Returns a list of dictionaries containing the image and mask filenames 
        for the specified coastal station or the entire dataset.

        Parameters:
        station_name (str): The name of the coastal station. If None, returns all images and masks. Default: None

        Returns:
        list: A list of dictionaries with 'image' and 'mask' full paths for each entry.
        """

        if station_name is None or station_name == 'global':
            return [{'image': os.path.join(self.data_path, "images", entry['image']['filename']), 'mask': os.path.join(self.data_path, "masks", entry['image']['mask']['filename'])} 
                for entry in self.metadata]
        
        return [{'image': os.path.join(self.data_path, "images", entry['image']['filename']), 'mask': os.path.join(self.data_path, "masks", entry['image']['mask']['filename'])} 
                for entry in self.metadata if entry['image']['site']['CSname'] == station_name]

    def split_data(self, val_size: float = 0.2, test_size: float = 0.1, random_state: int = 42):
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
            'train': {
                'images': [],
                'masks': [],
            },
            'validation': {
                'images': [],
                'masks': [],
            },
            'test': {
                'images': [],
                'masks': [],
            }
        }

        # Calculate the size of the training set
        train_size = 1 - (val_size + test_size)

        # Shuffle the data for each coastal site
        for station_name in station_names:

            # Get the image and mask filenames for the coastal site
            coast_data = self.get_images_and_masks(station_name=station_name)

            # Shuffle the data
            random.shuffle(coast_data)

            # Calculate the total number of images
            total = len(coast_data)

            print(f"Coast: {station_name}, Total size: {len(coast_data)}")

            # Training set
            start = 0
            end = int(total * train_size)
            data['train']['images'].extend([entry['image'] for entry in coast_data[start:end]])
            data['train']['masks'].extend([entry['mask'] for entry in coast_data[start:end]])

            # Validation set
            start = end
            end = int(total * (train_size + val_size))
            data['validation']['images'].extend([entry['image'] for entry in coast_data[start:end]])
            data['validation']['masks'].extend([entry['mask'] for entry in coast_data[start:end]])

            # Test set
            if test_size > 0:
                start = end
                data['test']['images'].extend([entry['image'] for entry in coast_data[start:]])
                data['test']['masks'].extend([entry['mask'] for entry in coast_data[start:]])

        return data