import os
import pickle
from typing import List

import mne
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

from config_handling import load_config

DATA_FILE_ENDING = '.pnd'
METADATA_FILE_ENDING = '.pkl'

# From DN3 utils
def min_max_normalize(x: Tensor, low: int = -1, high: int = 1):
    """
    Normalize a tensor between low and high.

    Args:
        x: Tensor to normalize.
        low: Lower bound.
        high: Upper bound.

    Returns:
        Normalized tensor.
    """
    xmin = x.min()
    xmax = x.max()
    if xmax - xmin == 0:
        x *= 0
        return x
    
    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    return (high - low) * x

def load_raw_data(base_path: str, file_type: str = '.fif') -> List[List[mne.io.Raw]]:
    """
    Loads raw data from a directory.

    Args:
        base_path: Path to the directory containing the raw data.
        file_type: File type of the raw data.

    Returns:
        List of lists of raw data.
    """
    all_raw_data = list()
    database_paths = [f.path for f in os.scandir(base_path) if f.is_dir()]
    
    for database_path in database_paths:
        database_raw_data = list()
        for dir_path, _, files in os.walk(database_path):
            for file in files:
                if file.lower().endswith(file_type):
                    raw_data = mne.io.read_raw_fif(os.path.join(dir_path, file), verbose=True, preload=True)
                    database_raw_data.append(raw_data)
        all_raw_data.append(database_raw_data)
    
    return all_raw_data

def preprocess_and_save_data(
    all_raw_data: List[List[mne.io.Raw]],
    config: dict,
    output_dir: str,
    metadata_fn: str):
    """
    Preprocesses raw data and saves it to a file.

    Args:
        all_raw_data: List of lists of raw data.
        config: Configuration dictionary.
        output_dir: Directory to save the preprocessed data.
        metadata_fn: File name of the metadata file.
    """
    all_data = [raw_data for database in all_raw_data for raw_data in database]
    list_of_data_samples_sizes = list()
    
    for idx, raw_data in enumerate(all_data):
        preprocessed_data = preprocess_data(raw_data, config)
       
        list_of_data_samples_sizes.append(preprocessed_data.shape[0])
    
        torch.save(preprocessed_data, f'{output_dir}/run_{idx}' + DATA_FILE_ENDING)
        
    metadata_path = os.path.join(output_dir, metadata_fn + METADATA_FILE_ENDING)
    with open(metadata_path, 'wb') as filehandle:
        pickle.dump(list_of_data_samples_sizes, filehandle)

    print('list_of_data_samples_sizes:', list_of_data_samples_sizes)


def preprocess_data(data: mne.io.Raw, config: dict) -> pd.DataFrame:
    # Only include MEG data
    if config['data_type'].lower() == 'meg':
        data = data.pick_types(meg=True)
    elif config['data_type'].lower() == 'grad':
        # Gets MEG gradiometers
        data = data.pick_types(meg='grad')
    elif config['data_type'].lower() == 'mag':
        # Gets MEG magnetometers
        data = data.pick_types(meg='mag')
    elif config['data_type'].lower() == 'eeg':
        # data = data.pick_types(eeg=True)
        raise NotImplementedError('EEG data not supported yet.')
    else:
        raise ValueError(f'Invalid data type: {config["data_type"]}')

    # Note: Unlike EEG, MEG always (at least for our datasets) uses a sfreq of 1000.0 Hz
    if data.info['sfreq'] != config['common_sfreq']:
        warnings.warn('Data should have sfreq of 1000.0 Hz; Please check data')
        data = data.resample(sfreq=config['common_sfreq'])

    # High-pass Filter code 
    if config['use_high_pass_filter']:
        # Don't change from 0.1 Hz 
        # (see https://mne.tools/0.15/auto_tutorials/plot_background_filtering.html#high-pass-problems)
        data = data.filter(l_freq=config['high_pass_cutoff'], h_freq=None, fir_design='firwin')

    # ICA code 
    if config['use_ica']:
        ica_config = config['ica']
        ica = mne.preprocessing.ICA(n_components=ica_config['n_components'], \
            random_state=config['seed'], max_iter=ica_config['max_iter'])
        ica.fit(data)
        data = ica.apply(data)
    
    data = data.to_data_frame()

    # Standardization vs. Normalization
    if config['use_standardization']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = torch.from_numpy(data.values).float()
    elif config['use_normalization']:
        data = torch.from_numpy(data.values).float()
        data = min_max_normalize(data)
        
    return data


class BatchTensorDataset(Dataset):
    """
    Dataset that groups a dataset of Tensors into batches.
    """
    def __init__(self, config: dict, metadata_path: str, load_dir: str):
        """
        Args:
            config: Dictionary of configuration parameters
            metadata_path: Path to the metadata file
            load_dir: Directory to load the data from
        """
        
        self.batch_size = config['batch_size']
        self.primary_unit_size = config['primary_unit_size']
        self.calib_unit_size = config['calibration_unit_size']
        self.total_unit_size = self.batch_size * self.primary_unit_size \
            + self.calib_unit_size

        with open(metadata_path, 'rb') as filehandle:
            list_of_data_samples_sizes = pickle.load(filehandle)
        
        # Each index corresponds to the index start of a new data batch (different fif file)
        # (Indexed by groups of `total_unit_size`)
        self.list_of_boundaries = [0]
        for samples_size in list_of_data_samples_sizes:
            self.list_of_boundaries.append((samples_size // self.total_unit_size) \
                              + self.list_of_boundaries[-1])

        # Used to cache a dataset
        self.last_idx_run_seen = None
        self.last_dataset_seen = None
        
        self.load_dir = load_dir

        print('List of dataset boundaries:', self.list_of_boundaries)

    def __len__(self):
        return self.list_of_boundaries[-1]

    def __getitem__(self, idx: int) -> Tensor:
        idx_run = None
        value_run = None
        # Ugly; change this; 
        # Perhaps try iterable-style dataset
        for j, value in enumerate(self.list_of_boundaries):
            if self.list_of_boundaries[j+1] > idx >= value:
                value_run = value
                idx_run = j
                break
        
        # Checking if the last dataset is in the cache
        if idx_run == self.last_idx_run_seen:
            data_run = self.last_dataset_seen
        else:
            data_run = torch.load(os.path.join(self.load_dir, f'run_{idx_run}{DATA_FILE_ENDING}'))

        idx_actual = idx - value_run

        start_idx = idx_actual * self.total_unit_size
        end_idx = start_idx + self.total_unit_size
        data = data_run[start_idx:end_idx]

        self.last_idx_run_seen = idx_run
        self.last_dataset_seen = data_run
        
        return {
            'calibration_input': data[:self.calib_unit_size].unsqueeze(0),
            'primary_input': data[self.calib_unit_size:].view(
                self.batch_size, self.primary_unit_size, data.shape[1])
        }


def prepare_dataloaders(config):
    base_dir = os.path.dirname(__file__)

    train_data_path = os.path.join(base_dir, config['train_val_preprocessed'])
    train_metadata_path = os.path.join(train_data_path, config['train_val_info'] + METADATA_FILE_ENDING)
    train_val_dataset = BatchTensorDataset(config, train_metadata_path, train_data_path)

    test_data_path = os.path.join(base_dir, config['test_preprocessed'])
    test_metadata_path = os.path.join(test_data_path, config['test_info'] + METADATA_FILE_ENDING)
    test_dataset = BatchTensorDataset(config, test_metadata_path, test_data_path)

    n_val_samples = int(config['val_split'] * len(train_val_dataset))
    n_train_samples = len(train_val_dataset) - n_val_samples

    train_dataset, val_dataset = random_split(
        train_val_dataset, [n_train_samples, n_val_samples])

    # Removes the first dimension input tensors, Otherwise they are
    # always 1 because the dataloader batch size is always 1
    collate_fn = lambda x: x[0]
    dataloaders = {
        'train': DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn) \
                if len(train_dataset) > 0 else None,
        'val': DataLoader(
            val_dataset, shuffle=True, collate_fn=collate_fn) \
                if len(val_dataset) > 0 else None,
        'test': DataLoader(
            test_dataset,  shuffle=True, collate_fn=collate_fn) \
                if len(test_dataset) > 0 else None
    }

    return dataloaders


# if __name__ == '__main__':
#     base_dir = os.path.dirname(__file__)
#     config_path = os.path.join(base_dir, 'configs/test_config.yaml')
#     config = load_config(config_path)
#     dataloaders = prepare_dataloaders(config)
#     sample = next(iter(dataloaders['train']))
#     print(sample)
#     print('primary shape:', sample['primary_input'].shape)
#     print('calibration shape:', sample['calibration_input'].shape)