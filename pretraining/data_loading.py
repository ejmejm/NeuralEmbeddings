import os
import pickle
from typing import List

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

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

def load_raw_data(base_path: str, file_type: str = '.fif') -> List[List[str]]:
    """
    Loads raw data from a directory.
    Args:
        base_path: Path to the directory containing the raw data.
        file_type: File type of the raw data.
    Returns:
        List of lists of raw data.
    """
    all_raw_data_paths = list()
    database_paths = [f.path for f in os.scandir(base_path) if f.is_dir()]
    
    for database_path in database_paths:
        database_raw_data_paths = list()
        for dir_path, _, files in os.walk(database_path):
            for file in files:
                if file.lower().endswith(file_type):
                    database_raw_data_paths.append(os.path.join(dir_path, file))
        all_raw_data_paths.append(database_raw_data_paths)
    
    return all_raw_data_paths

def correct_data(file_path: str):
    """
    Split data
    """
    original_file_name = os.path.basename(file_path)
    path = file_path.removesuffix(original_file_name)
    new_path = path + "split_" + original_file_name
    raw_data = mne.io.read_raw_fif(file_path, on_split_missing = "ignore", preload=True, verbose=True)
    raw_data.save(fname=new_path, overwrite=False, split_size = "0.8GB")
    
    os.remove(file_path)
    os.remove(new_path)
   

def correct_all_data(base_path: str, file_type: str = '.fif'):
    """
    Split any data that needs to be split (will otherwise cause error)
    """

    all_data_paths = load_raw_data(base_path, file_type)
    all_data_paths = [raw_data for database in all_data_paths for raw_data in database]
    
    for raw_data_path in all_data_paths:
        try:
            mne.io.read_raw_fif(raw_data_path, on_split_missing = 'raise', preload=True, verbose = True)
        except ValueError:
            print('Correcting and spliting data')
            correct_data(raw_data_path)

        del raw_data_path

def preprocess_and_save_data(
    all_raw_data_paths: List[List[str]],
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
    all_data_paths = [raw_data for database in all_raw_data_paths for raw_data in database]
    list_of_data_samples_sizes = list()

    # Sample some of the data for fitting the preprocessing models
    n_fit_samples = min(config['preprocess_fit_samples'], len(all_data_paths))
    print(f'Using {n_fit_samples} samples for fitting preprocessing models')
    fit_data_paths = np.random.choice(all_data_paths, n_fit_samples, False)

    print('Loading data for fitting preprocessing models...')
    fit_samples = []
    for raw_data_path in tqdm(fit_data_paths):
        # load raw data; skip any bad data (should)
        try:
            raw_data = mne.io.read_raw_fif(raw_data_path, on_split_missing='raise',
                preload=True, verbose=False)
            fit_samples.append(raw_data)
        except ValueError:
            warnings.warn('Should not have skipped; check data again')
            continue
    
    # Learn the needed preprocessing models
    print('Fitting preprocessing models...')
    learn_preprocessors(fit_samples, config)

    print('Preprocessing...')
    for idx, raw_data_path in enumerate(tqdm(all_data_paths)):
        # load raw data; skip any bad data (should)
        try:
            raw_data = mne.io.read_raw_fif(raw_data_path, on_split_missing='raise',
                preload=True, verbose=False)
        except ValueError:
            warnings.warn('Should not have skipped; check data again')
            continue

        preprocessed_data = preprocess_data(raw_data, config)
        list_of_data_samples_sizes.append(preprocessed_data.shape[0])

        torch.save(preprocessed_data, f'{output_dir}/run_{idx}' + DATA_FILE_ENDING)

    metadata_path = os.path.join(output_dir, metadata_fn + METADATA_FILE_ENDING)
    with open(metadata_path, 'wb') as filehandle:
        pickle.dump(list_of_data_samples_sizes, filehandle)

    print('list_of_data_samples_sizes:', list_of_data_samples_sizes)

def select_target_data(data: mne.io.Raw, config: dict) -> mne.io.Raw:
    """
    Selects the target type of data given the config.

    Args:
        data: Raw data.
        config: Config dictionary.

    Returns:
        Raw data of the target type.
    """
    if config['data_type'].lower() == 'meg':
        # Only include MEG data
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

    return data

def learn_preprocessors(data: List[mne.io.Raw], config: dict):
    """
    The function creates preprocessors specified in the config
    and fits them to a random sample of data.
    They are then saved for later use.

    Args:
        data: Raw data.
        config: Configuration dictionary.
        n_batches: Number of batches to use for fitting.
    """
    if len(data) == 0:
        raise ValueError('No data to fit preprocessors to.')

    for i in range(len(data)):
        # Get the specific channels of interest
        data[i] = select_target_data(data[i], config)
        # Resample and run the data through a high-pass filter
        data[i] = apply_model_free_preprocessing(data[i], config)

    if len(data) == 1:
        data = data[0]
    else:
        full_data = mne.concatenate_raws(data)
    preprocessing_models = {'ica': None, 'scaler': None}

    # ICA
    if config['use_ica']:
        ica_config = config['ica']
        ica = mne.preprocessing.ICA(n_components=ica_config['n_components'], \
            random_state=config['seed'], max_iter=ica_config['max_iter'])
        ica.fit(full_data)
        full_data = ica.apply(full_data)
        preprocessing_models['ica'] = ica
    
    full_data = full_data.to_data_frame()
    full_data.drop(['time'], axis=1, inplace=True)
    full_data = full_data.values

    # Standardization
    if config['use_standardization']:
        scaler = StandardScaler()
        # Reshape to -1 so all channel samples are scaled together,
        # otherwise each channel would be scaled independently
        flat_data = scaler.fit_transform(full_data.reshape(-1, 1))
        full_data = flat_data.reshape(full_data.shape)
        preprocessing_models['scaler'] = scaler
        
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, config['preprocessed_model_path'])
    with open(model_path, 'wb') as f:
        pickle.dump(preprocessing_models, f)

    return preprocessing_models

def apply_model_free_preprocessing(data: mne.io.Raw, config: dict) -> mne.io.Raw:
    """
    Applys preprocessing steps that don't require learning
    a model of the data.

    Args:
        data: Raw data of selected channels.
        config: Config dictionary.

    Returns:
        Partially preprocessed data.
    """
    # Note: Unlike EEG, MEG always (at least for our datasets) uses a sfreq of 1000.0 Hz
    if data.info['sfreq'] != config['common_sfreq']:
        warnings.warn('Data should have sfreq of 1000.0 Hz; Please check data')
        data = data.resample(sfreq=config['common_sfreq'])

    # High-pass Filter code 
    if config['use_high_pass_filter']:
        # Don't change from 0.1 Hz 
        # (see https://mne.tools/0.15/auto_tutorials/plot_background_filtering.html#high-pass-problems)
        data = data.filter(l_freq=config['high_pass_cut_off'], h_freq=None, fir_design='firwin')

    return data

def apply_model_based_preprocessing(data: mne.io.Raw, config: dict) -> mne.io.Raw:
    """
    Applys preprocessing steps that require a learned model of the data.

    Args:
        data: Data after pass through model free preprocessing.
        config: Config dictionary.

    Returns:
        Fully preprocessed data.
    """
    # Load preprocessing models
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, config['preprocessed_model_path'])
    with open(model_path, 'rb') as f:
        preprocessing_models = pickle.load(f)

    # ICA code 
    if config['use_ica']:
        ica = preprocessing_models['ica']
        data = ica.apply(data)
    
    data = data.to_data_frame()
    data.drop(['time'], axis=1, inplace=True)
    data = data.values

    # Standardization vs. normalization
    if config['use_standardization']:
        scaler = preprocessing_models['scaler']
        flat_data = scaler.transform(data.reshape(-1, 1))
        data = flat_data.reshape(data.shape)
        data = torch.from_numpy(data).float()
    elif config['use_normalization']:
        # Taking it out because no time to make a class and saving it for downstream tasks
        # Also no reason to believe it would perform better than the StandardScaler
        # Could reimplement later
        # data = torch.from_numpy(data.values).float()
        # data = min_max_normalize(data)
        raise NotImplementedError('Normalization not implemented yet.')

    return data


def preprocess_data(data: mne.io.Raw, config: dict) -> pd.DataFrame:
    # Get the specific channels of interest
    data = select_target_data(data, config)
    # Resample and run the data through a high-pass filter
    data = apply_model_free_preprocessing(data, config)
    # ICA and normalization/scaling
    data = apply_model_based_preprocessing(data, config)
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
