import glob
import os
from pathlib import Path
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

from config_handling import prepare_config


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
            mne.io.read_raw_fif(raw_data_path, on_split_missing='raise', preload=True, verbose=True)
        except ValueError:
            print('Correcting and spliting data')
            correct_data(raw_data_path)

        del raw_data_path

def preprocess_and_save_data(
    all_raw_data_paths: List[List[str]],
    config: dict,
    output_dir: str,
    metadata_fn: str,
    fit_preprocessors: bool = True):
    """
    Preprocesses raw data and saves it to a file.

    Args:
        all_raw_data: List of lists of raw data.
        config: Configuration dictionary.
        output_dir: Directory to save the preprocessed data.
        metadata_fn: File name of the metadata file.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        print(f'Creating output directory: {output_dir}')
        Path(output_dir).mkdir(parents=True)

    # Clear output directory
    print('Clearing output directory to make room for data...')
    for f in glob.glob(os.path.join(output_dir, '*')):
        os.remove(f)

    if all_raw_data_paths is None or len(all_raw_data_paths) == 0:
        warnings.warn(f'No raw data found in "{output_dir}".')

    all_data_paths = [raw_data for database in all_raw_data_paths for raw_data in database]
    list_of_data_samples_sizes = list()

    if fit_preprocessors:
        # Sample some of the data for fitting the preprocessing models
        n_fit_samples = min(config['preprocess_fit_samples'], len(all_data_paths))
        print(f'Using {n_fit_samples} samples for fitting preprocessing models')
        fit_data_paths = np.random.choice(all_data_paths, len(all_data_paths), False)

        print('Loading data for fitting preprocessing models...')
        fit_samples = []
        with tqdm(total=n_fit_samples) as bar:
            sample_idx = 0
            while len(fit_samples) < n_fit_samples and sample_idx < len(fit_data_paths):
                # load raw data; skip any bad data (should)
                try:
                    raw_data = mne.io.read_raw_fif(fit_data_paths[sample_idx],
                        on_split_missing='raise', preload=True, verbose=False)
                    sample_idx += 1

                    if len(raw_data) < config['min_recording_length']:
                        raise ValueError('Sample too small')
                        
                    fit_samples.append(raw_data)
                    bar.update(1)
                except ValueError:
                    warnings.warn('Should not have skipped; check data again')
                    sample_idx += 1
                    continue
        
        # Learn the needed preprocessing models
        print('Fitting preprocessing models...')
        learn_preprocessors(fit_samples, config)

    print('Preprocessing...')
    n_loaded = 0
    for idx, raw_data_path in enumerate(tqdm(all_data_paths)):
        # load raw data; skip any bad data (should)
        try:
            raw_data = mne.io.read_raw_fif(raw_data_path, on_split_missing='warn',
                preload=True, verbose=False)
        except ValueError as e:
            warnings.warn(f'Skipping {raw_data_path} due to error: {e}')
            continue

        if len(raw_data) < config['min_recording_length']:
            warnings.warn(f'Skipping sample {raw_data_path} because it is too short.')
            continue

        preprocessed_data = preprocess_data(raw_data, config)
        list_of_data_samples_sizes.append(preprocessed_data.shape[0])

        n_loaded += 1
        torch.save(preprocessed_data, f'{output_dir}/run_{idx}' + DATA_FILE_ENDING)

    print(f'Successfully preprocessed {n_loaded}/{len(all_data_paths)} runs.')

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

    # if len(data) == 1:
    #     data = data[0]
    # else:
    #     full_data = mne.concatenate_raws(data)
    preprocessing_models = {'ica': None, 'scaler': None}

    # ICA
    if config['use_ica']:
        # Taking this out for now for 2 reasons:
        # 1. It is not clear how to apply ICA to multiple datasets
        # 2. ICA is extremely slow

        # ica_config = config['ica']
        # ica = mne.preprocessing.ICA(n_components=ica_config['n_components'], \
        #     random_state=config['seed'], max_iter=ica_config['max_iter'])
        # ica.fit(full_data)
        # full_data = ica.apply(full_data)
        # preprocessing_models['ica'] = ica
        
        raise NotImplementedError('ICA has been disabled.')

    # Standardization
    if config['use_standardization']:
        # Flatten the data for fitting
        flat_data_batches = []
        for x in data:
            df = x.to_data_frame()
            df.drop(['time'], axis=1, inplace=True)
            vals = df.values.reshape(-1, 1)
            flat_data_batches.append(vals)
        flat_data = np.concatenate(flat_data_batches)

        scaler = StandardScaler()
        # Reshape to -1 so all channel samples are scaled together,
        # otherwise each channel would be scaled independently
        scaler.fit(flat_data)
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
        data = data.filter(l_freq=config['high_pass_cut_off'], h_freq=None, fir_design='firwin',
                           n_jobs=4, filter_length=config['high_pass_filter_length'])

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

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Preprocessing model file not found: {model_path}')

    with open(model_path, 'rb') as f:
        preprocessing_models = pickle.load(f)

    # ICA code 
    if config['use_ica']:
        # Taking this out for now for 2 reasons:
        # 1. It is not clear how to apply ICA to multiple datasets
        # 2. ICA is extremely slow

        # ica = preprocessing_models['ica']
        # data = ica.apply(data)
        raise NotImplementedError('ICA has been disabled.')
    
    # 
    data = data.to_data_frame()
    data.drop(['time'], axis=1, inplace=True)
    data = data.values

    # Standardization vs. normalization
    if config['use_standardization']:
        scaler = preprocessing_models['scaler']
        flat_data = scaler.transform(data.reshape(-1, 1))
        data = flat_data.reshape(data.shape)
    elif config['use_normalization']:
        # Taking it out because no time to make a class and saving it for downstream tasks
        # Also no reason to believe it would perform better than the StandardScaler
        # Could reimplement later
        # data = torch.from_numpy(data.values).float()
        # data = min_max_normalize(data)
        raise NotImplementedError('Normalization not implemented yet.')

    data = torch.from_numpy(data).float()
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

def get_first_element(x):
    return x[0]

def prepare_dataloaders(config):
    base_dir = os.path.dirname(__file__)

    train_data_path = os.path.join(base_dir, config['train_val_preprocessed'])
    train_metadata_path = os.path.join(train_data_path, config['train_val_info'] + METADATA_FILE_ENDING)

    # Check if the data and metadata paths/files exist, and load if it does
    if os.path.exists(train_data_path) and os.path.exists(train_metadata_path):
        train_val_dataset = BatchTensorDataset(config, train_metadata_path, train_data_path)
        n_val_samples = int(config['val_split'] * len(train_val_dataset))
        n_train_samples = len(train_val_dataset) - n_val_samples
        train_dataset, val_dataset = random_split(
            train_val_dataset, [n_train_samples, n_val_samples])
    else:
        warnings.warn('The training data or training metadata file does not exist. ' +
                      'Skipping loading of the training data.')
        train_dataset = None
        val_dataset = None

    test_data_path = os.path.join(base_dir, config['test_preprocessed'])
    test_metadata_path = os.path.join(test_data_path, config['test_info'] + METADATA_FILE_ENDING)

    # Check if the data and metadata paths/files exist, and load if it does
    if os.path.exists(train_data_path) and os.path.exists(train_metadata_path):
        test_dataset = BatchTensorDataset(config, test_metadata_path, test_data_path)
    else:
        warnings.warn('The testing data or testing metadata file does not exist. ' +
                      'Skipping loading of the training data.')
        test_dataset = None
        
    dataloaders = {
        'train': DataLoader(
            train_dataset, shuffle=True, collate_fn=get_first_element, num_workers=2) \
                if train_dataset is not None and len(train_dataset) > 0 else None,
        'val': DataLoader(
            val_dataset, shuffle=True, collate_fn=get_first_element, num_workers=2) \
                if val_dataset is not None and len(val_dataset) > 0 else None,
        'test': DataLoader(
            test_dataset,  shuffle=True, collate_fn=get_first_element, num_workers=2) \
                if test_dataset is not None and len(test_dataset) > 0 else None
    }

    return dataloaders







### EPOCHED DATA PREPROCESSING ###

def apply_model_based_preprocessing_epochs(data: mne.io.Raw, config: dict, events: np.ndarray) -> tuple[Tensor, np.ndarray]:
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

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Preprocessing model file not found: {model_path}')

    with open(model_path, 'rb') as f:
        preprocessing_models = pickle.load(f)

    # ICA code 
    if config['use_ica']:
        # Taking this out for now for 2 reasons:
        # 1. It is not clear how to apply ICA to multiple datasets
        # 2. ICA is extremely slow

        # ica = preprocessing_models['ica']
        # data = ica.apply(data)
        raise NotImplementedError('ICA has been disabled.')
    
    # Convert into epoches data
    epochs_data, labels = get_epoched_data(data, config, events)
    epochs_data_tensor = torch.zeros(epochs_data.shape)

    # Standardize for each epoch
    for epoch_data_num in range(epochs_data.shape[0]):
        epoch_data = epochs_data[epoch_data_num, :, :]

        # Standardization vs. normalization
        if config['use_standardization']:
            scaler = preprocessing_models['scaler']
            flat_data = scaler.transform(epoch_data.reshape(-1, 1))
            epoch_data = flat_data.reshape(epoch_data.shape)

        elif config['use_normalization']:
            # Taking it out because no time to make a class and saving it for downstream tasks
            # Also no reason to believe it would perform better than the StandardScaler
            # Could reimplement later
            # data = torch.from_numpy(data.values).float()
            # data = min_max_normalize(data)
            raise NotImplementedError('Normalization not implemented yet.')

        epoch_data = torch.from_numpy(epoch_data).float()
        epochs_data_tensor[epoch_data_num, :, :] = epoch_data

    return epochs_data_tensor, labels

def preprocess_epoched_data(data: mne.io.Raw, config: dict, include_events_list: list) -> tuple[Tensor, np.ndarray]:
    # Get events 
    events = get_events(data, config, include_events_list)
    # Get the specific channels of interest
    data = select_target_data(data, config)
    # Resample and run the data through a high-pass filter
    data = apply_model_free_preprocessing(data, config)
    # ICA and normalization/scaling
    data, labels = apply_model_based_preprocessing_epochs(data, config, events)
    return data, labels

def get_events(data: mne.io.Raw, config: dict, include_events_list: list) -> np.ndarray:
    """
    Designed for the following dataset: https://openneuro.org/datasets/ds003352/versions/1.0.0
    
    Needs to be tested for other datasets
    """

    # Grab events with id 1 to 10 (there are other event ids; not sure what they are for)
    events = mne.find_events(data, stim_channel = "STI101", shortest_event = 1, output = 'onset')

    # include only events that are in include_events_list
    events = mne.pick_events(events, include = include_events_list)
    
    return events

def get_epoched_data(data: mne.io.Raw, config: dict, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Designed for the following dataset: https://openneuro.org/datasets/ds003352/versions/1.0.0
    
    Needs to be tested for other datasets
    """

    # Get array of event ids (all of them are between 1 and 10)
    data_events_ids = events[:, 2]

    # Create epoched data from (raw) data and events
    tmin = -config['tmin_samples'] / data.info['sfreq']
    tmax = config['tmax_samples'] / data.info['sfreq']
    epochs = mne.Epochs(data, events, tmin=tmin, tmax=tmax,
                    preload=True, reject=None)

    # get new list of events (since epochs may be dropped if bad)
    selections = epochs.selection
    data_events_ids = data_events_ids[selections]

    # Convert into 3D array of dimension (n_events, n_channels, n_epoch_time_length)
    epochs_data = epochs.get_data()
    
    return epochs_data, data_events_ids

    # If you need to see plot of events over time 
    # event_dict = {1: "Light Pink Spiral", 2: "Dark Pink Spiral", 3: "Light Blue Spiral", 4: "Dark Blue Spiral", \
    #     5: "Light Green Spiral", 6: "Dark Green Spiral", 7: "Light Orange Spiral", 8: "Dark Orange Spiral", 9: "green", \
    #     10: "blue"}
    # event_dict_flipped = dict((value, key) for key, value in event_dict.items())
    # fig = mne.viz.plot_events(events, sfreq=raw_data.info['sfreq'],
    #                       first_samp=raw_data.first_samp, event_id=event_dict_flipped)
    # fig.subplots_adjust(right=0.7)  

def preprocess_and_save_epoched_data(
    all_raw_data_paths: List[List[str]],
    config: dict,
    output_dir: str,
    label_fn: str,
    include_events_list: list):
    """
    Preprocesses raw data and saves it to a file.

    Args:
        all_raw_data: List of lists of raw data.
        config: Configuration dictionary.
        output_dir: Directory to save the preprocessed data.
        metadata_fn: File name of the metadata file.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        print(f'Creating output directory: {output_dir}')
        Path(output_dir).mkdir(parents=True)

    # Clear output directory
    print('Clearing output directory to make room for data...')
    for f in glob.glob(os.path.join(output_dir, '*')):
        os.remove(f)

    if all_raw_data_paths is None or len(all_raw_data_paths) == 0:
        warnings.warn(f'No raw data found in "{output_dir}".')

    all_data_paths = [raw_data for database in all_raw_data_paths for raw_data in database]
    
    # Do not need to fit preprocessors for downstream task; only need to apply it 
    # (see apply_model_based_preprocessing_epochs)

    print('Preprocessing...')

    # Keep track of all labels
    all_labels = list()

    n_loaded = 0
    for idx, raw_data_path in enumerate(tqdm(all_data_paths)):
        # load raw data; skip any bad data (should)
        try:
            raw_data = mne.io.read_raw_fif(raw_data_path, on_split_missing='warn',
                preload=True, verbose=False)
        except ValueError as e:
            warnings.warn(f'Skipping {raw_data_path} due to error: {e}')
            continue

        if len(raw_data) < config['min_recording_length']:
            warnings.warn(f'Skipping sample {raw_data_path} because it is too short.')
            continue
        
        # get preprocessed data and save (by run and epoch)
        preprocessed_data, labels = preprocess_epoched_data(raw_data, config, include_events_list)
        all_labels.append(labels)
        n_loaded += 1
        save_epoched_data(preprocessed_data, output_dir, idx)

    
    all_labels_path = os.path.join(output_dir, f'{label_fn}{METADATA_FILE_ENDING}')
    with open(all_labels_path, 'wb') as filehandle:
        pickle.dump(all_labels, filehandle)

    print(f'Successfully preprocessed {n_loaded}/{len(all_data_paths)} runs.')
    
def save_epoched_data(preprocessed_data: Tensor, output_dir: str, run_num: str):
    """
    Save epoched data 
    
    Each epoch is saved to a separate file 
    """

    for epoch_data_num in range(preprocessed_data.shape[0]):
        # Remember to clone or else you'll save all of preprocessed_data
        epoch_data = preprocessed_data[epoch_data_num, :, :].clone()

        torch.save(epoch_data, f'{output_dir}/run_{run_num}_epoch_{epoch_data_num}' + DATA_FILE_ENDING)

class BatchTensorEpochedDataset(Dataset):
    """
    Dataset that groups a dataset of Tensors into batches.
    """
    def __init__(self, config: dict, label_path: str, load_dir: str):
        """
        Args:
            config: Dictionary of configuration parameters
            label_path: Path to the label files
            load_dir: Path to data files
        """

        self.config = config
        self.load_dir = load_dir
        
        with open(label_path, 'rb') as filehandle:
            self.all_labels = pickle.load(filehandle)

        self.list_of_boundaries = np.cumsum(np.array([0] + [len(labels) for labels in self.all_labels]))
        
    def __len__(self):
        return self.list_of_boundaries[-1]

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        # Convert idx (1D) into run, epoch (2D)
        run_num = None
        run_value = None
        for j, value in enumerate(self.list_of_boundaries):
            if self.list_of_boundaries[j+1] > idx >= value:
                run_num = j
                run_value = value
                break

        epoch_num = idx - run_value

        # (n_channels, sample_size)
        data_run = torch.load(os.path.join(self.load_dir, f'run_{run_num}_epoch_{epoch_num}' + DATA_FILE_ENDING))

        dict_data_run = {
            'calibration_input': data_run[:, :self.config['tmin_samples']],
            'primary_input': data_run[:, self.config['tmin_samples']:]
        }

        return dict_data_run, self.all_labels[run_num][epoch_num]


def prepare_downsteam_dataloaders(config):
    base_dir = os.path.dirname(__file__)

    downstream_data_path = os.path.join(base_dir, config['downstream_preprocessed'])
    label_path = os.path.join(downstream_data_path, config['label_info'] + METADATA_FILE_ENDING)

    # Check if the data and metadata paths/files exist, and load if it does
    if os.path.exists(downstream_data_path) and os.path.exists(label_path):
        downstream_dataset = BatchTensorEpochedDataset(config, label_path, downstream_data_path)
        n_val_samples = int(config['val_split'] * len(downstream_dataset))
        n_test_samples = int(config['test_split'] * len(downstream_dataset))
        n_train_samples = len(downstream_dataset) - n_val_samples - n_test_samples
        train_dataset, val_dataset, test_dataset = random_split(
            downstream_dataset, [n_train_samples, n_val_samples, n_test_samples])
    else:
        warnings.warn('The training data or training metadata file does not exist. ' +
                      'Skipping loading of the training data.')
        train_dataset = None
        val_dataset = None
        test_dataset = None

    dataloaders = {
        'train': DataLoader(
            train_dataset, shuffle=True, collate_fn=get_first_element, num_workers=2) \
                if train_dataset is not None and len(train_dataset) > 0 else None,
        'val': DataLoader(
            val_dataset, shuffle=True, collate_fn=get_first_element, num_workers=2) \
                if val_dataset is not None and len(val_dataset) > 0 else None,
        'test': DataLoader(
            test_dataset,  shuffle=True, collate_fn=get_first_element, num_workers=2) \
                if test_dataset is not None and len(test_dataset) > 0 else None
    }

    return dataloaders



def prepare_downstream_data(config):
    # Load and process the train data
    base_dir = os.path.dirname(__file__)
    file_type = config['data_file_type']
    label_fn = config['label_info']
    downstream_dir = os.path.join(base_dir, config['downstream_dir'])
    output_dir = os.path.join(base_dir, config['downstream_preprocessed'])

    # print('Fixing train split issues...')
    # correct_all_data(train_dir, file_type)

    print('Loading train data...')
    downstream_data = load_raw_data(downstream_dir, file_type)

    # Need to pass in a list of events 
    preprocess_and_save_epoched_data(downstream_data, config, output_dir, label_fn, include_events_list=list(range(1, 11)))


