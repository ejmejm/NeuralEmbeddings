import os

import mne
from mne.io.fiff.raw import Raw
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import yaml


class BatchTensorDataset(Dataset):
    """
    Dataset that groups a dataset of Tensors into batches.
    """
    def __init__(self, data: Tensor, unit_size: int):
        self.data = data
        self.unit_size = unit_size

    def __len__(self):
        return self.data.shape[0] // self.unit_size

    def __getitem__(self, idx: int) -> Tensor:
        start_idx = idx * self.unit_size
        end_idx = start_idx + self.unit_size
        return self.data[start_idx:end_idx]


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.load(f)

def load_data():
    """
    Loads a raw sample data for testing.
    """
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                        'sample_audvis_filt-0-40_raw.fif')
    raw_data = mne.io.read_raw_fif(sample_data_raw_file)
    raw_data.load_data()
    return raw_data

def preprocess_data(raw_data: Raw, data_type: str='MEG') -> pd.DataFrame:
    """
    Takes the target portion of the raw data and preprocesses it.

    Args:
        raw_data: The raw data to preprocess.
        data_type: The type of data to preprocess, ['MEG', 'EEG'].

    Returns:
        The preprocessed data.
    """
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw_data)
    ica.exclude = [1, 2]  # details on how we picked these are omitted here

    data = ica.apply(raw_data)
    df = data.to_data_frame()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    target_columns = [c for c in df.columns if data_type.lower() in c.lower()]
    target_data = df[target_columns]
    
    for i, column in enumerate(target_data.columns):
        target_data[column] = scaled_data[:, i]

    return target_data

def prepare_eeg_data(raw_data: Raw) -> pd.DataFrame:
    return preprocess_data(raw_data, data_type='EEG')

def prepare_meg_data(raw_data: Raw) -> pd.DataFrame:
    return preprocess_data(raw_data, data_type='MEG')

def prepare_dataloaders(data_config: dict, train_config: dict) \
        -> dict[str, DataLoader]:
    """
    Prepares the dataloaders for the training, validation, and testing sets.

    Args:
        data_config: The configuration for the data.
        train_config: The configuration for the training.
    
    Returns:
        A dictionary of dataloaders.
    """
    # Load and preprocess data
    raw_data = load_data()
    filtered_data = preprocess_data(raw_data, data_config['data_type'])

    # Create a dataset with the data
    dataset = BatchTensorDataset(
        torch.from_numpy(filtered_data.values).float(),
        unit_size = data_config['unit_size'])

    # Split the dataset into train and validation
    n_val_samples = int(train_config['val_split'] * len(dataset))
    n_test_samples = int(train_config['test_split'] * len(dataset))
    n_train_samples = len(dataset) - n_val_samples - n_test_samples

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train_samples, n_val_samples, n_test_samples])

    # Create a dataloader for each dataset
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            train_config['train_batch_size'],
            shuffle = True),
        'val': DataLoader(
            val_dataset,
            train_config['val_batch_size'],
            shuffle = True),
        'test': DataLoader(
            test_dataset,
            train_config['test_batch_size'],
            shuffle = True)
    }
    
    return dataloaders
