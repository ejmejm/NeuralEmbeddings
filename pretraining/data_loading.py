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
    def __init__(self, data: Tensor, config: dict):
        """
        Args:
            data: Tensor of shape (n_samples, n_channels)
            config: Dictionary of configuration parameters
        """
        self.data = data
        self.batch_size = config['batch_size']
        self.primary_unit_size = config['primary_unit_size']
        self.calib_unit_size = config['calibration_unit_size']
        self.total_unit_size = self.batch_size * self.primary_unit_size \
            + self.calib_unit_size

    def __len__(self):
        return self.data.shape[0] // self.total_unit_size

    def __getitem__(self, idx: int) -> Tensor:
        start_idx = idx * self.total_unit_size
        end_idx = start_idx + self.total_unit_size
        # TODO: Stop grabbing just the first channel once full model is ready
        data = self.data[start_idx:end_idx]
        return {
            'calibration_input': data[:self.calib_unit_size].unsqueeze(0),
            'primary_input': data[self.calib_unit_size:].view(
                self.batch_size, self.primary_unit_size, data.shape[1])
        }


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

def prepare_dataloaders(config: dict) \
        -> dict[str, DataLoader]:
    """
    Prepares the dataloaders for the training, validation, and testing sets.

    Args:
        config: The configuration for the data.
    
    Returns:
        A dictionary of dataloaders.
    """
    # Load and preprocess data
    raw_data = load_data()
    filtered_data = preprocess_data(raw_data, config['data_type'])

    # Create a dataset with the data
    dataset = BatchTensorDataset(
        torch.from_numpy(filtered_data.values).float(), config)

    # Split the dataset into train and validation
    n_val_samples = int(config['val_split'] * len(dataset))
    n_test_samples = int(config['test_split'] * len(dataset))
    n_train_samples = len(dataset) - n_val_samples - n_test_samples

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train_samples, n_val_samples, n_test_samples])

    # Removes the first dimension input tensors, Otherwise they are
    # always 1 because the dataloader batch size is always 1
    collate_fn = lambda x: x[0]
    # Create a dataloader for each dataset
    dataloaders = {
        'train': DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn),
        'val': DataLoader(
            val_dataset, shuffle=True, collate_fn=collate_fn),
        'test': DataLoader(
            test_dataset, shuffle=True, collate_fn=collate_fn),
    }
    
    return dataloaders

if __name__ == '__main__':
    # Testing to make sure the data is loaded correctly
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, 'configs/test_config.yaml')
    config = load_config(config_path)
    dataloaders = prepare_dataloaders(config)
    sample = next(iter(dataloaders['train']))
    print(sample)
    print('primary shape:', sample['primary_input'].shape)
    print('calibration shape:', sample['calibration_input'].shape)