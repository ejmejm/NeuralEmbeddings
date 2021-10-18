import os
import mne
import torch
import yaml
from mne.decoding import Scaler
from sklearn.preprocessing import StandardScaler
import warnings
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor
import pickle

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.load(f)

# From DN3 utils
def min_max_normalize(x: torch.Tensor, low=-1, high=1):
    xmin = x.min()
    xmax = x.max()
    if xmax - xmin == 0:
        x = 0
        return x
    
    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    return (high - low) * x
    

def load_raw_data(base_path, file_type = ".fif"):
    all_raw_data = list()
    database_paths = [f.path for f in os.scandir(base_path) if f.is_dir()]
    
    for database_path in database_paths:
        database_raw_data = list()
        for dir_path, _ , files in os.walk(database_path):
            for file in files:
                if file.lower().endswith(file_type):
                    raw_data = mne.io.read_raw_fif(os.path.join(dir_path, file), verbose=True, preload=True)
                    database_raw_data.append(raw_data)
        all_raw_data.append(database_raw_data)
    
    return all_raw_data

def preprocess_and_save_data(all_raw_data, config, save_info, save_data):
    all_data = [raw_data for database in all_raw_data for raw_data in database]
    list_of_data_samples_sizes = list()
    
    for idx, raw_data in enumerate(all_data):
        preprocessed_data = preprocess_data(raw_data, config)
       
        list_of_data_samples_sizes.append(preprocessed_data.shape[0])
    
        torch.save(preprocessed_data, f"./{save_data}/run{idx}")
        
    with open(save_info, 'wb') as filehandle:
        pickle.dump(list_of_data_samples_sizes, filehandle)

    print("list_of_data_samples_sizes", list_of_data_samples_sizes)


def preprocess_data(data, config):
     # Only include MEG data
    data = data.pick_types(meg=True)

    # Note: Unlike EEG, MEG always (at least for our datasets) uses a sfreq of 1000.0 Hz
    if data.info['sfreq'] != config["common_sfreq"]:
        warnings.warn("Data should have sfreq of 1000.0 Hz; Please check data")
        data = data.resample(sfreq = config["common_sfreq"])

    # High-pass Filter code 
    if config["use_high_pass_filter"]:
        # Don't change from 0.1 Hz 
        # (see https://mne.tools/0.15/auto_tutorials/plot_background_filtering.html#high-pass-problems)
        data = data.filter(l_freq=0.1, h_freq=None, fir_design='firwin')

    # ICA code 
    if config["use_ica"]:
        ica_config = config["ica"]
        ica = mne.preprocessing.ICA(n_components=ica_config["n_components"], \
            random_state=ica_config["random_state"], max_iter=ica_config["max_iter"])
        ica.fit(data)
        data = ica.apply(data)
    
    data = data.to_data_frame()

    # Standardization vs. Normalization
    if config["use_standardization"]:
        # requires epoched data
        # scaler = Scaler(scalings='mean')
        # data_df = scaler.fit_transform(data_df)

        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = torch.from_numpy(data.values).float()

    elif config["use_normalization"]:
        data = torch.from_numpy(data.values).float()
        data = min_max_normalize(data)
        
    return data


class BatchTensorDataset(Dataset):
    """
    Dataset that groups a dataset of Tensors into batches.
    """
    def __init__(self, config: dict, dataset_information, load_dir):
        """
        Args:
            config: Dictionary of configuration parameters
        """
        
        self.batch_size = config['batch_size']
        self.primary_unit_size = config['primary_unit_size']
        self.calib_unit_size = config['calibration_unit_size']
        self.total_unit_size = self.batch_size * self.primary_unit_size \
            + self.calib_unit_size

        with open(dataset_information, 'rb') as filehandle:
            list_of_data_samples_sizes = pickle.load(filehandle)
        
        self.list_of_boundaries = [0]
        for samples_size in list_of_data_samples_sizes:
            self.list_of_boundaries.append((samples_size // self.total_unit_size) \
                              + self.list_of_boundaries[-1])

        self.last_idx_run_seen = None
        self.last_dataset_seen = None
        
        self.load_dir = load_dir

        print(self.list_of_boundaries)

    def __len__(self):
        return self.list_of_boundaries[-1]

    def __getitem__(self, idx: int) -> Tensor:
        idx_run = None
        value_run = None
        # Ugly; change this; 
        # Perhaps try iterable-style dataset
        for j, value in enumerate(self.list_of_boundaries):
            if self.list_of_boundaries[j+1]> idx >= value:
                value_run = value
                idx_run = j
                break
        
        if idx_run == self.last_idx_run_seen:
            data_run = self.last_dataset_seen
        else:
            data_run = torch.load(f"./{self.load_dir}/run{idx_run}")

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
    train_val_dataset = BatchTensorDataset(config, config["train_val_info"], config["train_val_preprocessed"])
    # test_dataset = BatchTensorDataset(config, config["test_info"], config["test_preprocessed"])

    n_val_samples = int(config['val_split'] * len(train_val_dataset))
    n_train_samples = len(train_val_dataset) - n_val_samples

    train_dataset, val_dataset = random_split(
        train_val_dataset, [n_train_samples, n_val_samples])

    # Removes the first dimension input tensors, Otherwise they are
    # always 1 because the dataloader batch size is always 1
    collate_fn = lambda x: x[0]
    dataloaders = {
        'train': DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn),
        'val': DataLoader(
            val_dataset, shuffle=True, collate_fn=collate_fn),
        # 'test': DataLoader(
        #     test_dataset,  shuffle=True, collate_fn=collate_fn)
    }

    return dataloaders


def main():
    config = load_config("./configs/test_config.yaml")

    file_type = config["data_file_type"]
    train_dir = config["train_dir"]

    all_data = load_raw_data(train_dir, file_type)
    preprocess_and_save_data(all_data, config, config["train_val_info"], config["train_val_preprocessed"])

    # preprocess test data
    # file_type = config["data_file_type"]
    # test_dir = config["test_dir"]

    # raw_data_all = load_raw_data(test_dir, file_type)
    # preprocess_data(raw_data_all, config, config["test_info"], config["test_preprocessed"])


if __name__ == '__main__':
    main()

    # base_dir = os.path.dirname(__file__)
    # config_path = os.path.join(base_dir, 'configs/test_config.yaml')
    # config = load_config(config_path)
    # dataloaders = prepare_dataloaders(config)
    # sample = next(iter(dataloaders['train']))
    # print(sample)
    # print('primary shape:', sample['primary_input'].shape)
    # print('calibration shape:', sample['calibration_input'].shape)





