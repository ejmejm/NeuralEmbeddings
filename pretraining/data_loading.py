import os

import mne
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


def load_data():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                        'sample_audvis_filt-0-40_raw.fif')
    raw_data = mne.io.read_raw_fif(sample_data_raw_file)
    raw_data.load_data()
    return raw_data

def preprocess_data(raw_data, data_type='MEG'):
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

class MEGDataset(Dataset):
    def __init__(self, batch_size):
        raw_data = load_data()
        meg_data = prepare_meg_data(raw_data)
        self.meg_full_tensor = torch.from_numpy(meg_data.values)
        self.batch_size = batch_size

    def __len__(self):
        return self.meg_full_tensor.shape[0] // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        return self.meg_full_tensor[start_idx:end_idx]

def prepare_eeg_data(raw_data):
    return preprocess_data(raw_data, data_type='EEG')

def prepare_meg_data(raw_data):
    return preprocess_data(raw_data, data_type='MEG')