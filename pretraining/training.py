import torch

from data_loading import load_data, prepare_meg_data
from models import Wav2Vec

# All MEG data can be loaded as such:
#
# raw_data = load_data()
# meg_data = prepare_meg_data(raw_data)
# meg_full_tensor = torch.from_numpy(meg_data.values)
#
# `meg_full_tensor` is a tensor of shape (n_timepoints, n_channels)
# that gives the full training timeseries in one tensor.
# The dataloader below is a generator that yields batches of
# `batch_size` samples from this tensor.


# def train_msm(model: Wav2Vec, train_loader, n_epochs=5):
#     """
#     Runs a training loop to train a model on masked sequence modeling.
#     """