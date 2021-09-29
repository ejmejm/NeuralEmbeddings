import torch

from data_loading import load_data, prepare_meg_data
from models import Wav2Vec

if __name__ == '__main__':
    raw_data = load_data()
    meg_data = prepare_meg_data(raw_data)
    meg_full_tensor = torch.from_numpy(meg_data.values)

    input_seq = meg_full_tensor[:1024, 0]
    print('input_seq shape:', input_seq.shape)

    print('Model init')
    model = Wav2Vec(
        input_dim = 1024,
        embed_reduc_factor = 2,
        conv_width = 3,
        n_layers = 6,
        dropout = 0.5,
        include_conv = True,
        include_transformer = True)

    print('Model forward')
    output = model(input_seq)
    print('output shape:', output.shape)