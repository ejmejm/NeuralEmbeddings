import argparse
import os

import numpy as np
import torch

from data_loading import prepare_dataloaders, load_config
from models import Wav2Vec
from training import train_with_msm

### Create argparser for command line arguments ###

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/test_config.yaml')

### Main ###

if __name__ == '__main__':
    args = parser.parse_args()

    # Get the path of the config file in relation to this main.py file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, args.config)

    # Load the config file
    config = load_config(config_path)
    model_config = config['model']

    # Set seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Prepare the data
    dataloaders = prepare_dataloaders(config)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # Create the model
    conv_config = model_config['conv']
    encoder_config = model_config['single_channel_encoder']
    model = Wav2Vec(
        input_dim = model_config['max_input_seq_len'],
        embedding_dim = model_config['embedding_dim'],
        embed_reduc_factor = conv_config['stride'],
        conv_width = conv_config['filter_size'],
        n_layers = encoder_config['n_layers'],
        dropout = encoder_config['dropout'],
        n_head = encoder_config['n_head'],
        include_conv = conv_config['enabled'],
        include_transformer = encoder_config['enabled'])
    model = model.to(config['device'])

    # Train the model
    train_with_msm(model, config, train_loader, val_loader)

    # Save the model
    save_path = os.path.join(base_dir, model_config['save_path'])
    torch.save(model.state_dict(), save_path)

    # # Test the model
    # test_with_msm(model, test_loader, train_config)

    # # Testing stuff

    # print(next(iter(train_loader)).shape)
    # print(next(iter(val_loader)).shape)

    # print(len(train_loader))
    # print(len(val_loader))

    # test_input = next(iter(train_loader))[:, :, 0:1]
    # # test_input = next(iter(train_loader)).transpose(0, 2)

    # print('input shape:', test_input.shape)
    # output = model(test_input)
    # print('output shape:', output.shape)