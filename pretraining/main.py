import argparse
import os

import torch

from data_loading import prepare_dataloaders, load_config
from models import Wav2Vec

### Create argparser for command line arguments ###

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/test_config.yaml')

### Constants ###

MODEL_DIR = '../models'

### Main ###

if __name__ == '__main__':
    args = parser.parse_args()

    # Get the path of the config file in relation to this main.py file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, args.config)

    # Load the config file
    config = load_config(config_path)
    data_config = config['data']
    train_config = config['training']
    model_config = config['model']

    # Prepare the data
    dataloaders = prepare_dataloaders(data_config, train_config)
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

    # # Train the model
    # train_with_msm(model, train_loader, val_loader, train_config)

    # # Save the model
    # save_path = os.path.join(base_dir, model_config['save_path'])
    # torch.save(model.state_dict(), save_path)

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


    


# if __name__ == '__main__':
#     raw_data = load_data()
#     meg_data = prepare_meg_data(raw_data)
#     meg_full_tensor = torch.from_numpy(meg_data.values)

#     input_seq = meg_full_tensor[:1024, 0]
#     input_seq = input_seq.unsqueeze(0).unsqueeze(0).float()
#     print('input_seq shape:', input_seq.shape)

#     print('Model init')
#     model = Wav2Vec(
#         input_dim = 1024,
#         embedding_dim = 16,
#         embed_reduc_factor = 16,
#         conv_width = 32,
#         n_layers = 4,
#         dropout = 0.5,
#         n_head = 1,
#         include_conv = False,
#         include_transformer = True)

#     print('Model forward')
#     output = model(input_seq)
#     print('output shape:', output.shape)