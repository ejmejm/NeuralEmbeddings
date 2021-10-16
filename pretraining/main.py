import argparse
import os

import numpy as np
import torch
import wandb

from data_loading import prepare_dataloaders, load_config
from models import NeuroSignalEncoder
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

    # Init wandb for logging
    wandb.init(project='neural-embeddings', config=config)

    # Set seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Prepare the data
    dataloaders = prepare_dataloaders(config)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # Create the model
    model = NeuroSignalEncoder(model_config)
    model = model.to(config['device'])
    wandb.watch(model, log_freq=100)

    # out = model(
    #     torch.ones((2, 512, 20), dtype=torch.float32, device=config['device']),

    #     sc_sm_mask = torch.ones((2, 66), dtype=torch.long, device=config['device']),
    #     mc_sm_mask = torch.ones((2, 66), dtype=torch.long, device=config['device']),
    #     calib_sm_mask = torch.ones((1, 32), dtype=torch.long, device=config['device']),

    #     calibration_input = 2 * torch.ones((1, 1024, 20), dtype=torch.float32, device=config['device']),
    #     )
    # print([k for k, v in out.items() if v is not None])
    # print(out['embeddings'].shape)

    # Train the model
    train_with_msm(model, config, train_loader, val_loader)

    # Save the model
    save_path = os.path.join(base_dir, model_config['save_path'])
    torch.save(model.state_dict(), save_path)

    wandb.finish()

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

    ### Some extra code for testing MSM ###

    # print('STARTING EXPERIMENT...')
    # data = next(iter(test_loader))
    # data = data.to(config['device'])
    # mask = torch.zeros((1, 64))
    # mask[0, 1:3] = 1
    # mask = mask.to(config['device'])
    # mask = mask.type(torch.long)

    # output_dict = model(data, sm_mask=mask)

    # full_embeddings = output_dict['embeddings']
    # full_targets = output_dict['targets']

    # test_embeddings = full_embeddings[0, 1:3].detach().cpu().numpy().reshape(-1)
    # test_targets = full_targets[0, 1:3].detach().cpu().numpy().reshape(-1)

    # for a, b in zip(test_embeddings, test_targets):
    #     print(a, b)
        
    # print('MSE loss:', np.mean((test_embeddings - test_targets)**2))