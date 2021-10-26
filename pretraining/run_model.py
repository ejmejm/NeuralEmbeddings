import argparse
import os

import numpy as np
import torch
import wandb

from config_handling import load_config, to_wandb_format
from data_loading import prepare_dataloaders
from models import NeuroSignalEncoder
from training.msm import train_with_msm

### Create argparser for command line arguments ###

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/test_config.yaml',
                    help='Path to config file')
parser.add_argument('-t', '--train', dest='train', action='store_true',
                    help='Performs training when set to true.')
parser.add_argument('-n', '--no_train', dest='train', action='store_false',
                    help='Performs training when set to true.')
parser.add_argument('-e', '--test', action='store_true',
                    help='Performs testing when set to true.')
parser.set_defaults(gen_data=False, train=True, test=False)


### Main Functions ###


def train(config: dict):
    """
    Train the model.    
    """
    # Init wandb for logging
    wandb_config = to_wandb_format(config)
    wandb.init(project='neural-embeddings', config=wandb_config)

    # Choose random seed if not provided
    if config['seed'] is None:
        config['seed'] = np.random.randint(0, 2**31)

    # Set and log seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    wandb.log({'seed': config['seed']})
    print('seed:', config['seed'])

    # Prepare the data
    dataloaders = prepare_dataloaders(config)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Create the model
    model = NeuroSignalEncoder(model_config)
    model = model.to(config['device'])
    wandb.watch(model, log_freq=100)

    # Train the model
    train_with_msm(model, config, train_loader, val_loader)

    # Save the model
    save_path = os.path.join(base_dir, model_config['save_path'])
    torch.save(model.state_dict(), save_path)

    wandb.finish()

def test(config: dict):
    """
    Test the model.
    """
    raise NotImplementedError()


### Main ###


if __name__ == '__main__':
    args = parser.parse_args()

    # Get the path of the config file in relation to this main.py file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, args.config)

    # Load the config file
    config = load_config(config_path)
    model_config = config['model']

    # Train the model
    if args.train:
        train(config)
    # Test the model
    if args.test:
        test(config)