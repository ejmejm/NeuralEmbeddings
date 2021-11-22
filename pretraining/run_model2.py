import argparse
import os
from typing import Optional
import sklearn
import numpy as np
#from lstm import downstream_train
import torch
import wandb
from torch import nn, Tensor
import torch.nn.functional as F

from config_handling import prepare_config, to_wandb_format
from data_loading import prepare_dataloaders
from data_loading import prepare_downsteam_dataloaders
from models import NeuroSignalEncoder
from training.msm import train_with_msm
from training.cpc import train_with_cpc
from downstream import custom_model
from sklearn.metrics import accuracy_score

### Create argparser for command line arguments ###

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/default_config.yaml',
                    help='Path to config file')
parser.add_argument('-g', '--group', type=str, default=None,
                    help='Add the run to the given group for easier tracking.')
parser.add_argument('-t', '--train', dest='train', action='store_true',
                    help='Performs training when set to true.')
parser.add_argument('-n', '--no_train', action='store_false',
                    help='Performs training when set to true.')
parser.add_argument('-e', '--test', action='store_true',
                    help='Performs testing when set to true.')
parser.add_argument('-d', '--downstream',dest='downstream', action='store_true',
                    help='Performs downstream when set to true.')
parser.set_defaults(gen_data=False, train=False, test=False, downstream = False)


### Main Functions ###


def train(config: dict, group: Optional[str] = None):
    """
    Train the model.
    """
    # Init wandb for logging
    wandb_config = to_wandb_format(config)
    wandb.init(entity=config['wandb_entity'], group=group,
        project=config['wandb_project'], config=wandb_config)

    # Choose random seed if not provided
    if config['seed'] is None:
        config['seed'] = np.random.randint(0, 2**31)
        wandb.config.update({'seed': config['seed']}, True)

    # Set and log seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    print('seed:', config['seed'])

    # Prepare the data
    dataloaders = prepare_dataloaders(config)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Create the model
    model_config = config['model']
    model = NeuroSignalEncoder(model_config)
    # Add an LSTM head if CPC is being used
    if config['train_method'].lower() == 'cpc':
        cpc_config = config['cpc_params']
        model.add_lstm_head(cpc_config['embedding_dim'])
    model = model.to(config['device'])
    wandb.watch(model, log_freq=100)

    # Train the model
    if config['train_method'].lower() == 'msm':
        train_with_msm(model, config, train_loader, val_loader)
    elif config['train_method'].lower() == 'cpc':
        train_with_cpc(model, config, train_loader, val_loader)
       
    else:
        raise ValueError('Train method "{}" not recognized.'\
            .format(config['train_method']))

    # Save the model
    if model_config['save_path'] is not None:
        base_dir = os.path.dirname(__file__)
        save_path = os.path.join(base_dir, model_config['save_path'])
        torch.save(model.state_dict(), save_path)

    wandb.finish()

def test(config: dict):
    """
    Test the model.
    """
    raise NotImplementedError()


def downstream(config: dict, group: Optional[str] = None):
   
    # Init wandb for logging
    wandb_config = to_wandb_format(config)
    wandb.init(entity=config['wandb_entity'], group=group,
        project=config['wandb_project'], config=wandb_config)

    # Choose random seed if not provided
    if config['seed'] is None:
        config['seed'] = np.random.randint(0, 2**31)
        wandb.config.update({'seed': config['seed']}, True)

    # Set and log seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    print('seed:', config['seed'])

    #Prepare the data
    dataloaders = prepare_downsteam_dataloaders(config)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    epochs =2
    lr= 0.01
    
   
    all_labels =[]
    all_preds =[]
    for data,label in train_loader:
        all_labels.append(label)

    model = custom_model(config)
    model = model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
   
    for i in range(epochs):
        all_preds =[]
        model.train()
        for data,label in train_loader:
            primary_embeds = data['primary_input']
            primary_embeds= torch.transpose(primary_embeds,0, 1) 
            primary_embeds = primary_embeds.unsqueeze(0) 
            calibration_embeds = data['calibration_input']
            calibration_embeds= torch.transpose(calibration_embeds,0, 1)
            calibration_embeds = calibration_embeds.unsqueeze(0) 
            optimizer.zero_grad()
            pred = model(primary_embeds, calibration_embeds)
            pred=F.log_softmax(pred, dim=1)
            print("pred", pred)
            label_pass = torch.zeros(1,10)
            label_pass[:,label-1]= 1
            label_pass=label_pass.to(device="cuda")
            loss = F.cross_entropy(pred,label_pass)
            print("loss=",loss.item())
            pred=pred.argmax(dim=1)
            all_preds.append(pred.item()+1)
            loss.backward()
            optimizer.step()
        

        print("all preds", all_preds)
        print("all_labels",all_labels)
        print("accuracy:", accuracy_score(all_labels,all_preds))
        # all_preds= torch.FloatTensor(all_preds)
        # all_labels=torch.FloatTensor(all_labels)
        
        
    # Save the model
    if config['downstream_save_path'] is not None:
        base_dir = os.path.dirname(__file__)
        save_path = os.path.join(base_dir, config['downstream_save_path'])
        torch.save(model.state_dict(), save_path)

    wandb.finish()


### Main ###

if __name__ == '__main__':
    args = parser.parse_args()
    # Load the config
    config = prepare_config(args.config)

    # Train the model
    if args.train:
        train(config, args.group)
    # Test the model
    if args.test:
        test(config) 

    if args.downstream:
        downstream(config,args.group)
        

