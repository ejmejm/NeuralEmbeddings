import numpy as np
from models import NeuroSignalEncoder
from training.cpc import train_with_cpc
import torch
from torch import nn, Tensor,gt
import torch.nn.functional as F
import argparse
import os
from data_loading import prepare_downsteam_dataloaders
from data_loading import prepare_dataloaders
from config_handling import prepare_config
import wandb
from config_handling import prepare_config, to_wandb_format

class custom_model(nn.Module):
    def __init__(self,config: dict):
        super(custom_model, self).__init__()
        self.config = config
        self.model_config = config['model']
        self.model_neural = NeuroSignalEncoder(self.model_config)
        if self.config['train_method'].lower() == 'cpc':
            cpc_config = config['cpc_params']
            self.model_neural.add_lstm_head(cpc_config['embedding_dim'])
        self.model_neural.load_state_dict(torch.load(self.model_config['save_path']))
        self.linear_1 = nn.Linear(19097, 1024)
        self.linear_2 = nn.Linear(1024, 512)
        self.linear_3 = nn.Linear(512, 128)
        self.linear_4 = nn.Linear(128, 64)
        self.linear_5 = nn.Linear(64, 10)
        self.relu= nn.ReLU()

    def forward(self, primary_input,calibration_input):
        primary_input = primary_input.to(self.config['device'])
        calibration_input = calibration_input.to(self.config['device'])
        output_dict = self.model_neural(primary_input, calibration_input=calibration_input)
        primary_embed_mask = output_dict['primary_embeddings_mask']
        # Gives a 1 or 0 for each item in the output sequence, 1 is for primary sequence
        primary_embed_mask = primary_embed_mask.bool().unsqueeze(-1)
        primary_embeds= output_dict['lstm_embeddings']
        primary_embeds= primary_embeds.squeeze(-1)
        selected_out_embeds = primary_embeds.masked_select(primary_embed_mask)
        target_idx = np.ceil(self.config['n_stimulus_samples'] /self.config['primary_unit_size']*len(selected_out_embeds)).astype(int)
        primary_embeds = primary_embeds.reshape(1,primary_embeds.shape[1]*primary_embeds.shape[2])
        target_embeds = primary_embeds[:,target_idx:]
        print("target_embeds",target_embeds)
        #target_embeds = primary_embeds[:,target_idx]
        linear_output = self.linear_1(target_embeds)
        # linear_output= self.relu(linear_output)

        linear_output = self.linear_2(linear_output)
        # linear_output= self.relu(linear_output)

        linear_output = self.linear_3(linear_output)
        #linear_output= self.relu(linear_output)

        linear_output = self.linear_4(linear_output)
        #linear_output= self.relu(linear_output)

        linear_output = self.linear_5(linear_output)
        #linear_output= self.relu(linear_output)

        return linear_output
    

   
