import argparse
import os

from config_handling import load_config
from data_loading import load_raw_data, preprocess_and_save_data, correct_all_data


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/test_config.yaml',
                    help='Path to config file')


if __name__ == '__main__':
    args = parser.parse_args()

    # Get the path of the config file in relation to this main.py file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, args.config)

    # Load the config file
    config = load_config(config_path)
    
    # Load and process the train data
    file_type = config['data_file_type']
    metadata_fn = config['train_val_info']
    train_dir = os.path.join(base_dir, config['train_dir'])
    output_dir = os.path.join(base_dir, config['train_val_preprocessed'])

    print('Fixing train split issues...')
    correct_all_data(train_dir, file_type)

    print('Loading train data...')
    train_data = load_raw_data(train_dir, file_type)
    preprocess_and_save_data(train_data, config, output_dir, metadata_fn)

    # Load and process the test data
    metadata_fn = config['test_info']
    test_dir = os.path.join(base_dir, config['test_dir'])
    output_dir = os.path.join(base_dir, config['test_preprocessed'])

    print('Fixing test split issues...')
    correct_all_data(test_dir, file_type)

    print('Loading test data...')
    test_data = load_raw_data(test_dir, file_type)
    preprocess_and_save_data(test_data, config, output_dir, metadata_fn)