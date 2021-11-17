import argparse
import os

from config_handling import prepare_config
from data_loading import load_raw_data, preprocess_and_save_data, correct_all_data, preprocess_and_save_epoched_data


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/test_config.yaml',
                    help='Path to config file')

def prepare_data(config):
    # Load and process the train data
    base_dir = os.path.dirname(__file__)
    file_type = config['data_file_type']
    metadata_fn = config['train_val_info']
    train_dir = os.path.join(base_dir, config['train_dir'])
    output_dir = os.path.join(base_dir, config['train_val_preprocessed'])

    # print('Fixing train split issues...')
    # correct_all_data(train_dir, file_type)

    print('Loading train data...')
    train_data = load_raw_data(train_dir, file_type)
    preprocess_and_save_data(train_data, config, output_dir, metadata_fn)

    # Load and process the test data
    metadata_fn = config['test_info']
    test_dir = os.path.join(base_dir, config['test_dir'])
    output_dir = os.path.join(base_dir, config['test_preprocessed'])

    # print('Fixing test split issues...')
    # correct_all_data(test_dir, file_type)

    print('Loading test data...')
    test_data = load_raw_data(test_dir, file_type)
    preprocess_and_save_data(test_data, config,
        output_dir, metadata_fn, fit_preprocessors=False)


def prepare_downstream_data(config):
    # Load and process the train data
    base_dir = os.path.dirname(__file__)
    file_type = config['data_file_type']
    label_fn = config['label_info']
    downstream_dir = os.path.join(base_dir, config['downstream_dir'])
    output_dir = os.path.join(base_dir, config['downstream_preprocessed'])

    # print('Fixing train split issues...')
    # correct_all_data(train_dir, file_type)

    print('Loading train data...')
    downstream_data = load_raw_data(downstream_dir, file_type)

    # Need to pass in a list of events 
    # (for colors dataset we have are looking for events with id = 1, 2, ..., 10)
    preprocess_and_save_epoched_data(downstream_data, config, output_dir, label_fn, include_events_list=list(range(1, 11)))


if __name__ == '__main__':
    args = parser.parse_args()

    # Load the config file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, args.config)
    config = prepare_config(config_path)

    # Preprocess the data
    prepare_data(config)

    # Preprocess the downstream data
    prepare_downstream_data(config)