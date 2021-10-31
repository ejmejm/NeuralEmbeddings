# Nueral Embeddings

This is the code for our paper on learning robust representations for MEG data (paper name TBD). This repository implements a transformer-based architecture and self-supervised learning methods that allow learning representations for MEG data using unlabeled data.

## Prerequisites

This project has only been tested with Python 3.9. Other versions may also work, but functionality is not guaranteed. To get started, clone the repository then run a pip install on the `requirements.txt` file to get the necessary modules installed. It is recommended you do this in a new environment to avoid dependency conflicts.

```
git clone https://github.com/ejmejm/NeuralEmbeddings
cd ./NeuralEmbeddings
pip install -r requirements.txt
```

\* It is recommended that you install PyTorch with GPU support for faster training times. By default the above commands will install the CPU version, so an extra step is required if you want the GPU compatible version. For installation instructions please visit https://pytorch.org/.

## Getting Started

To get started, you first need to collect data. Once you have data, create two directories under pretraining called `data/train_datasets` and `data/test_datasets`, and move the uncompressed folders in there depending on what you want to use for training and testing. Each folder entry in the `pretraining/data/{train|test}_datasets` directories should be a different dataset. Then the fif files should be somewhere within each dataset folder (We check recursively, so it's okay if there are more nested folders).

The next two steps are to run the data preprocessor and then the model training. Each of these steps requires a shared yaml config file. Three separate premade config files are provided, one for running tests, masked sequeunce modeling training, and contrastive predictive coding training. We recommend starting with the test config file at `pretraining/configs/test_config.yaml` because it is the least compute intensive and contains all available options. Once you have chosen the desired config file, go through the settings and make sure to configure them as you like (make sure to either set the wandb mode to disabled or change the project/entity name to your own). The below examples use the test config file, but you can switch it out for another config file at any time.

To preprocess data (may take some time to run), and then run training, run the following commands:
```
cd preprocessing
python prepare_data.py -c configs/test_config.yaml # Run preprocessing
python run_model.py -c configs/test_config.yaml # Run training
```

Basic metric will be printed out, and more complicated metrics will be logged to wandb.

## Implementation

The implementation consists of 3 separate transformers that each serve different purposes. You can opt to train with masked sequence modeling (based on masked language modeling) or contrastive predictive coding. More details to come.

## Paper Authors

* **Edan Meyer** - [GitHub Profile](https://github.com/ejmejm)
* **Faisal Abutarab** - [GitHub Profile](https://github.com/FaisalAbutarab)
* **Katyani Singh** - [Github Profile](https://github.com/katyanisingh)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
