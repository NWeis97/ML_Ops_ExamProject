## -*- coding: utf-8 -*-

######################################
############## Imports ###############
######################################

# Standard
import os
import datetime
import re
import sys

# Data manipulation
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Transformers
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2Config

# Debugging
import pdb

# Configs
from hydra import compose, initialize
from omegaconf import OmegaConf

# Logging
import logging
logfp =  str(datetime.datetime.now().date()) + '/' + str(datetime.datetime.now().strftime("%H-%M-%S"))
result = re.search("(.*).py", os.path.basename(__file__))
fileName = result.group(1)
os.makedirs('outputs/'+fileName+'/'+logfp, exist_ok = True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
output_file_handler = logging.FileHandler('outputs/'+fileName+'/'+logfp+'.log', encoding='utf-8')
logger.addHandler(output_file_handler)

######################################
####### Define Dataset class #########
######################################
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

    def __select__(self,idx_from,idx_to):
        items = {key: val[idx_from:idx_to] for key, val in self.encodings.items()}
        return TorchDataset(items,self.labels[idx_from:idx_to])



######################################
######### Data load function #########
######################################

def main():

    #*************************************
    #********* Hyperparameters ***********
    #*************************************

    initialize(config_path="../../configs/", job_name="model")
    cfg = compose(config_name="makedata.yaml")
    logger.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    configs = cfg['hyperparameters']

    # Hyperparameters extracted
    input_filepath = configs['input_filepath']
    output_filepath = configs['output_filepath']
    model_name = configs['model_name']
    #n_labels = configs['n_labels']
    seed = configs['seed']


    #*************************************
    #********** Read raw data ************
    #*************************************

    # Read data from raw
    df = pd.read_csv(input_filepath + "train.csv", sep=",")
    X = df['text'] #df[['id', 'keyword', 'location', 'text']] let's keep it simple to begin with
    y = df["target"]

    #Split data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


    #*************************************
    #***** Tokenizer Initialization ******
    #*************************************

    # Get model's tokenizer.
    logger.info('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # default to left padding
    tokenizer.padding_side = "left"

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token


    #*************************************
    #************* Tokenize **************
    #*************************************

    # Tokenize text using Transformers gpt-2
    logger.info('Tokenizing training and test datasets')
    train_encodings = tokenizer(X_train.to_list(),truncation=True, padding=True)
    test_encodings = tokenizer(X_test.to_list(),truncation=True, padding=True)

    # Convert to PyTorch Datasets
    logger.info('Converting to TensorDataset')
    train_set = TorchDataset(train_encodings,y_train.to_list())
    test_set = TorchDataset(test_encodings,y_test.to_list())


    #*************************************
    #************ Save data **************
    #*************************************

    # Save datasets in data/processed
    X_test.to_csv(output_filepath + 'test_tweets.csv')
    torch.save(train_set, output_filepath + "train_dataset.pt")
    torch.save(test_set, output_filepath + "test_dataset.pt")

    logger.info("Successfully loaded and preprocessed data: train_dataset and test_dataset")



if __name__ == '__main__':
    main()
