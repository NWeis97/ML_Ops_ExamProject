# # -*- coding: utf-8 -*-

######################################
############## Imports ###############
######################################

# Standard
import os

# Data manipulation
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Transformers
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2Config

# Debugging
import pdb


######################################
########## Hyperparameters ###########
######################################
model_name = 'gpt2'
n_labels = 2


######################################
####### Define Dataset class #########
######################################
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



######################################
######### Data load function #########
######################################

def Load_Data():

    #*************************************
    #********** Read raw data ************
    #*************************************

    # Read data from raw
    df = pd.read_csv("./data/raw/train.csv", sep=",")
    X = df['text'] #df[['id', 'keyword', 'location', 'text']] let's keep it simple to begin with
    y = df["target"]

    #Split data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    #*************************************
    #***** Tokenizer Initialization ******
    #*************************************

    # Get model configuration.
    print('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=n_labels)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # default to left padding
    tokenizer.padding_side = "left"

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token


    #*************************************
    #************* Tokenize **************
    #*************************************

    # Tokenize text using Transformers gpt-2
    print('Tokenizing training and test datasets')
    train_encodings = tokenizer(X_train.to_list(),truncation=True, padding=True)
    test_encodings = tokenizer(X_test.to_list(),truncation=True, padding=True)

    # Convert to PyTorch Datasets
    print('Converting to TensorDataset')
    train_set = TorchDataset(train_encodings,y_train.to_list())
    test_set = TorchDataset(test_encodings,y_test.to_list())


    #*************************************
    #************ Save data **************
    #*************************************

    # Save datasets in data/processed
    output_filepath = "./data/processed/"
    torch.save(train_set, output_filepath + "train_dataset.pt")
    torch.save(test_set, output_filepath + "test_dataset.pt")

    print("Finished loading and preprocessing data: train_dataset and test_dataset")



if __name__ == '__main__':
    Load_Data()
