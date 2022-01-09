# # -*- coding: utf-8 -*-
import os

import torch
import pdb

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/raw/train.csv", sep=",")
df.columns
X = df[["id", "keyword", "location", "text"]]
y = df["target"]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.8)

train_set = pd.concat((X_train_val, y_train_val), axis=1)
test_set = pd.concat((X_test, y_test), axis=1)


# Save files
save_path = "./data/processed/"
train_set.to_csv(save_path + "train_processed.csv")
test_set.to_csv(save_path + "test_processed.csv")

print("Finished loading data into memory: X_train_val, X_test, y_train_val, y_test")


# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
