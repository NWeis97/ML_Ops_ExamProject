import argparse
import io
import os
import re
import sys

# Graphics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import wandb
from sklearn.metrics import accuracy_score, classification_report
from torch import nn, optim
from transformers import (AdamW, GPT2Config, GPT2ForSequenceClassification,
                          GPT2Tokenizer, Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup, set_seed)

sns.set_style("whitegrid")
import pdb

import numpy as np

# Hypereparameter (so far)
# Name of transformers model - will use already pretrained model.
model_name = "gpt2"

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = 2

# Set seed for reproducibility.
set_seed(123)

# Number of training epochs.
epochs = 4

# Number of batches - depending on the max sequence length and CPU memory
batch_size = 32

# Pad or truncate text sequences to a specific length
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = "none"

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    ###################################################
    ################### Load model ####################
    ###################################################

    # Get model configuration.
    print("Loading configuraiton...")
    model_config = GPT2Config.from_pretrained(
        pretrained_model_name_or_path=model_name, num_labels=n_labels
    )

    # Get model's tokenizer.
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    # default to left padding
    tokenizer.padding_side = "left"  # NB: Needs better understanding
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token  # NB: Needs better understanding

    # Get the actual model.
    print("Loading model...")
    model = GPT2ForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path, config=model_config
    )

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    model.to(device)
    print("Model loaded to `%s`" % device)

    ###################################################
    ################### Load data #####################
    ###################################################

    ###################################################
    ############## Calculate predictions ##############
    ###################################################
    # Run evaluation and write results to file
    top_classes = []
    top_probs = []
    all_labels = []
    running_acc = 0

    if Test.__getitem__(0).__len__() == 2:  # If labels are known
        for images, labels in test_set:
            log_ps = model(images)
            ps = torch.exp(log_ps)

            # get top 1 probs per item
            top_p, top_class = ps.topk(1, dim=1)
            top_classes.extend(top_class.squeeze().tolist())
            top_probs.extend(top_p.squeeze().tolist())

            all_labels.extend(labels.tolist())
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))

            # Accumulate loss and accuracy
            running_acc += accuracy
        else:
            # Save predictions to csv and calculate accuracy
            res = pd.DataFrame(
                {
                    "Predictions": top_classes,
                    "Probabilities": top_probs,
                    "Labels": all_labels,
                }
            )

            # Print scores
            running_acc = running_acc / len(test_set)
            print(f"Test accuracy is: {running_acc*100}%")

    else:  # If labels are unknown
        for images, labels in test_set:
            log_ps = model(images)
            ps = torch.exp(log_ps)

            # get top 1 probs per item
            top_p, top_class = ps.topk(1, dim=1)
            top_classes.extend(top_class.squeeze().tolist())
            top_probs.extend(top_p.squeeze().tolist())
        else:
            # Save predictions to csv and calculate accuracy
            res = pd.DataFrame(
                {
                    "Predictions": top_classes,
                    "Probabilities": top_probs,
                    "Labels": all_labels,
                }
            )

    # Save resulting table
    os.makedirs(
        "reports/predictions/" + path_to_model, exist_ok=True
    )  # Create if not already exist
    res.to_csv("reports/predictions/" + path_to_model + modelName + ".csv")

    # Save table to wandb
    my_table = wandb.Table(dataframe=res.iloc[0:500])
    my_table.add_column("image", [wandb.Image(im) for im in images[0:500]])
    # Log your Table to W&B
    wandb.log({"mnist_predictions_first500": my_table})

    print(
        'See predictions in "'
        + "reports/predictions/"
        + path_to_model
        + modelName
        + '.csv"'
    )
    print("Done!\n")


if __name__ == "__main__":
    main()
