## -*- coding: utf-8 -*-

# NB: ENABLE FOR RUNNING SUBSET OF DATA
subset = True

######################################
############## Imports ###############
######################################

# Standard
import os
import datetime
import sys
import re
from tqdm.auto import tqdm

# Data manipulation
import torch
from torch.utils.data import (random_split, DataLoader)
import pandas as pd
import numpy as np
sys.path.append('src/data')
from make_dataset import TorchDataset

# Transformers
from datasets import Dataset, load_metric
from transformers import (AdamW, GPT2Config, GPT2ForSequenceClassification,
                          get_linear_schedule_with_warmup, set_seed,
                          get_scheduler)

# Debugging
import pdb

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Logging (WandB)
import wandb

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
#stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(output_file_handler)
#logger.addHandler(stdout_handler)




######################################
#### Function for training model #####
######################################

def main():

    #*************************************
    #********* Hyperparameters ***********
    #*************************************

    initialize(config_path="../../configs/", job_name="train")
    cfg = compose(config_name="training.yaml")
    cfg_data = compose(config_name="makedata.yaml")
    logger.info('')
    logger.info(f"Data configurations: \n {OmegaConf.to_yaml(cfg_data)}")
    logger.info(f"Training configuration: \n {OmegaConf.to_yaml(cfg)}")
    configs = cfg['hyperparameters']

    # Data and model related
    model_name = cfg_data['hyperparameters']['model_name']
    n_labels = cfg_data['hyperparameters']['n_labels']
    data_output_filepath = cfg_data['hyperparameters']['output_filepath']
    
    # Training related
    batch_size = configs['batch_size']
    lr = configs['lr']
    lr_scheduler = configs['lr_scheduler']
    warmup_step_perc = configs['warmup_step_perc']
    epochs = configs['epochs']
    seed = configs['seed']
    optimizer_type = configs['optimizer_type']
    batch_ratio_validation = configs['batch_ratio_validation']
    weight_decay = configs['weight_decay']


    # Set seed for reproducibility.
    set_seed(seed)
    torch.manual_seed(seed)


    #*************************************
    #************ Load Data **************
    #*************************************

    # Load data and put in DataLoader (also split into train and validation data)
    logger.info("Loading data and splitting training and validation set...")
    Train = torch.load(data_output_filepath + "train_dataset.pt")

    # To be out-commented (only running a subset of the data)
    if subset == True:
        Train = Train.__select__(0,1000)

    num_val = int(batch_ratio_validation*Train.__len__())
    (Train, Val) = random_split(Train, [Train.__len__()-num_val,num_val])

    train_set = DataLoader(Train, batch_size=batch_size, shuffle=True)
    val_set = DataLoader(Val, batch_size=batch_size, shuffle=False)

    #*************************************
    #*********** Load Model **************
    #*************************************

    # Get model configuration.
    logger.info("Loading configuration...")
    model_config = GPT2Config.from_pretrained(
        pretrained_model_name_or_path=model_name, num_labels=n_labels
    )
    # Get the actual model.
    logger.info("Loading model...")
    model = GPT2ForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name, config=model_config
    )

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Move model to cuda if applicable
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    model.to(device)    


    #*************************************
    #********* Set optimizer *************
    #*************************************
    logger.info("Setting up optimizer...")
    if optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)
    else:
        logger.debug("Unknown optimizer_type! Select either:\n - adamw")

    if not(lr_scheduler == 'none' or lr_scheduler == ''):
        logger.info("Setting up scheduler...")
        num_training_steps = epochs * len(train_set)
        num_warmup_steps = int(warmup_step_perc * num_training_steps)
        lr_scheduler = get_scheduler(lr_scheduler,
                                     optimizer=optimizer,
                                     num_warmup_steps=num_warmup_steps,
                                     num_training_steps=num_training_steps
                                    )


    #*************************************
    #********** Train model **************
    #*************************************
    logger.info('')
    logger.info('Training model...')
    logger.info('')

    # Extra
    progress_bar = tqdm(range(epochs * len(train_set)))

    # Initialize lists
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for e in range(epochs):
        # Reset for new epoch
        model.train()
        acc_train= load_metric("accuracy")
        acc_val= load_metric("accuracy")
        running_loss_train = 0
        running_loss_val = 0
        
        for batch in train_set:
            # Load batch and send to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            # Update optimizer and scheduler
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            if not(lr_scheduler == 'none' or lr_scheduler == ''):
                lr_scheduler.step()

            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Update running loss and accuracy (train)
            running_loss_train += loss.item()
            acc_train.add_batch(predictions=predictions, references=batch["labels"])
        else:
            model.eval()
            for batch in val_set:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)

                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                # Update running loss and accuracy (train)
                running_loss_val += outputs.loss.item()
                acc_val.add_batch(predictions=predictions, references=batch["labels"])


            # Calculate training and validation loss and log results
            acc_train = acc_train.compute()
            acc_val = acc_val.compute()
            train_losses.append(running_loss_train / len(train_set))
            val_losses.append(running_loss_val / len(val_set))
            train_accs.append(acc_train['accuracy'])
            val_accs.append(acc_val['accuracy'])
            logger.info("Epoch: " + str(e+1) + "/" + str(epochs))
            logger.info("Training_loss: " + str(train_losses[e]))
            logger.info("Validation_loss: " + str(val_losses[e]))
            logger.info("Training_accuracy: " + str(train_accs[e]))
            logger.info("Validation_accuracy: " + str(val_accs[e]))
            logger.info("")


    # Save model
    os.makedirs("./models/" + logfp, exist_ok=True) #Create if not already exist
    model.save_pretrained(
        "models/"
        + logfp
        + '/'
        + model_name
    )

if __name__ == "__main__":
    main()
