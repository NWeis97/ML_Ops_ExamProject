<<<<<<< HEAD
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
=======
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

>>>>>>> a38d39ddfd405d4659fc64ca7522a72fedbbc23f

if __name__ == "__main__":
    main()
