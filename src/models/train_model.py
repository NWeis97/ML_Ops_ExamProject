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
import argparse
import glob

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
                          get_scheduler,WEIGHTS_NAME, CONFIG_NAME)

# Debugging
import pdb

# WandB
import wandb

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Logging (WandB)
import wandb

# Import the Secret Manager client library.
from google.cloud import storage, secretmanager
from google.oauth2 import service_account

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



#*************************************
#******** Get WandB API Key **********
#*************************************
def get_wandb_api_key(project_id):
    # Get cridentials from json
    credentials = service_account.Credentials.from_service_account_file(
    './credentials/SecretManagerAccessor.json')
    # ID of the secret to create.
    secret_id = "wandb-apikey-secret"
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient(credentials=credentials)
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    # Access the secret version.
    return client.access_secret_version(name=name).payload.data.decode("utf-8")


#*************************************
#************ Load Data **************
#*************************************
def load_data(data_output_filepath, batch_ratio_validation, batch_size):

    # Load data and put in DataLoader (also split into train and validation data)
    print("Loading data and splitting training and validation set...")
    Train = torch.load(data_output_filepath + "train_dataset.pt")

    # To be out-commented (only running a subset of the data)
    if subset == True:
        Train = Train.__select__(0,300)

    num_val = int(batch_ratio_validation*Train.__len__())
    (Train, Val) = random_split(Train, [Train.__len__()-num_val,num_val])

    train_set = DataLoader(Train, batch_size=batch_size, shuffle=True)
    val_set = DataLoader(Val, batch_size=batch_size, shuffle=False)
    
    return train_set, val_set


#*************************************
#*********** Load Model **************
#*************************************
def load_model(model_name, n_labels, device):
    # Get model configuration.
    print("Loading configuration...")
    model_config = GPT2Config.from_pretrained(
        pretrained_model_name_or_path=model_name, num_labels=n_labels
    )

    # Get the actual model.
    print("Loading model...")
    model = GPT2ForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name, config=model_config
    )

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Move model to cuda if applicable
    model.to(device)    
    
    return model

#*************************************
#*********** Save model **************
#*************************************
def save_model(model, job_dir, model_name):
    """Saves the model to Google Cloud Storage

    Args:
      args: contains name for saved model.
    """
    local_model_path = ""

    scheme = 'gs://'
    bucket_name = job_dir[len(scheme):].split('/')[0]

    prefix = '{}{}/'.format(scheme, bucket_name)
    bucket_path = job_dir[len(prefix):].rstrip('/')

    datetime_ = datetime.datetime.now().strftime('model_%Y%m%d_%H%M%S')

    if bucket_path:
        model_path = '{}/{}/{}'.format(bucket_path, datetime_, model_name)
    else:
        model_path = '{}/{}'.format(datetime_, model_name)


    #If we have a distributed model, save only the encapsulated model
    #It is wrapped in PyTorch DistributedDataParallel or DataParallel
    model_to_save = model.module if hasattr(model, 'module') else model
    #If you save with a pre-defined name, you can use 'from untrained' to load
    output_model_file = os.path.join(local_model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(local_model_path, CONFIG_NAME)

    # Save model state_dict and configs locally
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    # 
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(os.path.join(model_path, WEIGHTS_NAME))
    blob.upload_from_filename(WEIGHTS_NAME)
    blob = bucket.blob(os.path.join(model_path, CONFIG_NAME))
    blob.upload_from_filename(CONFIG_NAME)

    """
    print('output_model_file: ' + output_model_file)
    print('output_model_file: ' + output_config_file)
    # Find model locally on vm and upload all files to gs
    assert os.path.isdir(local_model_path)
    
    print('bucket_name: ' + bucket_name)
    for local_file in glob.glob(local_model_path+ '/**'):
        print('local_file: ' + local_file)
        print('model_path: ' + model_path)
        file_name = os.path.basename(local_file)
        print('file_name: ' + file_name)
    """
        
    


#*************************************
#********* Set optimizer *************
#*************************************
def load_optimizer(model, train_set, optimizer_type, lr, weight_decay, lr_scheduler, warmup_step_perc, epochs):
    print("Setting up optimizer...")
    if optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)
    else:
        logger.debug("Unknown optimizer_type! Select either:\n - adamw")

    if not(lr_scheduler == 'none' or lr_scheduler == ''):
        print("Setting up scheduler...")
        num_training_steps = epochs * len(train_set)
        num_warmup_steps = int(warmup_step_perc * num_training_steps)
        lr_scheduler = get_scheduler(lr_scheduler,
                                     optimizer=optimizer,
                                     num_warmup_steps=num_warmup_steps,
                                     num_training_steps=num_training_steps
                                    )
    return optimizer, lr_scheduler

    #*************************************
    #********** Train model **************
    #*************************************
def train(model, train_set, optimizer, device, lr_scheduler, progress_bar):

    # Reset for new epoch
    model.train()
    acc_train= load_metric("accuracy")
    running_loss_train = 0
    
    for batch in train_set:
        # Load batch and send to device
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Update optimizer and scheduler
        optimizer.step()
        optimizer.zero_grad()
        if not(lr_scheduler == 'none' or lr_scheduler == ''):
            lr_scheduler.step()

        # Get predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Update running loss and accuracy (train)
        running_loss_train += loss.item()
        acc_train.add_batch(predictions=predictions, references=batch["labels"])

        # Update progress bar
        progress_bar.update(1)
        
    
    # Evaluate loss and acc
    acc_train = acc_train.compute()
    train_loss = running_loss_train / len(train_set)
    train_acc = acc_train['accuracy']
    
    
    return train_loss, train_acc


def validate(model, val_set, device):
    #Evaluation mode
    model.eval()
    running_loss_val = 0

    acc_val = load_metric("accuracy")
        
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
    acc_val  = acc_val.compute()
    val_loss = running_loss_val / len(val_set)
    val_acc  = acc_val['accuracy']
    
    return val_loss, val_acc



def run():
    """Load the data, train, evaluate, and export the model for serving and
     evaluating.
     """
     #*************************************
     #************ Arguments **************
     #*************************************
    print('Loading arguments...\n')

    args_parser = argparse.ArgumentParser()
    # Save to GS bucket
    # Saved model arguments
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to export models')
    args_parser.add_argument(
        '--project-id',
        help='GCS project id name')
        
    args = args_parser.parse_args()


    #*************************************
    #********* Hyperparameters ***********
    #*************************************
    print('Loading hyperparameters...\n')

    initialize(config_path="../../configs/", job_name="train")
    cfg = compose(config_name="training.yaml")
    cfg_data = compose(config_name="makedata.yaml")
    print(f"\nData configurations: \n {OmegaConf.to_yaml(cfg_data)}")
    print(f"Training configuration: \n {OmegaConf.to_yaml(cfg)}")
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
    #*********** WandB setup *************
    #*************************************
    print('Setting up WandB connection and initialization...\n')

    args.project_id = 'examproject-mlops'
    wandb_api_key = get_wandb_api_key(args.project_id)
    os.environ["WANDB_API_KEY"] = wandb_api_key


    #os.system("gcloud auth login")
    #os.system("gcloud auth activate-service-account ACCOUNT --key-file=KEY-FILE")

    wandb.init(project=args.project_id, 
              entity="mlops_swaggers",
              config={"Model&Data": cfg_data['hyperparameters'], "Train": configs},
              job_type="Train")
    
    
    #*************************************
    #************** Run ******************
    #*************************************
    
    # Load data
    print('Loading data...\n')
    train_set, val_set = load_data(data_output_filepath, 
                                   batch_ratio_validation, 
                                   batch_size
                                   )
    
    # Load model
    print('Loading model...\n')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(model_name, n_labels, device)

    # Optional
    wandb.watch(model, log_freq=100)

    # Set optimizer
    optimizer, lr_scheduler = load_optimizer(model, train_set, optimizer_type, lr, weight_decay, lr_scheduler, warmup_step_perc, epochs)
    
    # Train the model
    print('Training model...\n')
    
    # Extra
    progress_bar = tqdm(range(epochs * len(train_set)))
    
    for e in range(epochs):
        train_loss, train_acc = train(model, 
                                     train_set, 
                                     optimizer, 
                                     device, 
                                     lr_scheduler, 
                                     progress_bar
                                     )
        val_loss, val_acc = validate(model, 
                                    val_set, 
                                    device
                                    )
        
        
        # Logger 
        print("Epoch: " + str(e+1) + '/' + str(epochs))
        print("Training_loss: " + str(train_loss))
        print("Training_accuracy: " + str(train_acc))
        print("Validation_loss: " + str(val_loss))
        print("Validation_accuracy: " + str(val_acc) + "\n")

        #wandb
        wandb.log({"Training_loss": train_loss})
        wandb.log({"Validation_loss": val_loss})
        wandb.log({"Training_accuracy": train_acc})
        wandb.log({"Validation_accuracy": val_acc})

    
    # Save model
    print("Saving model...\n")
    save_model(model, args.job_dir, model_name)
    

if __name__ == "__main__":
    run()
    
        
