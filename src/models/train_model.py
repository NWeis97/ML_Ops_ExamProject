# -*- coding: utf-8 -*-
# #####################################
# ############# Imports ###############
# #####################################

# Standard
import os
import datetime
import re
from tqdm.auto import tqdm
import argparse

# Data manipulation
import torch
from torch.utils.data import random_split, DataLoader

# Transformers
from datasets import load_metric
from transformers import (
    AdamW,
    GPT2Config,
    GPT2ForSequenceClassification,
    set_seed,
    get_scheduler,
    WEIGHTS_NAME,
    CONFIG_NAME,
)

# Debugging
# import pdb

# WandB
import wandb

# Import the Secret Manager client library.
from google.cloud import storage

# Configs
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import hydra

from omegaconf import OmegaConf

# Logging
import logging

logfp = (
    str(datetime.datetime.now().date()) + "/" + str(datetime.datetime.now().strftime("%H-%M-%S"))
)
result = re.search("(.*).py", os.path.basename(__file__))
fileName = result.group(1)
os.makedirs("outputs/" + fileName + "/" + logfp, exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
output_file_handler = logging.FileHandler(
    "outputs/" + fileName + "/" + logfp + ".log", encoding="utf-8"
)
logger.addHandler(output_file_handler)

# Get TorchDataset clas
# import sys
# sys.path.append("src/data")
# from src.data.make_dataset import TorchDataset


# *************************************
# ************ Load Data **************
# *************************************
def load_data(data_output_filepath, batch_ratio_validation, batch_size, subset):

    # Load data and put in DataLoader (also split into train and validation data)
    Train = torch.load(data_output_filepath + "train_dataset.pt")

    # To be out-commented (only running a subset of the data)
    if subset == "True":
        Train = Train.__select__(0, 300)

    num_val = int(batch_ratio_validation * Train.__len__())
    (Train, Val) = random_split(Train, [Train.__len__() - num_val, num_val])

    train_set = DataLoader(Train, batch_size=batch_size, shuffle=True)
    val_set = DataLoader(Val, batch_size=batch_size, shuffle=False)

    return train_set, val_set


# *************************************
# *********** Load Model **************
# *************************************
def load_model(model_name, n_labels, device):
    # Get model configuration.
    model_config = GPT2Config.from_pretrained(
        pretrained_model_name_or_path=model_name, num_labels=n_labels
    )

    # Get the actual model.
    model = GPT2ForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name, config=model_config
    )

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Move model to cuda if applicable
    model.to(device)

    return model


# *************************************
# *********** Save model **************
# *************************************
def save_model(model, job_dir, model_name):
    """Saves the model to Google Cloud Storage

    Args:
      args: contains name for saved model.
    """
    local_model_path = ""

    scheme = "gs://"
    bucket_name = job_dir[len(scheme):].split("/")[0]

    prefix = "{}{}/".format(scheme, bucket_name)
    bucket_path = job_dir[len(prefix):].rstrip("/")

    datetime_ = datetime.datetime.now().strftime("model_%Y%m%d_%H%M%S")

    if bucket_path:
        model_path = "{}/{}/{}".format(bucket_path, datetime_, model_name)
    else:
        model_path = "{}/{}".format(datetime_, model_name)

    # If we have a distributed model, save only the encapsulated model
    # It is wrapped in PyTorch DistributedDataParallel or DataParallel
    model_to_save = model.module if hasattr(model, "module") else model
    # If you save with a pre-defined name, you can use 'from untrained' to load
    output_model_file = os.path.join(local_model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(local_model_path, CONFIG_NAME)

    # Save model state_dict and configs locally
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    # Save model to bucket
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(os.path.join(model_path, WEIGHTS_NAME))
    blob.upload_from_filename(WEIGHTS_NAME)
    blob = bucket.blob(os.path.join(model_path, CONFIG_NAME))
    blob.upload_from_filename(CONFIG_NAME)


# *************************************
# ********* Set optimizer *************
# *************************************
def load_optimizer(
    model, train_set, optimizer_type, lr, weight_decay, lr_scheduler, warmup_step_perc, epochs
):

    if optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        logger.debug("Unknown optimizer_type! Select either:\n - adamw")

    if not (lr_scheduler == "none" or lr_scheduler == ""):
        print("Setting up scheduler...")
        num_training_steps = epochs * len(train_set)
        num_warmup_steps = int(warmup_step_perc * num_training_steps)
        lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    return optimizer, lr_scheduler


# *************************************
# ********** Train model **************
# *************************************
def train(model, train_set, optimizer, device, lr_scheduler, progress_bar):

    # Reset for new epoch
    model.train()
    acc_train = load_metric("accuracy")
    running_loss_train = 0

    step = 0
    for batch in train_set:
        # Update step
        step += 1

        # Load batch and send to device
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Update optimizer and scheduler
        optimizer.step()
        optimizer.zero_grad()
        if not (lr_scheduler == "none" or lr_scheduler == ""):
            lr_scheduler.step()

        # Get predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Update running loss and accuracy (train)
        running_loss_train += loss.item()
        acc_train.add_batch(predictions=predictions, references=batch["labels"])

        # Update progress bar
        progress_bar.update(1)

        # Write progress for gcp
        print(str(step) + "/" + str(len(train_set)))

    # Evaluate loss and acc
    acc_train = acc_train.compute()
    train_loss = running_loss_train / len(train_set)
    train_acc = acc_train["accuracy"]

    return train_loss, train_acc


def validate(model, val_set, device):
    # Evaluation mode
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
    acc_val = acc_val.compute()
    val_loss = running_loss_val / len(val_set)
    val_acc = acc_val["accuracy"]

    return val_loss, val_acc


def run():
    """Load the data, train, evaluate, and export the model for serving and
    evaluating.
    """
    # *************************************
    # ************ Arguments **************
    # *************************************
    print("Loading arguments...\n")

    args_parser = argparse.ArgumentParser()
    # Save to GS bucket
    # Saved model arguments
    args_parser.add_argument("--job-dir", help="GCS location to export models")
    args_parser.add_argument("--project-id", help="GCS project id name")
    args_parser.add_argument("--subset", help="Use subset of training data?")

    # WandB related
    args_parser.add_argument("--wandb-api-key", help="Your WandB API Key for login")
    args_parser.add_argument("--entity", help="WandB project entity")

    # Add arguments
    args = args_parser.parse_args()

    # *************************************
    # ********* Hyperparameters ***********
    # *************************************
    print("Loading hyperparameters...\n")

<<<<<<< HEAD
    #*************************************
    #********* Hyperparameters ***********
    #*************************************
    print('Loading hyperparameters...\n')
    
    GlobalHydra().clear()
    
=======
    GlobalHydra.instance().clear()
>>>>>>> e9b8ab593aab8848da62bfac7cceb0445bf32355
    initialize(config_path="../../configs/", job_name="train")
    cfg = compose(config_name="training.yaml")
    cfg_data = compose(config_name="makedata.yaml")
    print(f"Data configurations: \n {OmegaConf.to_yaml(cfg_data)}")
    print(f"Training configuration: \n {OmegaConf.to_yaml(cfg)}")
<<<<<<< HEAD
    configs = cfg['hyperparameters']
    
=======
    configs = cfg["hyperparameters"]

>>>>>>> e9b8ab593aab8848da62bfac7cceb0445bf32355
    # Data and model related
    model_name = cfg_data["hyperparameters"]["model_name"]
    n_labels = cfg_data["hyperparameters"]["n_labels"]
    data_output_filepath = cfg_data["hyperparameters"]["output_filepath"]

    # Training related
    batch_size = configs["batch_size"]
    lr = configs["lr"]
    lr_scheduler = configs["lr_scheduler"]
    warmup_step_perc = configs["warmup_step_perc"]
    epochs = configs["epochs"]
    seed = configs["seed"]
    optimizer_type = configs["optimizer_type"]
    batch_ratio_validation = configs["batch_ratio_validation"]
    weight_decay = configs["weight_decay"]

    # Set seed for reproducibility.
    set_seed(seed)
    torch.manual_seed(seed)

    # *************************************
    # *********** WandB setup *************
    # *************************************
    if args.wandb_api_key is not None:
        print("Setting up WandB connection and initialization...\n")

        # Get configs
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

        wandb.init(
            project=args.project_id,
            entity=args.entity,
            config={"Model&Data": cfg_data["hyperparameters"], "Train": configs},
            job_type="Train",
        )

    # *************************************
    # ************** Run ******************
    # *************************************

    # Load data
    print("Loading data...\n")
    train_set, val_set = load_data(
        data_output_filepath, batch_ratio_validation, batch_size, args.subset
    )

    # Load model
    print("Loading model...\n")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(model_name, n_labels, device)

    # If WandB is applicable
    if args.wandb_api_key is not None:
        wandb.watch(model, log_freq=100)
        wandb.log({"Using subset of data": args.subset})

    # Set optimizer
    print("Setting up optimizer...")
    optimizer, lr_scheduler = load_optimizer(
        model, train_set, optimizer_type, lr, weight_decay, lr_scheduler, warmup_step_perc, epochs
    )

    # Train the model
    print("Training model...\n")

    # Extra
    progress_bar = tqdm(range(epochs * len(train_set)))

    for e in range(epochs):
        train_loss, train_acc = train(
            model, train_set, optimizer, device, lr_scheduler, progress_bar
        )
        val_loss, val_acc = validate(model, val_set, device)

        # Logger
        print("Epoch: " + str(e + 1) + "/" + str(epochs))
        print("Training_loss: " + str(train_loss))
        print("Training_accuracy: " + str(train_acc))
        print("Validation_loss: " + str(val_loss))
        print("Validation_accuracy: " + str(val_acc) + "\n")

        # wandb
        if args.wandb_api_key is not None:
            wandb.log(
                {
                    "Training_loss": train_loss,
                    "Validation_loss": val_loss,
                    "Training_accuracy": train_acc,
                    "Validation_accuracy": val_acc,
                }
            )

    # Save model
    if args.job_dir is not None:
        print("Saving model...\n")
        save_model(model, args.job_dir, model_name)
    else:
        print(
            "Job_dir not given, thus not saving model (will no save model when running locally)..."
        )


if __name__ == "__main__":
    run()
