# -*- coding: utf-8 -*-

# #####################################
# ############# Imports ###############
# #####################################

# Standard
import os
import datetime
import re
from tqdm.auto import tqdm

# Data manipulation
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

# Transformers
from datasets import load_metric
from transformers import GPT2ForSequenceClassification

# Debugging
import pdb

# Import the Secret Manager client library.
from google.cloud import storage
from google.oauth2 import service_account

# Graphics
import seaborn as sns

from ml_things import plot_confusion_matrix

# Logging (WandB)
# import wandb

# Configs
from hydra import compose, initialize
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

sns.set_style("whitegrid")


class TorchDataset(torch.utils.data.Dataset):
    """
    Torch data set class
    
    __init__ initialize dataset
    __getitem__ get item in dataset
    __len__ length of dataset
    __select__ select pieces of dataset
    
    """
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

    def __select__(self, idx_from, idx_to):
        items = {key: val[idx_from:idx_to] for key, val in self.encodings.items()}
        return TorchDataset(items, self.labels[idx_from:idx_to])

# #####################################
# ### Function for training model #####
# #####################################
def main():

    # *************************************
    # ********* Hyperparameters ***********
    # *************************************

    initialize(config_path="../../configs/", job_name="predict")
    cfg = compose(config_name="predict.yaml")
    cfg_data = compose(config_name="makedata.yaml")
    logger.info("")
    logger.info(f"Data configurations: \n {OmegaConf.to_yaml(cfg_data)}")
    logger.info(f"Training configuration: \n {OmegaConf.to_yaml(cfg)}")
    configs = cfg["hyperparameters"]

    # Data and model related
    model_name = cfg_data["hyperparameters"]["model_name"]
    data_output_filepath = cfg_data["hyperparameters"]["output_filepath"]

    # Predict related
    path_to_model = configs["path_to_model"]

    # *************************************
    # *********** Load Data  **************
    # *************************************

    # Load data and put in DataLoader (also split into train and validation data)
    logger.info("Loading data and model")
    Test = torch.load(data_output_filepath + "test_dataset.pt")

    test_set = DataLoader(Test, batch_size=64, shuffle=False)
    pdb.set_trace()

    # *************************************
    # *********** Load Model **************
    # *************************************

    # Get the actual model
    logger.info("Loading model...")

    # Get model from bucket (cloud storage)
    credentials = service_account.Credentials.from_service_account_file(os.getcwd()+'/credentials/ReadBucketData.json')
    storage_client = storage.Client(project='examproject-mlops',credentials=credentials)
    bucket_name = 'gpt2_exam_project_bucket'
    state_data = 'pytorch_model.bin'
    config_data = 'config.json'
    temp_state_data = 'pytorch_model.bin'
    temp_config_data = 'config.json'

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob('models/base/'+path_to_model+state_data)
    blob.download_to_filename(temp_state_data)
    blob = bucket.blob('models/base/'+path_to_model+config_data)
    blob.download_to_filename(temp_config_data)

    model = GPT2ForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=os.getcwd() + '/' + temp_state_data,
        local_files_only=True, config=os.getcwd() + '/' + temp_config_data
    )

    # remove files again
    os.remove(os.getcwd() + '/' + temp_state_data)
    os.remove(os.getcwd() + '/' + temp_config_data)

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Move model to cuda if applicable
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # *************************************
    # ********* Evaluate model ************
    # *************************************
    logger.info("")
    logger.info("Evaluating model...")
    logger.info("")

    # Run evaluation and write results to file
    predictions = []
    labels_all = []
    probs_all = []
    model.eval()
    acc_test = load_metric("accuracy")

    # Extra
    progress_bar = tqdm(range(len(test_set)))

    # Run batch through model
    for batch in test_set:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        # Get predictions and probabilities
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        logits_max = torch.max(logits, dim=-1)
        probs = np.exp(logits_max.values) / (1 + np.exp(logits_max.values))

        # Add to accumulating lists
        predictions.extend(preds.tolist())
        labels_all.extend(batch["labels"].tolist())
        probs_all.extend(probs.tolist())

        # Calculate accuracy
        acc_test.add_batch(predictions=preds, references=batch["labels"])

        progress_bar.update(1)

    acc_test = acc_test.compute()
    print(f"Test accuracy is: {acc_test['accuracy']*100}%")

    # Create the evaluation report.
    evaluation_report = classification_report(
        labels_all, predictions, target_names=["Not Disaster", "Disaster"], output_dict=True
    )
    evaluation_report = pd.DataFrame(evaluation_report)
    # Show the evaluation report.
    print(evaluation_report)

    # Plot confusion matrix.
    os.makedirs(
        "reports/figures/" + model_name + "/" + path_to_model, exist_ok=True
    )  # Create if not already exist
    plot_confusion_matrix(
        y_true=labels_all,
        y_pred=predictions,
        classes=["Not Disaster", "Disaster"],
        normalize=True,
        show_plot=False,
        magnify=0.1,
        use_dpi=200,
        path="reports/figures/" + model_name + "/" + path_to_model + "confusion_matrix.png",
    )

    # Save predictions to csv and calculate accuracy (NB: need actual tweets as well)
    res = pd.DataFrame(
        {"Predictions": predictions, "Probabilities": probs_all, "Labels": labels_all}
    )

    # Save resulting table
    os.makedirs(
        "reports/predictions/" + model_name + "/" + path_to_model, exist_ok=True
    )  # Create if not already exist
    res.to_csv("reports/predictions/" + model_name + "/" + path_to_model + "predictions.csv")
    evaluation_report.to_csv(
        "reports/predictions/" + model_name + "/" + path_to_model + "evaluation_report.csv"
    )


if __name__ == "__main__":
    main()
