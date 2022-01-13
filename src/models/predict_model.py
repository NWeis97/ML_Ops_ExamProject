## -*- coding: utf-8 -*-
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
from sklearn.metrics import classification_report

# Transformers
from datasets import load_metric
from transformers import (GPT2Config, GPT2ForSequenceClassification)

# Debugging
import pdb

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from ml_things import plot_dict, plot_confusion_matrix, fix_text

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
logger.addHandler(output_file_handler)




######################################
#### Function for training model #####
######################################

def main():

    #*************************************
    #********* Hyperparameters ***********
    #*************************************

    initialize(config_path="../../configs/", job_name="predict")
    cfg = compose(config_name="predict.yaml")
    cfg_data = compose(config_name="makedata.yaml")
    logger.info('')
    logger.info(f"Data configurations: \n {OmegaConf.to_yaml(cfg_data)}")
    logger.info(f"Training configuration: \n {OmegaConf.to_yaml(cfg)}")
    configs = cfg['hyperparameters']

    # Data and model related
    model_name = cfg_data['hyperparameters']['model_name']
    n_labels = cfg_data['hyperparameters']['n_labels']
    data_output_filepath = cfg_data['hyperparameters']['output_filepath']
    
    # Predict related
    path_to_model = configs['path_to_model']

    #*************************************
    #*********** Load Data  **************
    #*************************************

    # Load data and put in DataLoader (also split into train and validation data)
    logger.info("Loading data and model")
    Test = torch.load(data_output_filepath + "test_dataset.pt")

    test_set = DataLoader(Test, batch_size=64, shuffle=False)


    #*************************************
    #*********** Load Model **************
    #*************************************

    # Get model configuration.
    logger.info("Loading configuration...")
    #model_config = GPT2Config.from_pretrained(
    #    pretrained_model_name_or_path='../../models/' + path_to_model + model_name, num_labels=n_labels, local_files_only=True
    #)
    # Get the actual model.
    logger.info("Loading model...")
    model = GPT2ForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path= os.getcwd() + '/models/' + path_to_model + model_name, local_files_only=True #, config=model_config
    )

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Move model to cuda if applicable
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    model.to(device)    


    #*************************************
    #********* Evaluate model ************
    #*************************************
    logger.info('')
    logger.info('Evaluating model...')
    logger.info('')

    # Run evaluation and write results to file
    predictions = []
    labels_all = []
    probs_all = []
    model.eval()
    acc_test= load_metric("accuracy")

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
        probs = np.exp(logits_max.values)/(1+np.exp(logits_max.values))

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
    evaluation_report = classification_report(labels_all, predictions, target_names=['Not terror','Terror'], output_dict=True)
    evaluation_report = pd.DataFrame(evaluation_report)
    # Show the evaluation report.
    print(evaluation_report)

    # Plot confusion matrix.
    os.makedirs("reports/figures/"+ model_name + '/' + path_to_model, exist_ok=True) #Create if not already exist
    plot_confusion_matrix(y_true=labels_all, y_pred=predictions, 
                          classes=['Not terror','Terror'], normalize=True, show_plot=False,
                          magnify=0.1,use_dpi=200,path="reports/figures/"
                                                    + model_name
                                                    + '/'
                                                    + path_to_model
                                                    + 'confusion_matrix.png');

    # Save predictions to csv and calculate accuracy (NB: need actual tweets as well)
    res = pd.DataFrame(
        {
            "Predictions": predictions,
            "Probabilities": probs_all,
            "Labels": labels_all
        }
    )
    
    # Save resulting table
    os.makedirs("reports/predictions/"+ model_name + '/' + path_to_model, exist_ok=True) #Create if not already exist
    res.to_csv("reports/predictions/" + model_name + '/' + path_to_model + 'predictions.csv')
    evaluation_report.to_csv("reports/predictions/" + model_name + '/' + path_to_model + 'evaluation_report.csv')

if __name__ == "__main__":
    main()
