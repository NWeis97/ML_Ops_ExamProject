#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:23:45 2022

@author: frederikhartmann
"""

## -*- coding: utf-8 -*-

# NB: ENABLE FOR RUNNING SUBSET OF DATA
subset = True

######################################
############## Imports ###############
######################################

# Data manipulation
import torch
import os

# Debugging
import pdb

import seaborn as sns
sns.set_style("whitegrid")

import hydra

# Import pytest
import pytest

from src.data.make_dataset import read_data, tokenizer, convert_to_torchdataset

# testing if data is being read correctly
@pytest.mark.skipif(
    not (
        os.path.exists("data/processed/train_dataset.pt")
        or os.path.exists("data/processed/test_dataset.pt")
    ),
    reason="Data files not found",
)
def test_is_tokenized():
    # read dataset
    X_train, X_test, y_train, y_test = read_data()
    
    # Tokenizer
    train_encodings, test_encodings = tokenizer(X_train, X_test, y_train, y_test)
    
    # Assert types
    assert hasattr(train_encodings, 'input_ids'), "Tokenized train data does not have input_ids attribute"
    assert hasattr(train_encodings, 'attention_mask'), "Tokenized train does not have attention_mask attribute"
    assert hasattr(test_encodings, 'input_ids'), "Tokenized test data does not have input_ids attribute"
    assert hasattr(test_encodings, 'attention_mask'), "Tokenized test does not have attention_mask attribute"
    
@pytest.mark.skipif(
    not (
        os.path.exists("data/processed/train_dataset.pt")
        or os.path.exists("data/processed/test_dataset.pt")
    ),
    reason="Data files not found",
)
def test_is_converted():
    # read dataset
    X_train, X_test, y_train, y_test = read_data()
    
    # Tokenizer
    train_encodings, test_encodings = tokenizer(X_train, X_test, y_train, y_test)
    
    # Convert
    train_set, test_set = convert_to_torchdataset(train_encodings, test_encodings, y_train, y_test)
    
    # Assert types for all data
    for i in range(len(X_train)):
        assert type(train_set.__getitem__(i)['input_ids']) == torch.Tensor, "Train input_ids data not a tensor"
        assert type(train_set.__getitem__(i)['attention_mask']) == torch.Tensor, "Train attention_mask data not a tensor"
        assert type(train_set.__getitem__(i)['labels']) == torch.Tensor, "Train label data not a tensor"
    
    for i in range(len(X_test)):
        assert type(test_set.__getitem__(i)['input_ids']) == torch.Tensor, "Test input_ids data not a tensor"
        assert type(test_set.__getitem__(i)['attention_mask']) == torch.Tensor, "Test attention_mask data not a tensor"
        assert type(test_set.__getitem__(i)['labels']) == torch.Tensor, "Test label data not a tensor"
        
@pytest.mark.skipif(
    not (
        os.path.exists("data/processed/train_dataset.pt")
        or os.path.exists("data/processed/test_dataset.pt")
    ),
    reason="Data files not found",
)
def test_load_data():
    X_train, X_test, y_train, y_test = read_data()
    
    assert X_train is not None, "X_train not loaded"
    assert X_test is not None, "X_test not loaded"
    assert y_train is not None, "y_train not loaded"
    assert y_test is not None, "y_test not loaded"
    
