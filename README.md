ML_Ops_ExamProject
==============================

This is the exam project of group 20 for Machine Learning Operation at DTU.

The project will be about categorizing tweets, whether they are about real disasters or not.
It is inspired by the Kaggle competition https://www.kaggle.com/c/nlp-getting-started.

We will use the transformers framework in order to use the state-of-the-art model GPT-2 to categorize
the competition tweets. It is a network that has been trained in an unsupervised manner,
which we will adapt to our needs with a supervised finetuning using the data provided by Kaggle.

As mentioned, data is provided by Kaggle. It has 3 types of information

1) a *text* of a tweet
2) a *keyword* from that tweet (possibly blank)
3) the *location* the tweet was sent from (possibly blank)

We are predict whether the tweet is about a real disaster *1* or not *0*.
It consists of 3 files; **train.csv**, **test.csv**, and **sample_submission.csv**

The data has 5 columns.
1) *id* - a unique identifier for each tweet
2) *text* - the text of the tweet
3) *location* - the location the tweet was sent from (may be blank)
4) *keyword* - a particular keyword from the tweet (may be blank)
5) *target* - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

As mentioned we will be using the state-of-the-art deep learning model GPT-2 for the classification.

The overall goal of the project is to apply the Transformers framework to our chosen
Kaggle competition. Furthermore we seek to apply the cookiecutter structure, while also applying
our newly found knowledge about model configurations using Hydra, Data Version Controlling (DVC),
GitHub and so forth.
Furthermore we seek to be able to be pep8 compliant while doing the project in order to follow
standard coding guidelines.

Your Sincerely,
Machine Learning Operations Team 20, DTU
David Ribberholt Ipsen, S164522
Nicolai Weisbjerg, S174466
Frederik Hartmann, S174471

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
