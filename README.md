# Code for: Human mobility is well described by closed-form gravity-like models learned automatically from data

rguimera-machine-scientist/ - Contains the code of the Bayesian Machine Scientist.

Data/ - Contains scripts for data scrapping and datasets used to train and test models.

Data/checkpoints/ - Contains partial saved results and models such as BMS models and real-predicted values.

model_sampling_BMS.ipynb - Reads the train dataset, runs the BMS and saves the BMS Plausible and the BMS Ensemble models.

model_validation.ipynb - Uses the train dataset to fit the models and evaluates the test dataset for all models and sates.

plots.ipynb - Read the real-predicted values generated previously and plots the results.

validation_second_week.ipynb Uses the real values of the second week to analyse the predicted values with the first week. Validation test for temporal changes.

fairness_PDP.ipynb - Analyses the fairness of the models using  the Partial Demographic Parity (PDP) metric

radiation_models.ipynb - Predictions of different modifications of the Radiation model.

DeepGravity/ - Folder that contains the code version of the Deep Gravity model used in this project. It also contains the datasets and the predicted values.
