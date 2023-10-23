#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pathlib
import struct
import pandas as pd
import gc
import os.path
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
from tabulate import tabulate
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


# In[6]:


### Usable Files

# Path File where the Mixed&Sliced signals are stored
direction = pathlib.Path("Location where data is stored")

file_list = list(direction.iterdir())

file_num = len(file_list)


# Diving # of test files
num_test_files = 2

# Locate 80% for training
eighty = round(0.8*file_num)

# Located 80% - #Test files for validation
twenty = file_num-eighty - num_test_files

# Listing the files for each category
list_of_file_ids_train = np.arange(eighty, dtype=int)

list_of_file_ids_val = np.arange(eighty,eighty+twenty-num_test_files, dtype=int)

list_of_file_ids_test =np.arange(file_num-num_test_files,file_num)


# In[7]:



##### Functions to process the data

        ### Unnormalization of each signal individually
def Unnormalized(batch_signals):
        
        return batch_signals
        
        ### Normalization of each signal individually
def Normalized(batch_signals):

        for i in range(len(batch_signals)):
            batch_signals[i] = batch_signals[i]/np.max(batch_signals[i])
            
        return batch_signals
            
        
        ### Normalization of the entire value by one common denominator      
def Denominator(batch_signals):  
    
        denominator = 3953.48
        batch_signals = batch_signals/denominator
        
        return batch_signals


# In[10]:


def create_model():
    model = keras.models.Sequential()
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=(1998, 1)))
    model.add(Conv1D(filters=8, kernel_size=5, dilation_rate=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=4, kernel_size=5, dilation_rate=2, activation='relu'))
    model.add(Conv1D(filters=4, kernel_size=5, strides=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=4, kernel_size=3, strides=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    return model


# In[11]:


# Pre-processing options
Processing = {
    "Unnormalized": Unnormalized,
    "Normalized": Normalized,
    "Denominator": Denominator
}
Process = ["Unnormalized","Normalized","Denominator"]


# In[13]:


MC_path = "Location where the models are stored"
process = Process[1]

# Get the list of model file names
file_models = os.listdir(MC_path)

# Create dictionaries to store metrics for each combination of mass and energy
mae_dict = {}
mse_dict = {}
mean_diff_dict = {}
std_diff_dict = {}

# Loop over all files
for model_file in file_models:
    # Extract the mass and energy values from the model file name
    mass, energy = float(model_file.split("_")[1]), float(model_file.split("_")[2])

    # Load the data from the corresponding file
    # Assuming you want to load the first data file in the list (file_list[0])
    for file in file_list:
        # Change this index according to your requirement
        df = pd.read_hdf(file, key=None)

        # Assuming you have a function 'create_model()' that returns the compiled model
        model = create_model()

        # Load the pre-trained model weights from the file
        model.load_weights(os.path.join(MC_path, model_file))

        # Continue with the rest of the code to make predictions and perform statistics
        mask = (df['A'] == mass) & (df['Z'] == energy)
        df_new = df[mask]
        print(len(df_new))
        # Assuming you have a pre-trained model loaded, let's make predictions and calculate the differences between predicted and true labels.
        labels1 = df_new.iloc[:, 9].values
        labels2 = df_new['ToF'].values
        labels = labels1 + labels2

        signals = df_new[df_new.columns[10:-2]].values
        signals = Processing[process](signals)
        signals = signals[:, :, np.newaxis]

        predicted = model.predict(signals)
        predicted = np.squeeze(predicted)
        true = labels
        print(len(predicted-true))

        if len(predicted-true) == 1:
            # Handle the case where both 'predicted' and 'true' are scalars
            # Calculate a custom error metric, such as absolute difference
            custom_error = abs(predicted - true)
            print("Custom Error (Absolute Difference):", custom_error)

            # You can add additional handling or metrics specific to scalar values here

            # Include the custom error in the overall metrics dictionaries
            key = (mass, energy)
            mae_dict.setdefault(key, []).append(custom_error)
            mse_dict.setdefault(key, []).append(custom_error)
            mean_diff_dict.setdefault(key, []).append(custom_error)
            std_diff_dict.setdefault(key, []).append(custom_error)
        else:
            # Calculate metrics for arrays of values
            mae = mean_absolute_error(true, predicted)
            mse = mean_squared_error(true, predicted)
            mean_diff = np.mean(predicted - true)
            std_diff = np.std(predicted - true)

            # Store metrics in dictionaries based on mass and energy values
            key = (mass, energy)
            mae_dict.setdefault(key, []).append(mae)
            mse_dict.setdefault(key, []).append(mse)
            mean_diff_dict.setdefault(key, []).append(mean_diff)
            std_diff_dict.setdefault(key, []).append(std_diff)


# In[14]:


# Calculate mean and standard deviation of the metrics for each group (combination of mass and energy)
combined_metrics = {}
for key in mae_dict.keys():
    mae_mean = np.mean(mae_dict[key])
    mae_std = np.std(mae_dict[key])
    mse_mean = np.mean(mse_dict[key])
    mse_std = np.std(mse_dict[key])
    mean_diff_mean = np.mean(mean_diff_dict[key])
    mean_diff_std = np.std(mean_diff_dict[key])
    std_diff_mean = np.mean(std_diff_dict[key])
    std_diff_std = np.std(std_diff_dict[key])

    combined_metrics[key] = {
        "MAE": (mae_mean, mae_std),
        "MSE": (mse_mean, mse_std),
        "Mean Difference": (mean_diff_mean, mean_diff_std),
        "Std Difference": (std_diff_mean, std_diff_std)
    }

# Print or use the calculated metrics as required
for key in combined_metrics:
    print(f"Mass={key[0]}, Energy={key[1]}")
    print("MAE: Mean =", combined_metrics[key]["MAE"][0], "Std =", combined_metrics[key]["MAE"][1])
    print("MSE: Mean =", combined_metrics[key]["MSE"][0], "Std =", combined_metrics[key]["MSE"][1])
    print("Mean Difference: Mean =", combined_metrics[key]["Mean Difference"][0], "Std =", combined_metrics[key]["Mean Difference"][1])
    print("Std Difference: Mean =", combined_metrics[key]["Std Difference"][0], "Std =", combined_metrics[key]["Std Difference"][1])


# In[15]:


# Calculate mean and standard deviation of the metrics across all groups
all_mae_values = [value for values in mae_dict.values() for value in values]
all_mse_values = [value for values in mse_dict.values() for value in values]
all_mean_diff_values = [value for values in mean_diff_dict.values() for value in values]
all_std_diff_values = [value for values in std_diff_dict.values() for value in values]

mae_mean_all = np.mean(all_mae_values)
mae_std_all = np.std(all_mae_values)
mse_mean_all = np.mean(all_mse_values)
mse_std_all = np.std(all_mse_values)
mean_diff_mean_all = np.mean(all_mean_diff_values)
mean_diff_std_all = np.std(all_mean_diff_values)
std_diff_mean_all = np.mean(all_std_diff_values)
std_diff_std_all = np.std(all_std_diff_values)

# Create a dictionary for the combined metrics
combined_metrics_all = {
    "MAE": (mae_mean_all, mae_std_all),
    "MSE": (mse_mean_all, mse_std_all),
    "Mean Difference": (mean_diff_mean_all, mean_diff_std_all),
    "Std Difference": (std_diff_mean_all, std_diff_std_all)
}

# Print or use the combined metrics as required
print("Combined Metrics for All Mass and Energy:")
print("MAE: Mean =", combined_metrics_all["MAE"][0], "Std =", combined_metrics_all["MAE"][1])
print("MSE: Mean =", combined_metrics_all["MSE"][0], "Std =", combined_metrics_all["MSE"][1])
print("Mean Difference: Mean =", combined_metrics_all["Mean Difference"][0], "Std =", combined_metrics_all["Mean Difference"][1])
print("Std Difference: Mean =", combined_metrics_all["Std Difference"][0], "Std =", combined_metrics_all["Std Difference"][1])


# In[16]:


# Create a dictionary for the combined metrics
combined_metrics_all = {
    "MAE": (mae_mean_all, mae_std_all),
    "MSE": (mse_mean_all, mse_std_all),
    "Mean Difference": (mean_diff_mean_all, mean_diff_std_all),
    "Std Difference": (std_diff_mean_all, std_diff_std_all)
}

# Print or use the combined metrics as required
print("Combined Metrics for All Mass and Energy:")
for metric_name, (mean_value, std_value) in combined_metrics_all.items():
    mean_value = float(mean_value)
    std_value = float(std_value)
    print(f"{metric_name}: {mean_value:.3f} \u00B1 {std_value:.3f}")

