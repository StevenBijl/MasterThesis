#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


# Pre-processing options
Processing = {
    "Unnormalized": Unnormalized,
    "Normalized": Normalized,
    "Denominator": Denominator
}
Process = ["Unnormalized","Normalized","Denominator"]


# In[6]:


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


# In[7]:


model = create_model()


# In[9]:


# Initialize lists to store the metrics for each model
mae_per_model = []
mse_per_model = []
mean_per_model = []
std_per_model = []

# Process the data and calculate metrics for each model
for run in range(len(file_models)):
    print(f"Run {run + 1}/{len(file_models)}")
    
    # Load the model weights
    weights_path = os.path.join(MC_path, file_models[run])
    model = create_model()
    model.load_weights(weights_path)
    
    # Extract labels and signals for the i-th model
    labels1 = df_test.iloc[:, 9].values
    labels2 = df_test['ToF'].values
    labels = labels1 + labels2

    signals = df_test[df_test.columns[10:-2]].values
    signals = Processing[process](signals)
    signals = signals[:, :, np.newaxis]

    predicted = model.predict(signals)
    predicted = np.squeeze(predicted)
    true = labels

    difference = predicted - true

    # Calculate metrics for the i-th model
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    mean = np.mean(predicted-true)
    std = np.std(predicted-true)

    # Append metrics to the respective lists
    mae_per_model.append(mae)
    mse_per_model.append(mse)
    mean_per_model.append(mean)
    std_per_model.append(std)

    # Print metrics for the i-th model
    print(f"Metrics for Model {run+1}:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("Mean:", mean)
    print("STD:", std)
    print()

# Calculate metrics for all models combined
combined_mae = np.mean(mae_per_model)
combined_mse = np.mean(mse_per_model)
combined_mean = np.mean(mean_per_model)
combined_std = np.mean(std_per_model)

# Calculate errors for combined metrics
combined_mae_error = np.std(mae_per_model)
combined_mse_error = np.std(mse_per_model)
combined_mean_error = np.std(mean_per_model)
combined_std_error = np.std(std_per_model)

# Print metrics for all models combined
print("Metrics for All Models Combined:")
print(f"Combined MAE: {combined_mae:.3f} \u00B1 {combined_mae_error:.3f}")
print(f"Combined MSE: {combined_mse:.3f} \u00B1 {combined_mse_error:.3f}")
print(f"Combined Mean: {combined_mean:.3f} \u00B1 {combined_mean_error:.3f}")
print(f"Combined STD: {combined_std:.3f} \u00B1 {combined_std_error:.3f}")


# In[10]:


df_test = pd.concat([pd.read_hdf(file_list[i], key=None) for i in list_of_file_ids_test])
unique_mass_test = df_test['A'].unique()

# Initialize lists to store the metrics for each unique combination of mass and energy
metrics_per_combination = []

# Test Set
for mass_test in unique_mass_test:
    print(f"Unique Mass: {mass_test}")
    df_intermediate = df_test[df_test['A'] == mass_test]
    unique_energy = df_intermediate['Z'].unique()

    for energy in unique_energy:
        print(f"Unique Energy: {energy}")

        # Initialize lists to store the metrics for the current combination of mass and energy
        mae_per_model_combination = []
        mse_per_model_combination = []
        mean_per_model_combination = []
        std_per_model_combination = []
        num_signals = 0

        for run in range(len(file_models)):
            # Load the model weights
            weights_path = os.path.join(MC_path, file_models[run])
            model = create_model()
            model.load_weights(weights_path)

            # Extract labels and signals for the i-th model
            labels1 = df_intermediate[df_intermediate['Z'] == energy].iloc[:, 9].values
            labels2 = df_intermediate[df_intermediate['Z'] == energy]['ToF'].values
            labels = labels1 + labels2

            signals = df_intermediate[df_intermediate['Z'] == energy][df_intermediate.columns[10:-2]].values
            signals = Processing[process](signals)
            signals = signals[:, :, np.newaxis]

            predicted = model.predict(signals)
            predicted = np.squeeze(predicted)
            true = labels

            difference = predicted - true

            # Calculate metrics for the i-th model and the current combination of mass and energy
            mae = mean_absolute_error(true, predicted)
            mse = mean_squared_error(true, predicted)
            mean = np.mean(predicted - true)
            std = np.std(predicted - true)

            # Append metrics to the respective lists
            mae_per_model_combination.append(mae)
            mse_per_model_combination.append(mse)
            mean_per_model_combination.append(mean)
            std_per_model_combination.append(std)
            num_signals = len(signals)

        # Calculate metrics for the current combination of mass and energy
        combined_mae = np.mean(mae_per_model_combination)
        combined_mse = np.mean(mse_per_model_combination)
        combined_mean = np.mean(mean_per_model_combination)
        combined_std = np.mean(std_per_model_combination)

        # Calculate errors for combined metrics
        combined_mae_error = np.std(mae_per_model_combination)
        combined_mse_error = np.std(mse_per_model_combination)
        combined_mean_error = np.std(mean_per_model_combination)
        combined_std_error = np.std(std_per_model_combination)

        # Append metrics for the current combination of mass and energy to the main list
        metrics_per_combination.append((mass_test, energy, combined_mae, combined_mae_error, combined_mse, combined_mse_error, combined_mean, combined_mean_error, combined_std, combined_std_error, num_signals))

# Print metrics for all unique combinations of mass and energy
print("Metrics for All Unique Combinations of Mass and Energy:")
for combination in metrics_per_combination:
    mass_test, energy, combined_mae, combined_mae_error, combined_mse, combined_mse_error, combined_mean, combined_mean_error, combined_std, combined_std_error, num_signals = combination
    print(f"Mass: {mass_test}, Energy: {energy}")
    print(f"Combined MAE: {combined_mae} ± {combined_mae_error}")
    print(f"Combined MSE: {combined_mse} ± {combined_mse_error}")
    print(f"Combined Mean: {combined_mean} ± {combined_mean_error}")
    print(f"Combined STD: {combined_std} ± {combined_std_error}")
    print(f"Number of signals: {num_signals}")
    print()


# In[11]:


# Initialize lists to store the values for scatter plots
mass_list = []
energy_list = []
mean_loss_list = []
std_loss_list = []

# Extract the mass, energy, mean, std, and num_signals values from the metrics_per_combination list
for combination in metrics_per_combination:
    mass_test, energy, combined_mae, combined_mae_error, combined_mse, combined_mse_error, combined_mean, combined_mean_error, combined_std, combined_std_error, num_signals = combination
    
    # Append the values to the respective lists
    mass_list.append(mass_test)
    energy_list.append(energy)
    mean_loss_list.append(combined_mean)
    std_loss_list.append(combined_std)

# Scatter plot for Mean Test Loss
plt.figure()
plt.scatter(mass_list, energy_list, c=mean_loss_list, cmap='coolwarm')
plt.colorbar(label='Mean Test Loss [ns]')
plt.xlabel('Mass Number')
plt.ylabel('Energy [MeV]')
plt.title('Scatter Plot: Mean Test Loss [ns]')
plt.show()

# Scatter plot for Standard Deviation Test Loss
plt.figure()
plt.scatter(mass_list, energy_list, c=std_loss_list, cmap='coolwarm')
plt.colorbar(label='Standard Deviation Test Loss [ns]')
plt.xlabel('Mass Number')
plt.ylabel('Energy [MeV]')
plt.title('Scatter Plot: Standard Deviation Test Loss [ns]')
plt.show()


# In[12]:


# Initialize lists to store the values for scatter plots
mass_list = []
energy_list = []
mean_loss_list = []
std_loss_list = []

# Extract the mass, energy, mean, std, and num_signals values from the metrics_per_combination list
for combination in metrics_per_combination:
    mass_test, energy, combined_mae, combined_mae_error, combined_mse, combined_mse_error, combined_mean, combined_mean_error, combined_std, combined_std_error, num_signals = combination
    
    # Append the values to the respective lists
    mass_list.append(mass_test)
    energy_list.append(energy)
    mean_loss_list.append(combined_mean)
    std_loss_list.append(combined_std)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Scatter plot for Mean Test Loss
scatter1 = ax1.scatter(mass_list, energy_list, c=mean_loss_list, cmap='coolwarm')
ax1.set_xlabel('Mass Number')
ax1.set_ylabel('Energy [MeV]')
ax1.set_title('Scatter Plot: Mean Test Loss [ns]')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Mean Test Loss [ns]')

# Scatter plot for Standard Deviation Test Loss
scatter2 = ax2.scatter(mass_list, energy_list, c=std_loss_list, cmap='coolwarm')
ax2.set_xlabel('Mass Number')
ax2.set_ylabel('Energy [MeV]')
ax2.set_title('Scatter Plot: Standard Deviation Test Loss')
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Standard Deviation Test Loss [ns]')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()


# In[13]:


# Initialize lists to store the values for scatter plots
mass_list = []
energy_list = []
mean_loss_list = []
std_loss_list = []

# Extract the mass, energy, mean, std, and num_signals values from the metrics_per_combination list
for combination in metrics_per_combination:
    mass_test, energy, combined_mae, combined_mae_error, combined_mse, combined_mse_error, combined_mean, combined_mean_error, combined_std, combined_std_error, num_signals = combination
    
    # Append the values to the respective lists
    mass_list.append(mass_test)
    energy_list.append(energy)
    mean_loss_list.append(combined_mean)
    std_loss_list.append(combined_std)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Scatter plot for Mean Test Loss
scatter1 = ax1.scatter(mass_list, energy_list, c=mean_loss_list, cmap='coolwarm',norm=colors.LogNorm())
ax1.set_xlabel('Mass Number')
ax1.set_ylabel('Energy [MeV]')
ax1.set_title('Scatter Plot: Mean Test Loss')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Mean Test Loss [ns]')

# Scatter plot for Standard Deviation Test Loss
scatter2 = ax2.scatter(mass_list, energy_list, c=std_loss_list, cmap='coolwarm',norm=colors.LogNorm())
ax2.set_xlabel('Mass Number')
ax2.set_ylabel('Energy [MeV]')
ax2.set_title('Scatter Plot: Standard Deviation Test Loss ')
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Standard Deviation Test Loss [ns]')

# Create a custom colormap with blue as the color for 'bad' values
my_cmap = copy.copy(plt.cm.get_cmap('coolwarm'))
my_cmap.set_bad(my_cmap(0))

# Apply the custom colormap to the scatter plots
scatter1.set_cmap(my_cmap)
scatter2.set_cmap(my_cmap)

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()


# In[14]:


df_test = pd.concat([pd.read_hdf(file_list[i], key=None) for i in list_of_file_ids_test])

# Get the number of files in the path
file_models = os.listdir(MC_path)
num_files = len(file_models)

# Get the number of runs
num_runs = 5

# Initialize a list to store the differences between predicted and true values
differences = []

# Process the data and calculate differences for each model
for run in range(len(file_models)):
    print(f"Run {run + 1}/{len(file_models)}")
    
    # Load the model weights
    weights_path = os.path.join(MC_path, file_models[run])
    model = create_model()
    model.load_weights(weights_path)
    
    # Extract labels and signals for the i-th model
    labels1 = df_test.iloc[:, 9].values
    labels2 = df_test['ToF'].values
    labels = labels1 + labels2

    signals = df_test[df_test.columns[10:-2]].values
    signals = Processing[process](signals)
    signals = signals[:, :, np.newaxis]

    predicted = model.predict(signals)
    predicted = np.squeeze(predicted)
    true = labels

    difference = predicted - true
    
    # Append the differences to the list
    differences.extend(difference)
    
    
# Calculate mean and standard deviation of the differences
mean_difference = np.mean(differences)
std_difference = np.std(differences)

# Create a histogram of the differences
plt.hist(differences, bins=50, edgecolor='black')
plt.xlabel('Difference (Predicted - True) [ns]')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted - True Differences')

# Add text box with mean and standard deviation
text_box = f"Mean: {mean_difference:.2f}\nStd: {std_difference:.2f}"
plt.text(0.75, 0.95, text_box, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.show()


# In[15]:


df_test = pd.concat([pd.read_hdf(file_list[i], key=None) for i in list_of_file_ids_test])

# Get the number of files in the path
file_models = os.listdir(MC_path)
num_files = len(file_models)

# Get the number of runs
num_runs = 5

# Create a figure with a 2x3 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# Process the data and plot histograms for each model
for run, ax in zip(range(len(file_models)), axes.flatten()[:-1]):
    print(f"Run {run + 1}/{len(file_models)}")
    
    # Load the model weights
    weights_path = os.path.join(MC_path, file_models[run])
    model = create_model()
    model.load_weights(weights_path)
    
    # Extract labels and signals for the i-th model
    labels1 = df_test.iloc[:, 9].values
    labels2 = df_test['ToF'].values
    labels = labels1 + labels2

    signals = df_test[df_test.columns[10:-2]].values
    signals = Processing[process](signals)
    signals = signals[:, :, np.newaxis]

    predicted = model.predict(signals)
    predicted = np.squeeze(predicted)
    true = labels

    difference = predicted - true
    
    # Create a histogram for the differences
    ax.hist(difference, bins=200, edgecolor='black')
    ax.set_xlabel('Difference (Predicted - True) [ns]')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Model {run + 1}')
    
    # Calculate and display the mean and standard deviation
    mean = np.mean(difference)
    std = np.std(difference)
    ax.text(0.05, 0.9, f'Mean: {mean:.2f}\nStd: {std:.2f}', transform=ax.transAxes)

# Remove the empty subplot in the second row
fig.delaxes(axes[1, -1])

# Adjust the spacing between subplots
plt.tight_layout()

plt.show()


# In[16]:


df_test = pd.concat([pd.read_hdf(file_list[i], key=None) for i in list_of_file_ids_test])

# Get the number of files in the path
file_models = os.listdir(MC_path)
num_files = len(file_models)

# Get the number of runs
num_runs = 5

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Process the data and plot overlaid histograms for each model
for run in range(len(file_models)):
    print(f"Run {run + 1}/{len(file_models)}")
    
    # Load the model weights
    weights_path = os.path.join(MC_path, file_models[run])
    model = create_model()
    model.load_weights(weights_path)
    
    # Extract labels and signals for the i-th model
    labels1 = df_test.iloc[:, 9].values
    labels2 = df_test['ToF'].values
    labels = labels1 + labels2

    signals = df_test[df_test.columns[10:]].values
    signals = Processing[process](signals)
    signals = signals[:, :, np.newaxis]

    predicted = model.predict(signals)
    predicted = np.squeeze(predicted)
    true = labels

    difference = predicted - true
    
    # Plot the histogram of differences for the current model
    ax.hist(difference, bins=200, alpha=0.5, label=f'Model {run + 1}')

ax.set_xlabel('Difference (Predicted - True) [ns]')
ax.set_ylabel('Frequency')
ax.set_title('Overlaid Histograms of Model Differences')
ax.legend()

plt.show()


# In[17]:


# Create a histogram of the differences
plt.hist(differences, bins=200, edgecolor='black')
plt.xlabel('Difference (Predicted - True)')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted - True Differences')

# Add text box with mean and standard deviation
text_box = f"Mean: {mean_difference:.2f}\nStd: {std_difference:.2f}"
plt.text(0.75, 0.95, text_box, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
plt.show()


# In[18]:


# Create a histogram of the differences
plt.hist(differences, bins=200, edgecolor='black')
plt.xlabel('Difference (Predicted - True)')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted - True Differences')
plt.yscale('log')

# Add text box with mean and standard deviation
text_box = f"Mean: {mean_difference:.2f}\nStd: {std_difference:.2f}"
plt.text(0.75, 0.95, text_box, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
plt.show()


# In[19]:


# Group the data by unique mass and energy
grouped_data = df_test.groupby(['A'])

# Create an empty list to store the energies for each unique mass
energies_per_mass = []

# Iterate over each unique mass group
for mass, group in grouped_data:
    energies = group['Z'].unique()  # Get the unique energies for the current mass
    energies_per_mass.append((mass, energies))  # Append the unique mass and energies tuple to the list

# Create a list to store the average test losses for each unique mass
avg_test_losses = []

# Create an empty list to store the differences for each unique mass
differences_per_mass = []

# Iterate over the unique masses
for i, (mass, energies) in enumerate(energies_per_mass, 1):
    # Create a list to store the differences for the current mass
    differences = []

    # Iterate over the energies for the current mass
    for energy in energies:
        # Retrieve the data for the current mass and energy
        data = df_test[(df_test['A'] == mass) & (df_test['Z'] == energy)]
        
        # Extract the features (signals) and labels (true values)
        signals = data[data.columns[10:-2]].values
        signals = Processing[process](signals)
        signals = signals[:, :, np.newaxis]
        labels1 = data.iloc[:, 9].values
        labels2 = data['ToF'].values
        labels = labels1 + labels2

        # Predict the labels using the model
        predicted = model.predict(signals)
        predicted = np.squeeze(predicted)

        # Calculate the difference between predicted and true labels
        difference = predicted - labels
        differences.append(difference)  # Store the difference for the current energy

    differences_per_mass.append(differences)  # Store the differences for the current mass

    # Calculate the average test loss for the current mass
    avg_test_loss = np.mean(np.abs(np.concatenate(differences)))  # Concatenate the differences for all energies
    avg_test_losses.append(avg_test_loss)  # Store the average test loss

# Iterate over the unique masses and corresponding differences
for i, (mass, differences) in enumerate(zip(energies_per_mass, differences_per_mass), 1):
    energies = mass[1]  # Get the energies for the current mass

    # Sort the energies in ascending order
    sorted_energies = np.sort(energies)

    # Get the indices to reorder the differences based on the sorted energies
    indices = np.argsort(energies)
    reordered_differences = [differences[index] for index in indices]

    plt.figure(figsize=(150, 6))  # Adjust the figure size as needed
    plt.subplot(1, len(energies_per_mass), i)  # Create a subplot for each unique mass
    plt.boxplot(reordered_differences)
    plt.title(f"Unique Mass: {mass[0]}")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("Difference (Predicted - True) [ns]")

    # Set the x-axis tick labels to the sorted energies
    plt.xticks(range(1, len(sorted_energies) + 1), sorted_energies)

    # Add a text box with counts for each energy
    counts = [len(d) for d in reordered_differences]
    text_box = "\n".join([f"E{e}: {count}" for e, count in zip(sorted_energies, counts)])
    plt.text(0.95, 0.95, text_box, transform=plt.gca().transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()

