#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pathlib
import struct
import pandas as pd
import gc
import os.path
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tabulate import tabulate
import csv

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint


# In[5]:


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


# In[6]:



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


##### Class

class TrainDataset(tf.data.Dataset):

    def _generator(file_id):  
#         print(f'Using Train Class')
        if(file_id == 0):
#             print("reshuffling")
            np.random.shuffle(list_of_file_ids_train)             

        i_file = list_of_file_ids_train[file_id]

#         print(f'file_id: {file_id}, i_file: {i_file}')
#         print()
        signal_filename = direction/f'{i_file+1}.h5'

        
         # Load the labels and signals from the files
        df = pd.read_hdf(signal_filename,key=None)    
        
        labels = df[variable]
        signals = df[df.columns[10:-2]].values
        
        
        # Determine how many batches can be made from this file
        num_batches = len(signals) // batch_size

        # Shuffle the signals within the file
        signal_indices = np.arange(len(signals))
        np.random.shuffle(signal_indices)        
        
        # Loop through each batch in the file
        for batch_idx in range(num_batches):
            # Get the signals and labels for this batch
            batch_signal_indices = signal_indices[batch_idx*batch_size:(batch_idx+1)*batch_size]      
 
            batch_signals = signals[batch_signal_indices]
            
            batch_signals = Processing[process](batch_signals)
                
            batch_signals = batch_signals[:,:,np.newaxis] # Can also be done with signals = signals[:,:,np.newaxis]
            batch_labels = labels.iloc[batch_signal_indices]

            # Yield the signals and labels as a tuple
            yield batch_signals, batch_labels.values 
             
    def __new__(cls, file_id):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((batch_size, 1998,1), (batch_size, )),
            args=(file_id,)
        )

class ValDataset(tf.data.Dataset):

    def _generator(file_id):  
#         print(f'Using Val Class')
        i_file = list_of_file_ids_val[file_id]
    
        signal_filename = direction/f'{i_file+1}.h5'

         # Load the labels and signals from the files
        df = pd.read_hdf(signal_filename,key=None)    
        
        labels = df[variable]
        signals = df[df.columns[10:-2]].values
        
        
        # Determine how many batches can be made from this file
        num_batches = len(signals) // batch_size

        # Shuffle the signals within the file
        signal_indices = np.arange(len(signals))
        np.random.shuffle(signal_indices)        
        
        # Loop through each batch in the file
        for batch_idx in range(num_batches):
            # Get the signals and labels for this batch
            batch_signal_indices = signal_indices[batch_idx*batch_size:(batch_idx+1)*batch_size]      
 
            batch_signals = signals[batch_signal_indices]
            
            batch_signals = Processing[process](batch_signals)
                
            batch_signals = batch_signals[:,:,np.newaxis] # Can also be done with signals = signals[:,:,np.newaxis]
            batch_labels = labels.iloc[batch_signal_indices]

            # Yield the signals and labels as a tuple
            yield batch_signals, batch_labels.values 
             
    def __new__(cls, file_id):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((batch_size, 1998,1), (batch_size, )),
            args=(file_id,)
        )


# In[7]:


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


# In[8]:


# Pre-processing options
Processing = {
    "Unnormalized": Unnormalized,
    "Normalized": Normalized,
    "Denominator": Denominator
}
Process = ["Unnormalized","Normalized","Denominator"]
process = Process[1]
# Loss Function

loss_function = ['mean_absolute_error','mean_squared_error']
lf = 1

# Training Variables
batch_size = 32
num_epochs = 30

steps_per_epoch = eighty*5000 // batch_size

# Learning Rate
initial_lr = 1e-03
final_lr = 1e-06

# initial_lr = 1e-03
# final_lr = 1e-03

def step_decay(epoch):
    lrate = initial_lr * (final_lr/initial_lr)**(epoch/num_epochs)

    print(f'Current Learning rate: {lrate}')
    return lrate

# Compile the model
# model.compile(loss=loss_function[lf], optimizer = keras.optimizers.Adam(initial_lr), metrics=['mean_absolute_error','mean_squared_error'])

# Configuring training dataset
dataset_train = tf.data.Dataset.range(eighty).interleave(
        TrainDataset,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=True).repeat().prefetch(1)



# Configuring training dataset
dataset_val = tf.data.Dataset.range(twenty-num_test_files).interleave(
        ValDataset,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=True).prefetch(1)


# Callback Functions
LRS = tf.keras.callbacks.LearningRateScheduler(step_decay)

ES = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True,verbose=1)

CSV = tf.keras.callbacks.CSVLogger("Location_for_csvlog/Log.csv",
                                separator=",", append=True)


MC = ModelCheckpoint(
    filepath=MC_path,  # Filepath to save the model weights
    monitor='val_loss',  # Quantity to monitor (e.g., validation loss)
    save_best_only=True,  # Save only the best model based on the monitored quantity
    save_weights_only=True  # Save only the model weights, not the entire model
)

callbacks = [MC,LRS,CSV]



# Variable of interest
variable = 'PDT'


# In[9]:


num_runs = 5  # Number of times to run the model

# The following method ensures that the initialization starts random for each training

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")
    
    model = create_model()
    
    model.compile(loss=loss_function[lf], optimizer = keras.optimizers.Adam(initial_lr), metrics=['mean_absolute_error','mean_squared_error'])
    
    MC_path = f"Location_for_model_checkpoint/model_checkpoint_{run + 1}.h5"
    MC = ModelCheckpoint(
    filepath=MC_path,  # Filepath to save the model weights
    monitor='val_loss',  # Quantity to monitor (e.g., validation loss)
    save_best_only=True,  # Save only the best model based on the monitored quantity
    save_weights_only=True  # Save only the model weights, not the entire model
    )

    callbacks = [MC,LRS,CSV]
    
    # Train the model
    history = model.fit(x=dataset_train, validation_data = dataset_val, steps_per_epoch=steps_per_epoch, epochs=num_epochs,callbacks=callbacks)

