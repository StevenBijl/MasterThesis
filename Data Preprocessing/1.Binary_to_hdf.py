import numpy as np
from pathlib import Path
import pandas as pd
import math
import struct

#%%

def main():
    print("From .bin to hdf")

if __name__ == '__main__':
    main()


#%% 
# Path to where the detector signals are stored as binary data
binary_data = Path(r"C:\Users\steve\OneDrive\Bureaublad\Binaries_with_MCP_time")
files = list(binary_data.iterdir())

# Path where you want to store new data
path = r"C:\Users\steve\OneDrive\Bureaublad\new"

#%% Creating hdf files

### The binary files are structured such that the first 72 bits are the constant variables
# 1 flight path distance
# 2 flight path distance uncertainty
# 3 Mass Number
# 4 Proton Number
# 5 Energy
# 6 Time of Flight
# 7 Time of Flight Uncertainty
# 8 Plasma Delay Time

### The remaining data can be split into 2001 * 8 bits
# The first 8 bits are the MCP time
# The remaining 2000 * 8 bits are the samples of the PIPS detector

n = 0
for file in files:
    # Create list for the first 8 variables
    fixed_variables = []
    
    # Read the file
    with open(file, 'br') as f:
        buffer = f.read()
        f.close()

    # Find the size
    size = str(int(len(buffer) / 8)) + 'd'

    # Unpack the data
    data = struct.unpack(size, buffer)

    # test the heading
    fixed_variables.append(data[0:9])

    # Find total number of signals on each file
    number_of_signals = int((len(buffer) - 9 * 8) / (8 * 2001))
    signals = np.array([data[i * 2001 + 9:(i + 1) * 2001 + 9] for i in range(number_of_signals)])

    Flight_Path = np.full(signals.shape[0], data[0])
    Flight_Path_U = np.full(signals.shape[0], data[1])
    Mass_Number = np.full(signals.shape[0], data[2])
    Proton_Number = np.full(signals.shape[0], data[3])
    four = np.full(signals.shape[0], data[4])
    Energy_U = np.full(signals.shape[0], data[5])
    ToF = np.full(signals.shape[0], data[6])
    ToF_U = np.full(signals.shape[0], data[7])
    PDT = np.full(signals.shape[0], data[8]) - 10.531742530532568  # - Delay Due to Machinery

    labels = {'Flight_Path': Flight_Path, 'Flight_Path_U': Flight_Path_U, 'A': Mass_Number, 'Z': Proton_Number, 'E': four,
              'E_U': Energy_U, 'ToF': ToF, 'ToF_U': ToF_U, 'PDT': PDT}
    df1 = pd.DataFrame(labels)
    df2 = pd.DataFrame(signals)
    df = pd.concat((df1, df2), axis=1)

    # Filter out faulty data based on threshold
    filtered_indices = np.where(df.iloc[:, 9] <= 510)[0]
    df_filtered = df.iloc[filtered_indices]

    # Reset the index of the filtered DataFrame
    df_filtered = df_filtered.reset_index(drop=True)

    # Save the filtered DataFrame to a new file
    path_label = Path(f'{path}/M{math.floor(data[2])}_E{math.floor(data[4])}.df')
    df_filtered.to_hdf(path_label, key='data.df', mode='w')
