import struct
import numpy as np
from pathlib import Path
import pandas as pd
import math

#%%

def main():
    print("Slicing and Mixing the hdf files")

if __name__ == '__main__':
    main()

#%%

# Path where the hdf are  store new data
path = r"Where the hdf files should be stored"

# Path where the mixed and sliced hdf files should be stored
location = r"Where the hdf files should be stored"

#%%

### Distribute all hdf files into variable(number_of_cuts) files evenly
# In this code we distribute all signals equally over x amount of files
# This is required for how the Neural Network is trained on the data

files = list(path.iterdir())

Range = int(len(files))

def cutting(number_of_cuts):
    
    number = 1/number_of_cuts
    arange = np.arange(0,1+number,number)
    
    
    for i in range(Range):
        print(i)
        reading_files = pd.read_hdf(files[i], key=None, mode='r')
        
        array = arange*len(reading_files)
        array = np.round(array)
        
        
        for j in range(number_of_cuts):
            
            df_file = reading_files[int(array[j]):int(array[j+1])]
            
            if i == 0: # Create the file
                Location = Path(f'{location}/{j+1}.h5')
                df_file.to_hdf(Location, key='data.df', mode='w', format='table') 
        

            else: # Add to file
                Location = Path(f'{location}/{j+1}.h5')
                df_file.to_hdf(Location, key='data.df', format='table', append=True) 

    
cutting(40)
