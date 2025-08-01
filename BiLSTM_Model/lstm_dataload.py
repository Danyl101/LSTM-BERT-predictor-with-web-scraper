import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import logging
import traceback

try:
    #Importing Normalized data
    train_load=pd.read_csv('Datasets/train_scaled.csv')
    train_data=train_load[['Open','High','Low','Close','Volume']]

    val_load=pd.read_csv('Datasets/val_scaled.csv')
    val_data=val_load[['Open','High','Low','Close','Volume']]

    test_load=pd.read_csv('Datasets/test_scaled.csv')
    test_data=test_load[['Open','High','Low','Close','Volume']]

    # Introduce noise or offset in test
    #test_data["Close"] += np.random.normal(0, 0.5, size=len(test_data))  # small noise

    #Defining constants
    lookback=60
    featurescale=['Open','High','Low','Volume']

except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    raise

#Sequence Handling Class
class TimeSeriesDataset(Dataset):
    #Class calls for argument with constructor
    def __init__(self, data, target_col, lookback,shift_dict):
        #Extracts values from columns and creates sequences
        self.target = data[target_col].values
        self.lookback = lookback
        
        #Copies data for shifting of required columns
        data_copy = data.copy()
        if shift_dict:
            for col, shift_amt in shift_dict.items():
                data_copy[col] = data_copy[col].shift(shift_amt)

        # Drop rows with NaNs introduced by shifting
        data_copy = data_copy.dropna().reset_index(drop=True)
        self.data = data_copy[featurescale].values
        self.target = data_copy[target_col].values

    #Returns how many sequences are there in the dataset
    def __len__(self):
        return len(self.data) - self.lookback

    #Returns the sequence and the target value at the index
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.lookback]
        y = self.target[idx+self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
        
try:
    #Calls the TimeSeriesDataset class to create sequences for each dataset
    train_dataset=TimeSeriesDataset(train_data, 'Close', lookback,shift_dict={'Close':0})
    val_dataset=TimeSeriesDataset(val_data, 'Close', lookback,shift_dict={'Close':0})
    test_dataset=TimeSeriesDataset(test_data, 'Close', lookback,shift_dict={'Close':0})

    # DataLoader to handle batching and shuffling 
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader=DataLoader(test_dataset, batch_size=32)
except Exception as e:
    logging.error(f"Error in DataLoader creation: {e}")
    logging.debug(traceback.format_exc())