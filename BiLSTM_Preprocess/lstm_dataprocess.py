import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


#Importing raw data
data=pd.read_csv('Datasets/nifty_data.csv')
print(data.head())

#Accessing the necessary features 
featurescale=['Close','High','Low','Open','Volume']

data=pd.read_csv('Datasets/nifty_scaled.csv')
print("Data after transformation:")
print(data.head())
#Defining the scaler
lookback=60

#Splitting the data into train, validation, and test sets
total_samples=len(data)-lookback
train_size=int(total_samples*0.7)
val_size=int(total_samples*0.15)
test_size=int(total_samples*0.15)

#Defines Beginning and End of datasets [begin:end]
train_data=data[:train_size+lookback]
val_data=data[train_size:train_size+val_size+lookback]
test_data=data[train_size+val_size:train_size+val_size+test_size+lookback]

#Acquiring the data of features defined above
prescaletrain=train_data[featurescale]
prescaleval=val_data[featurescale]
prescaletest=test_data[featurescale]

#Scaling the data
scaler = RobustScaler()
scaledtrain = scaler.fit_transform(prescaletrain)
finaltrain = pd.DataFrame(scaledtrain, columns=featurescale)

scaledval = scaler.transform(prescaleval)
finalval = pd.DataFrame(scaledval, columns=featurescale)

scaledtest = scaler.transform(prescaletest)
finaltest = pd.DataFrame(scaledtest, columns=featurescale)

# Inserting the date column if it exists in the original data
if 'Date' in train_data.columns:
    finaltrain.insert(0, 'Date', train_data['Date'].reset_index(drop=True))

if 'Date' in val_data.columns:
    finalval.insert(0, 'Date', val_data['Date'].reset_index(drop=True))#Index reset to avoid misalignment

if 'Date' in test_data.columns:
    finaltest.insert(0, 'Date', test_data['Date'].reset_index(drop=True))

# Displaying the first few rows of the scaled data
print(finaltrain.head())
print(finalval.head())
print(finaltest.head())

# Saving the scaled data to a CSV file
finaltrain.to_csv('Datasets/train_scaled.csv', index=False)
finalval.to_csv('Datasets/val_scaled.csv', index=False)
finaltest.to_csv('Datasets/test_scaled.csv', index=False)
