							      DATA MANIPULATION DEBUG LOG


Tags:[Pandas,Index,Mismatch,Dataframe,Numpy,Standardization]
________________________________________________________________________________________
Error  [Pandas]

pandas.errors.InvalidIndexError: (60, [1])

(data[i ,[1]])

Reason & Solution

Pandas don't allow tuple indexing ([1]) , either convert dataframe to numpy or use iloc

(data.iloc[i, 1])
_________________________________________________________________________________________
Warning	[Pandas] [Index]

 Series.__getitem__ treating keys as positions is deprecated.

 x.append(data.[i-lookback:i])

 Reason & Solution

 data[index] could return series or multiple values 

 x.append(data.iloc[i-lookback:i]) #Returns single locations
____________________________________________________________________________________________
Error [Mismatch] [Pandas]
 
 ValueError: expected sequence of length 5 at dim 1 (got 60)

 x.append(data[i-lookback:i].T)

Reason & Solution

 Transpose (T) caused a mismatch in the input shapes thus instead of 5 inputs (5,60) it got (60,5)

 x.append(data[i-lookback:i])
________________________________________________________________________________________________
Error [Dataframe] [Numpy]

 ValueError: could not determine the shape of object type 'DataFrame'

 x.append(data[i-lookback:i])

Reason and Solution

 Pytorch cannot convert dataframe datatype into a tensor ,it has to be converted into a numpy array

 x.append(data[i-lookback:i].values) 

 Only required for X since is has 2D values (5 features,60 data values)

 Y is scalar only 1D values (1 data point)
_______________________________________________________________________________________________
Error [Standardization]

scaledval=scaler.fit_transform(scaledval)
scaledtest=scaler.fit_transform(scaledtest)

Reason and Solution

Scaler leaks training info onto validation and test dataset thus illegitimizing the model itself as test and validation now contains output

scaler = StandardScaler()
scaledtrain = scaler.fit_transform(scaledtrain)     
scaledval = scaler.transform(scaledval)            
scaledtest = scaler.transform(scaledtest)
___________________________________________________________________________________________________
Error [Pandas] [Index]

Date     Close      High       Low      Open    Volume
0  2013-01-21 -1.425454 -1.436820 -1.417166 -1.425363 -0.781241
1  2013-01-22 -1.439776 -1.433888 -1.427650 -1.427732 -0.790007
2  2013-01-23 -1.437318 -1.447174 -1.435896 -1.439283 -0.753097
3  2013-01-24 -1.452127 -1.449072 -1.441564 -1.442096 -0.530716
4  2013-01-25 -1.428695 -1.442640 -1.438751 -1.451277 -0.704192

  Date     Close      High       Low      Open    Volume
0  NaN -2.139165 -2.167102 -2.200176 -2.232370 -2.912294
1  NaN -2.085786 -2.065502 -2.033640 -2.078285  2.023973
2  NaN -2.123761 -2.129471 -2.068114 -2.071510  1.921373
3  NaN -2.634693 -2.388409 -2.590284 -2.301863  3.322284
4  NaN -2.445549 -2.508750 -2.564805 -2.572626  2.889833

if 'Date' in train_data.columns:
    finaltrain.insert(0, 'Date', train_data['Date'])

if 'Date' in val_data.columns:
    finalval.insert(0, 'Date', val_data['Date'])

Reason and Solution

The second table dosent have values because both tables were taken from same dataset , but since the date of train has correct indexing [0] as it starts 

from index 0 ,the dates of train have been extracted ,while val whose first date index is not 0 was not taken due to val dates index starting from 900

________________________________________________________________________________________

Error [Standardization]
for col in featurescale:
    data[col] = np.log(data[col]) - np.log(data[col].shift(1))  # or: np.log(data[col] / data[col].shift(1))

# Drop the first row (NaN due to shift)
data.dropna(inplace=True)

ValueError: Input X contains infinity or a value too large for dtype('float64').

Reason and Solution

The log of negative numbers are undefined and for 0 is infinity thus causing issues storing the data as it would either be undefined or too large

for col in featurescale:
    data[col] = data[col].apply(lambda x: x if x > 0 else 1e-8)  
    data[col] = np.log(data[col]) - np.log(data[col].shift(1))

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True) 

Replace negative and 0 with preset values


___________________________________________________________________________________________

Note [Standardization]

for col in cols_to_log:
    data[col] = data[col].apply(lambda x: np.log(x) if x > 0 else x)

Apply log transformation before scaling as log transformation closes the gap between large and small values ie 20000 will be closer to 5000 after transformation , since there are no negative values here else just contains x , if not add a placeholder or random value 


_____________________________________________________________________________________________
