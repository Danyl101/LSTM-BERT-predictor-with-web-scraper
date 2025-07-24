import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error,r2_score,mean_absolute_percentage_error

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

#Sequence Handling Class
class TimeSeriesDataset(Dataset):
    #Class calls for argument with constructor
    def __init__(self, data, target_col, lookback):
        #Extracts values from columns and creates sequences
        self.data = data[featurescale].values
        self.target = data[target_col].values
        self.lookback = lookback

    #Returns how many sequences are there in the dataset
    def __len__(self):
        return len(self.data) - self.lookback

    #Returns the sequence and the target value at the index
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.lookback]
        y = self.target[idx+self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


#Calls the TimeSeriesDataset class to create sequences for each dataset
train_dataset=TimeSeriesDataset(train_data, 'Close', lookback)
val_dataset=TimeSeriesDataset(val_data, 'Close', lookback)
test_dataset=TimeSeriesDataset(test_data, 'Close', lookback)

# DataLoader to handle batching and shuffling 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader=DataLoader(test_dataset, batch_size=32)

# 1. LSTM Model Definition
class BiLSTMModel(nn.Module):
    #Defines the structure of the LSTM model
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=self.bidirectional)#Model is defined
        self.fc = nn.Linear(hidden_size*2, 1)

    #Defines movement of data through the model
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last timestep
        return self.fc(out).squeeze()
    
class TimeWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        batch_size = y_true.shape[0]
        weights = torch.linspace(1, 2, steps=batch_size).to(y_pred.device)
        return torch.mean(weights * (y_pred - y_true) ** 2)
    
def evaluate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true)).mean() * 100

    return mse, rmse, mae, r2, mape

#Check if GPU is available and set device accordingly    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initialize the model, loss function, and optimizer
model = BiLSTMModel(input_size=4).to(device)
criterion = nn.MSELoss()
criterion_train=TimeWeightedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to evaluate the model on validation or test data
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:#Loads arguments from sequences
            x, y = x.to(device), y.to(device) 
            preds = model(x) #Passes the sequences through the model
            loss = criterion(preds, y) #Calculates the loss
            total_loss += loss.item()
    return total_loss / len(loader)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = criterion_train(preds, y)

        optimizer.zero_grad() #Resets accumulated gradients from previous batch
        loss.backward() #Computes gradients of loss w.r.t. model parameters
        optimizer.step() #Updates model weights

        total_train_loss += loss.item()

    val_loss = evaluate(model, val_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
      
def predict(model, loader):
    model.eval()
    preds, targets = [], [] #Defines empty lists to store predictions and targets
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.append(pred) #Appends models predictions to the list
            targets.append(y) #Appends actual values to the list
            
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    mse, rmse, mae, r2, mape = evaluate_metrics(preds, targets)
    return preds,targets,mse,rmse,mae,r2,mape

test_loss = evaluate(model, test_loader)
# Optional: Get predictions and true values for plotting
test_preds, test_actuals, mse, rmse, mae, r2, mape = predict(model, test_loader)

print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")

#Plotting the predictions vs actual values
plt.figure(figsize=(10, 5))
plt.plot(test_preds, label='Predicted')
plt.plot(test_actuals, label='Actual')
plt.title("LSTM Model Predictions vs Actual")
plt.xlabel("Time")
plt.ylabel("Scaled Close Price")
plt.legend()
plt.show()

   
    
        

        
        
    


    



