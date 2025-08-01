import torch
import torch.nn as nn
import optuna
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import psutil
import os
import logging
import time
import traceback
import gc
import numbers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error,r2_score,mean_absolute_percentage_error

try:
    logging.basicConfig(
        filename="Logs/Optuna.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",
    )
    
    def log_cpu_memory(tag=""):
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 ** 2  # in MB
        print(f"[{tag}] CPU RAM Usage: {mem_mb:.2f} MB")

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
        try:
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
        except Exception as e:
            logging.error(f"Error in TimeSeriesDataset: {e}")
            logging.debug(traceback.format_exc())
            
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
        
    # 1. LSTM Model Definition
    class BiLSTMModel(nn.Module):
        try:
        #Defines the structure of the LSTM model
            def __init__(self, input_size=4, hidden_size=64, num_layers=2,dropout=0.3, batch_size=32):
                super().__init__()
                self.hidden_size=hidden_size
                self.batch_size=batch_size
                self.num_layers=num_layers
                self.dropout=dropout
                self.bidirectional=True
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=self.bidirectional)#Model is defined
                self.dropout=nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size*2, 1)

            #Defines movement of data through the model
            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]  # Take the last timestep
                out = self.dropout(out)
                return self.fc(out).view(-1) #Removes last dimension from output tensor
        except Exception as e:
            logging.error(f"Error in BiLSTMModel: {e}")
            logging.debug(traceback.format_exc())

    #Bayesian Optimization
    def objective(trial):
        try:
            logging.info(f"Starting trial {trial.number}")
            start_time= time.time()
            hidden_size = trial.suggest_categorical("hidden_size", [32, 64])
            dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5])
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32])
            
            logging.info(f"Trial #{trial.number} Start Time:{start_time}  Hyperparameters: "
                          f"lr={learning_rate}, dropout={dropout}, hidden_size={hidden_size}, batch_size={batch_size}")

            train_loader = DataLoader(train_dataset,batch_size=batch_size,num_workers=0)
            test_loader = DataLoader(test_dataset,batch_size=batch_size,num_workers=0)
            
            log_cpu_memory("Before Trial")

            model = BiLSTMModel(input_size=4, hidden_size=hidden_size, dropout=dropout).to(device) #Defines model for bayesian
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Defines optimizer

            epoch = 15 #No of epochs
            # Training loop (light version)
            for epoch in range(epoch):
                model.train()
                for xb, yb in train_loader: #Loads input features and target features
                    xb, yb = xb.to(device), yb.to(device)               
                    optimizer.zero_grad() #Resets optimizer back to zero gradient after every epoch
                    output = model(xb)
                    loss = criterion(output, yb) #Checks models output with the actual output
                    loss.backward() #Propagates backwards and finds gradients of loss 
                    optimizer.step() #Updates the weights

            # Evaluate
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    preds.append(pred) #Appends models predictions to the list
                    truths.append(y) #Appends actual values to the list
            
            log_cpu_memory("After Trial")

            preds = torch.cat(preds, dim=0) #Converts datatype back for metrics evaluation
            truths = torch.cat(truths, dim=0)

            mse, _, _, _, _ = evaluate_metrics(truths, preds)
            del model
            del optimizer
            torch.cuda.empty_cache()  # no GPU, but still clears PyTorch cache
            gc.collect()

            return mse
        

        except Exception as e:
            logging.error(f"Issue occured at trial {trial.number} {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return float("inf")  # Return a high value to indicate failure

    #Custom Loss function
    class TimeWeightedLoss(nn.Module):
        try:
            def __init__(self):
                super().__init__()

            def forward(self, y_pred, y_true):
                batch_size = y_true.shape[0]  
                weights = torch.linspace(1, 2, steps=batch_size).to(y_pred.device)
                return torch.mean(weights * (y_pred - y_true) ** 2)
        except Exception as e:
            logging.error(f"Error in TimeWeightedLoss: {e}")
            logging.debug(traceback.format_exc())
        
    #Function for metrics evaluation
    def evaluate_metrics(y_true, y_pred):
        try:
            y_true = y_true.detach().cpu().numpy() #Takes the actual values
            y_pred = y_pred.detach().cpu().numpy() #Takes the predicted values

            mse = mean_squared_error(y_true, y_pred)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = (np.abs((y_true - y_pred) / y_true)).mean() * 100
        except Exception as e:
            logging.error(f"Error in evaluate_metrics: {e}")
            logging.debug(traceback.format_exc())
            return None, None, None, None, None

        return mse, rmse, mae, r2, mape #Returns various metrics

    #Check if GPU is available and set device accordingly
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Initialize the model, loss function, and optimizer
        model = BiLSTMModel(input_size=4).to(device)
        criterion = nn.MSELoss()
        criterion_train=TimeWeightedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    except Exception as e:
        logging.error(f"Error in device setup or model initialization: {e}")
        logging.debug(traceback.format_exc())

    # Function to evaluate the model on validation or test data
    def evaluate(model, loader):
        log_cpu_memory("Before Trial")
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:#Loads arguments from sequences
                x, y = x.to(device), y.to(device) 
                preds = model(x) #Passes the sequences through the model
                loss = criterion(preds, y) #Calculates the loss
                total_loss += loss.item()
                
        log_cpu_memory("After Trial")        
        return total_loss

    epochs = 15
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
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {total_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    def predict(model, loader):
        model.eval()
        preds, targets = [], [] #Defines empty lists to store predictions and targets
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                preds.append(pred) #Appends models predictions to the list
                targets.append(y) #Appends actual values to the list
                    
            preds = torch.cat(preds, dim=0) #Converts datatype back for metrics evaluation
            targets = torch.cat(targets, dim=0)

        mse, rmse, mae, r2, mape = evaluate_metrics(preds, targets)
        return preds,targets,mse,rmse,mae,r2,mape

    test_loss = evaluate(model, test_loader)
    # Optional: Get predictions and true values for plotting
    test_preds, test_actuals, mse, rmse, mae, r2, mape = predict(model, test_loader)

    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
    
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10,timeout=600)

    except Exception as e:
        logging.error(f"Error during Optuna study: {e}")
        logging.debug(traceback.format_exc())

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (MSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    def plot(test_prediction,test_actuals):
        #Plotting the predictions vs actual values
        plt.figure(figsize=(10, 5))
        plt.plot(test_prediction, label='Predicted')
        plt.plot(test_actuals, label='Actual')
        plt.title("LSTM Model Predictions vs Actual")
        plt.xlabel("Time")
        plt.ylabel("Scaled Close Price")
        plt.legend()
        plt.show()
    
    plot(test_preds,test_actuals)
except Exception as e:
    logging.error(f"An error occurred: {e}")
    logging.debug(traceback.format_exc())

   
    
        

        
        
    


    



