import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import logging
import traceback
import psutil
import os

logging.basicConfig(
        filename="Logs/Optuna.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",
    )

def log_cpu_memory(tag=""):
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 ** 2  # in MB
        logging.info(f"[{tag}] CPU RAM Usage: {mem_mb:.2f} MB")
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
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
    
def evaluate(model, loader):
    log_cpu_memory("Before Trial")
    criterion = nn.MSELoss()
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