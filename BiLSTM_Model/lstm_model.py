import torch
import torch.nn as nn
import optuna
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
import time
import traceback
import gc

from lstm_dataload import train_dataset, test_dataset
from lstm_utils import evaluate_metrics, log_cpu_memory, device
   
# 1. LSTM Model Definition
class BiLSTMModel(nn.Module):
    #Defines the structure of the LSTM model
        def __init__(self, input_size=4, hidden_size=64, num_layers=2,dropout=0.3, batch_size=32):
            super().__init__()
            try:
                self.hidden_size=hidden_size
                self.batch_size=batch_size
                self.num_layers=num_layers
                self.dropout=dropout
                self.bidirectional=True
            except Exception as e:
                logging.error(f"Error in BiLSTMModel: {e}")
                logging.debug(traceback.format_exc())
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=self.bidirectional)#Model is defined
            self.dropout=nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size*2, 1)

        #Defines movement of data through the model
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]  # Take the last timestep
            out = self.dropout(out)
            return self.fc(out).view(-1) #Removes last dimension from output tensor

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
        torch.save(model.state_dict(), f"checkpoints/best_model_trial_{trial.number}.pt")
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
        def __init__(self):
            super().__init__()
        def forward(self, y_pred, y_true):
            try:
                batch_size = y_true.shape[0]  
                weights = torch.linspace(1, 2, steps=batch_size).to(y_pred.device)
                return torch.mean(weights * (y_pred - y_true) ** 2)
            except Exception as e:
                logging.error(f"Error in TimeWeightedLoss: {e}")
                logging.debug(traceback.format_exc())
                return None
        
   
    
        

        
        
    


    



