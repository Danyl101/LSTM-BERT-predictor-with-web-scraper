
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
def data_loader(data, target_col, lookback, shift_dict=None):
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

    #correlation_matrix=train_data.corr()  # Calculate correlation matrix
    #print("Correlation Matrix:\n", correlation_matrix)

    #Calls the TimeSeriesDataset class to create sequences for each dataset
    train_dataset=TimeSeriesDataset(train_data, 'Close', lookback,shift_dict={'Close':0})
    val_dataset=TimeSeriesDataset(val_data, 'Close', lookback,shift_dict={'Close':0})
    test_dataset=TimeSeriesDataset(test_data, 'Close', lookback,shift_dict={'Close':0})

    # DataLoader to handle batching and shuffling 
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader=DataLoader(test_dataset, batch_size=32)

def evaluate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy() #Takes the actual values
    y_pred = y_pred.detach().cpu().numpy() #Takes the predicted values

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true)).mean() * 100

    return mse, rmse, mae, r2, mape #Returns various metrics

#Bayesian Optimization
def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
    train_loader = DataLoader(train_dataset) 
    test_loader=DataLoader(test_dataset)

    model = BiLSTMModel(input_size=4, hidden_size=hidden_size, dropout=dropout,batch_size=batch_size).to(device) #Defines model for bayesian
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Defines optimizer

    epoch = 50 #No of epochs
    # Training loop (light version)
    for epoch in range(epoch):
        model.train()
        for xb, yb in train_loader: #Loads input features and target features
            xb, yb = xb.to(device), yb.to(device)               
            optimizer.zero_grad() #Resets optimizer back to zero gradient after every epoch
            output = model(xb)
            loss = criterion(output.squeeze(), yb) #Checks models output with the actual output
            loss.backward() #Propagates backwards and finds gradients of loss 
            optimizer.step() #Updates the weights

    # Evaluate
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).squeeze().cpu().numpy()
            preds.extend(pred)
            truths.extend(yb.numpy())
            
    preds = torch.cat(preds, dim=0) #Converts datatype back for metrics evaluation
    truths = torch.cat(truths, dim=0)

    return evaluate_metrics(truths, preds)


#Plotting the predictions vs actual values
def plot_predictions(test_preds, test_actuals):
    plt.figure(figsize=(10, 5))
    plt.plot(test_preds, label='Predicted')
    plt.plot(test_actuals, label='Actual')
    plt.title("LSTM Model Predictions vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Scaled Close Price")
    plt.legend()
    plt.show()
    
