from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ta

#-------------------------
# CREATING TEST ARRAYS
#-------------------------

def create_test_arrays(X_test_tens, df, best_model, mm):
    """
    Input:
    X_test_tens : tensors of X_test
    df : the original df - as directly read in by the .csv reader

    Returns:
    
    """
    preds = create_test_preds(X_test_tens, best_model, mm)
    value_y_test = create_differenced_y(df['Close'])
    y_groundtruth = create_groundtruth_y(df['Close'])

    assert preds.shape[0] == y_groundtruth.shape[0], "ERROR: UNEQUAL X, Y LEN"

    result = [x + y for (x, y) in zip(preds, value_y_test)]
    # for (x, y) in zip(preds, value_y_test):
    #     result.append(x + y)
    return result, value_y_test, y_groundtruth, preds


def create_test_preds(X_test_tensors_final, best_model, mm):
    """pass test set tensors into LSTM, inverse and reshape"""
    WINDOW_SIZE = 20

    preds = [best_model(window.view(1,WINDOW_SIZE,6)).detach().numpy() for window in X_test_tensors_final]
    preds = np.array(preds)
    preds = preds.reshape(-1, 1)
    preds = mm.inverse_transform(preds)
    return preds

def create_differenced_y(df_col):
    # USE 1 DAY AGO DATA FOR PREDICTION
    VAL_END = 4500 + 1300
    WINDOW_SIZE = 20
    return df_col.iloc[VAL_END+WINDOW_SIZE-1+2:-1].values
    
def create_groundtruth_y(df_col):
    # REINDEX BY VALIDATION SPLIT INDEX + MOVING WINDOW SIZE + DIFFERENCING SHIFT OF 2
    VAL_END = 4500 + 1300
    WINDOW_SIZE = 20
    return df_col.iloc[VAL_END+WINDOW_SIZE+2:].values

#--------------------------
## METRICS AND VISUALIZATION
#-------------------------
#
def compute_metrics(y, predictions):

    mse = mean_squared_error(y, predictions)
    print(f'MSE: {mse:.6f}')

    mae = mean_absolute_error(y, predictions)
    print(f'MAE: {mae:.6f}')

    rmse = math.sqrt(mean_squared_error(y, predictions))
    print(f'RMSE: {rmse:.6f}')

    mape = get_MAPE(y, predictions)
    print(f'MAPE: {mape:.6f}')

    return (mse, mae, rmse, mape)
#
def get_MAPE(y, predictions):
    y, predictions = np.array(y), np.array(predictions)
    return np.mean(np.abs((y - predictions) / y)) * 100

#
def plot_TS(data_predict, dataY_plot, divline = None):
    
    plt.figure(figsize=(10,6)) #plotting

    if divline != None:
        plt.axvline(x=divline, c='r', linestyle='--') #size of the training set

    plt.plot(dataY_plot, label='Actual Data') #actual plot
    plt.plot(data_predict, label='Predicted Data') #predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()

#
def create_N_day_mean(col, N, fillna=False):
    """
        Calculate the N-day Simple Moving Average (SMA) of any array.

        Args:
            col (pandas.Series)
            N (int): The number of days to compute the SMA over.
            fillna (bool, optional): If True, fills NaN values in the result. Defaults to False.

        Returns:
            pandas.Series: A Series containing the Simple Moving Average values.
        """
    sma = ta.trend.SMAIndicator(col, N, fillna)
    return sma.sma_indicator()[N-1:]
 

#------------------------
# DATASET CONSTRUCTION
#-------------------------

#
def build_base_X_y_difference(X, y):
    """ The basic X, y datasets should differ by 1 day AND have same length
        Returns df slices of X and y respectively
        """
    X = X[:-1]
    y = y[1:]

    print(len(X), len(y))

    return X, y

#
def apply_residual_change(df_X):
    """Apply residual change transformatio to any dataframe"""
    X1, X2 = df_X[:-1].values, df_X[1:].values
    return X2 - X1

#
def create_sliding_windows(X_ss, y_mm):

    TRAIN_END = 4500
    VAL_END = 4500 + 1300
    WINDOW_SIZE = 20

    X = []

    for i in range(WINDOW_SIZE, len(X_ss)):
        X.append(X_ss[i-WINDOW_SIZE:i, :])

    X = np.array(X)

    X_train = X[:TRAIN_END, :]  # Updated to use TRAIN_END for training data end index
    X_test = X[TRAIN_END:, :]  # Test data starts after TRAIN_END

    y_train = y_mm[WINDOW_SIZE:TRAIN_END + WINDOW_SIZE, :]
    y_test = y_mm[TRAIN_END + WINDOW_SIZE:, :]

    return X_train, X_test, y_train, y_test

#
def create_datatensors(scaled_X, scaled_y):
    """
    Takes in X and y data arrays as input
    Returns sliding window X tensors and non-sliding y tensors.
    X is reshape into: batch_len * seq_length * num_features
    y is not reshaped
    """

    if 'SEQ_LEN' not in globals():
        SEQ_LEN = 20

    assert len(scaled_X.shape) == 2
    assert len(scaled_y.shape) == 2
    assert scaled_y.shape[1] == 1  # single dim column

    X_train, X_temp, y_train, y_temp = create_sliding_windows(scaled_X, scaled_y)

    VAL_TEST_SPLIT = 1300

    X_val, X_test, y_val, y_test = X_temp[:VAL_TEST_SPLIT], X_temp[VAL_TEST_SPLIT:], \
                                y_temp[:VAL_TEST_SPLIT], y_temp[VAL_TEST_SPLIT:]

    assert X_val.shape[0] == y_val.shape[0] and len(X_val) != 0
    assert X_test.shape[0] == y_test.shape[0] and len(X_test) != 0
    assert X_train.shape[0] == y_train.shape[0] and len(X_train) != 0

    #----- CONVERT TO TENSORS
    X_train_tensors = torch.Tensor(X_train)
    X_val_tensors = torch.Tensor(X_val)
    X_test_tensors = torch.Tensor(X_test)
    y_train_tensors = torch.Tensor(y_train)
    y_val_tensors = torch.Tensor(y_val)
    y_test_tensors = torch.Tensor(y_test)

    #---- RESHAPE X FOR FEEDING INTO NETWORK
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0],\
                                                            SEQ_LEN, X_train_tensors.shape[-1]))
    X_val_tensors_final = torch.reshape(X_val_tensors, (X_val_tensors.shape[0],\
                                                        SEQ_LEN, X_val_tensors.shape[-1]))
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0],\
                                                        SEQ_LEN, X_test_tensors.shape[-1]))
    
    return X_train_tensors_final, X_val_tensors_final, X_test_tensors_final, \
            y_train_tensors, y_val_tensors, y_test_tensors


#-------------
# Model Training
#-------------
#
def train_val_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs=1000, early_stopping_patience=10):
    """
    Trains on training data and evaluates on validation data with early stopping
    """
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            loss_val = criterion(outputs_val, y_val)
            val_losses.append(loss_val.item())

            # Check for early stopping
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience = 0
            else:
                patience += 1
                if patience > early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} with validation loss: {best_val_loss}")
                    break

    return model, train_losses, val_losses, best_val_loss


def batch_train_val_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, batch_size=32, num_epochs=1000, early_stopping_patience=10):
    """
    Trains on training data and evaluates on validation data with early stopping
    """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))  # Average training loss per epoch
        

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs_val, targets_val in val_loader:
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, targets_val)
                epoch_val_loss += loss_val.item()
        val_losses.append(epoch_val_loss / len(val_loader))  # Average validation loss per epoch

        # Check for early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience = 0
        else:
            patience += 1
            if patience > early_stopping_patience:
                print(f"Early stopping at epoch {epoch} with validation loss: {best_val_loss}")
                break

    return model, train_losses, val_losses, best_val_loss



def grid_search_model(X_ss, y_mm, model, **settings):
    
    input_size = settings['input_size']
    num_layers = settings['num_layers']
    num_classes = settings['num_classes']
    criterion = settings['criterion']
    learning_rate = settings['learning_rate']
    hidden_sizes = settings['hidden_sizes']
    lookback = settings['lookback']
    sequence_len = settings['sequence_len']

    # initialization
    best_params = None
    best_historical_losses = None
    best_val_loss = np.inf
    best_model = None
    historical_losses = dict()

    for lb in lookback:
        # create lookback window size
        X_train_tensors_final, X_val_tensors_final, _, \
        y_train_tensors, y_val_tensors, _ = create_datatensors(X_ss, y_mm)
        
        for hidden_size in hidden_sizes:
            current_params = {'hidden_size': hidden_size, 'lookback': lb}
            
            print('#-------------------------')
            print(r'Now running model: Hidden-size {},  lookback {}'.format(hidden_size, lb))
            
            if model == 'lstm':
                model = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])

            if model == 'rnn':
                model = RNN(input_size, num_layers, hidden_size, sequence_len, num_classes)

            if model == 'gru':
                model = GRU(input_size, hidden_size, num_layers, num_classes, sequence_len)

            if model == 'blstm':
                model = BiLSTM(input_size, hidden_size, num_layers, num_classes)

            if model == 'stackedlstm':
                model = StackedLSTM(input_size, hidden_size, num_layers, 1)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # model, train_loss, val_loss, model_val_loss = train_val_model(model, criterion, \
            #                                         optimizer, X_train_tensors_final, y_train_tensors,\
            #                                         X_val_tensors_final, y_val_tensors)
            model, train_loss, val_loss, model_val_loss = batch_train_val_model(model, criterion, \
                                                    optimizer, X_train_tensors_final, y_train_tensors,\
                                                    X_val_tensors_final, y_val_tensors)
            # store all losses'
            historical_losses[str(current_params)] = {'train_loss' : train_loss,
                                                 'val_loss' : val_loss
                                                 }
            # record best parameters/model/losses
            if best_val_loss > model_val_loss:
                # update best loss
                best_val_loss = model_val_loss
                # store best params
                best_params = current_params
                # store best historical loss
                best_historical_losses = [train_loss, val_loss]
                # store best model
                best_model = model
                
    return best_params, best_historical_losses, best_val_loss, best_model, historical_losses

#############
# NN MODELS
##############

#
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()

    def init_hidden(self):
        return
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
    
#
class RNN(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, sequence_length, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size= hidden_size
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes)
    
    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        return out
    
#
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(GRU, self).__init__()
        self.hidden_size  = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out,_ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        return out

#   
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
    
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the stacked LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size,
                    hidden_size,
                    num_layers=1,
                    batch_first=True)
            for i in range(num_layers)
        ])

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        for i in range(self.num_layers):
            out, (h_0, c_0) = self.lstm_layers[i](x, (h_0, c_0))
            x = out

        # Extract the output of the last LSTM layer and pass it through a fully connected layer
        out = self.fc(out[:, -1, :])  # You can change the indexing depending on your specific task

        return out
