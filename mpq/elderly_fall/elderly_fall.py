import init_utils
import common

# Initialize the environment and get the name
name = init_utils.initialize_environment(__file__)
args = init_utils.get_args()

# Set arguments from command line
max_acc_drop = args.max_acc_drop
device = args.device

import pandas as pd 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Load our Dataset

pelvis_acc = pd.read_excel('elderly_fall/Posterior Pelvis Accelerometer Measures.xlsx',
                   sheet_name = 'ST')

y_data = pelvis_acc['Faller (1), Non-Faller (0)'].to_numpy()

pelvis_acc = pelvis_acc.drop(columns = [pelvis_acc.columns[0], 'Faller (1), Non-Faller (0)'])

head_acc = pd.read_excel('elderly_fall/Head Accelerometer Measures.xlsx',
                    sheet_name = 'ST')

head_acc = head_acc.drop(columns = [head_acc.columns[0], 'Faller (1), Non-Faller (0)'])


pressure_sen = pd.read_excel('elderly_fall/Pressure-Sensing Insole Measures.xlsx',
                   sheet_name = 'ST')

pressure_sen = pressure_sen.drop(columns = [pressure_sen.columns[0], 'Faller (1), Non-Faller (0)'])


left_shank = pd.read_excel('elderly_fall/Left Shank Accelerometer Measures.xlsx',
                   sheet_name = 'ST')

left_shank = left_shank.drop(columns = [left_shank.columns[0], 'Faller (1), Non-Faller (0)'])


X_data = pd.concat([head_acc, pressure_sen, pelvis_acc, left_shank], axis = 1)

# Preprocess the Data

def normalize_column(column):
    old_min = column.min()
    old_max = column.max()
    new_min = 0
    new_max = 255
    normalized_column = ((column - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return normalized_column.astype(int)

X_data = X_data.apply(normalize_column)
X_d = X_data/255.0
X_train, X_test, y_train, y_test = train_test_split(X_d, y_data, 
                        test_size = 0.15, random_state = 42)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

BATCH_SIZE = 1
epochs = 5
lr = 0.0001

class EF_MLP(nn.Module):
    def __init__(self):
        super(EF_MLP, self).__init__()
        self.linear1 = nn.Linear(117, 20)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(20, 2)
        
    def forward(self,X):
        X = self.relu1(self.linear1(X))
        X = self.linear2(X)
        return F.log_softmax(X, dim = 1)
    
net = EF_MLP()

common.create_ibex_qnn(net, name, device, X_train, y_train, X_test, y_test, BATCH_SIZE = BATCH_SIZE, 
                 epochs = epochs, lr = lr, max_acc_drop = max_acc_drop)
