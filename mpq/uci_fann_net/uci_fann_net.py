import init_utils
import common

# Initialize the environment and get the name
name = init_utils.initialize_environment(__file__)
args = init_utils.get_args()

# Set arguments from command line
max_acc_drop = args.max_acc_drop
device = args.device
method = args.method

import pandas as pd 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler

# Load our Dataset
df = pd.read_csv("uci_fann_net/Data_Cortex_Nuclear.csv")
# Fix the Missing values

missing = df.isnull().sum().sum()
print("There are total {} null values".format(missing) )

missval_col = df.columns[df.isna().any()].tolist()

# Imputing Missing values with mean values 
for i in range(len(missval_col)):
    mean_value = np.mean(df[missval_col[i]])
    #df[missval_col[i]].fillna(value = mean_value, inplace = True)
    df.fillna({missval_col[i] : mean_value}, inplace = True)

print("Total number of Null Value in the updated dataset is {}".format(df.isnull().sum().sum()))

numcol = df.describe().columns.tolist()
X = df[numcol] # X dataset should include cols with numerical datasets, the protein expression level

r, c = df.shape
r2, c2 = X.shape

# Preprocess the Data with the MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 255))
df_normalized = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# Convert the normalized values to integers
df_normalized = df_normalized.round()/255.0
df_normalized = df_normalized.drop(columns = 'pAKT_N')

label_encoder = preprocessing.LabelEncoder()

X_data = df_normalized.to_numpy()
y_data = label_encoder.fit_transform(df['class'])

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size = 0.3)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size = 0.5)

### The following variables could be set as inputs 

BATCH_SIZE = 16
epochs = [80, 20]
lr = [0.0001, 0.00005]

class FANN_NET(nn.Module):
    def __init__(self):
        super(FANN_NET, self).__init__()        
        self.fc1 = nn.Linear(76, 300)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 100)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(100, 10)
        
    def forward(self,X):
        X = self.relu1(self.fc1(X))
        X = self.relu2(self.fc2(X))
        X = self.relu3(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim = 1)

net = FANN_NET()

common.create_ibex_qnn(net, name, device, X_train, y_train, X_test, y_test, 
                X_val = X_val, y_val = y_val, pretrained = False, 
                BATCH_SIZE = BATCH_SIZE, method = method, epochs = epochs, 
                lr = lr, max_acc_drop = max_acc_drop)
