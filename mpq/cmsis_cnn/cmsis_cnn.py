import init_utils
import common

# Initialize the environment and get the name
name = init_utils.initialize_environment(__file__)
args = init_utils.get_args()

# Set arguments from command line
max_acc_drop = args.max_acc_drop
device = args.device
method = args.method

from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

# Load our Dataset

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = np.squeeze(y_train, axis = 1)
y_test = np.squeeze(y_test, axis = 1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15)

X_train = (np.transpose(X_train, (0,3,1,2)) - 128.0)/255.0
X_test = (np.transpose(X_test, (0,3,1,2)) - 128.0)/255.0
X_val = (np.transpose(X_val, (0,3,1,2)) - 128.0)/255.0

BATCH_SIZE = 32
epochs = [50, 25]
lr = [0.0001, 0.00001]

class CMSIS_CNN(nn.Module):
    def __init__(self):
        super(CMSIS_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, padding = 2)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d(2,2)
        self.d1 = nn.Dropout(p = 0.25)
        
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, padding = 2)
        self.relu2 = nn.ReLU()
        self.max2 = nn.MaxPool2d(2,2)
        self.d2 = nn.Dropout(p = 0.25)
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 2)
        self.relu3 = nn.ReLU()
        self.max3 = nn.MaxPool2d(2,2)
        self.d3 = nn.Dropout(p = 0.4)
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 10)
        
    def forward(self,X):
        X = self.relu1((self.conv1(X)))
        X = self.max1(X)
        X = self.d1(X)
        
        X = self.relu2((self.conv2(X)))
        X = self.max2(X)
        X = self.d2(X)
        
        X = self.relu3((self.conv3(X)))
        X = self.max3(X)
        X = self.d3(X)
        
        X = self.flatten(X)

        X = self.linear1(X)
        return F.log_softmax(X, dim = 1)

net = CMSIS_CNN()

common.create_ibex_qnn(net, name, device, X_train, y_train, X_test, y_test, 
                X_val = X_val, y_val = y_val, pretrained = False, 
                BATCH_SIZE = BATCH_SIZE, method = method, epochs = epochs, 
                lr = lr, max_acc_drop = max_acc_drop)
