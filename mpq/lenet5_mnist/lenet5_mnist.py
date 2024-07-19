import init_utils
import common

# Initialize the environment and get the name
name = init_utils.initialize_environment(__file__)
args = init_utils.get_args()

# Set arguments from command line
max_acc_drop = args.max_acc_drop
device = args.device

from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

# Load our Dataset

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15)

X_train = np.expand_dims(X_train, axis = 1)
X_test = np.expand_dims(X_test, axis = 1)
X_val = np.expand_dims(X_val, axis = 1)

BATCH_SIZE = 32
epochs = 1
lr = 0.0001

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding = 'same')
        self.relu1 = nn.ReLU()
        self.avg1 = nn.AvgPool2d(2,2)
        
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        self.relu2 = nn.ReLU()
        self.avg2 = nn.AvgPool2d(2,2)
        
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()

        self.linear2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        self.linear3 = nn.Linear(84, 10)
        
    def forward(self,X):
        X = self.relu1(self.conv1(X))
        X = self.avg1(X)
        
        X = self.relu2(self.conv2(X))
        X = self.avg2(X)

        X = self.flatten(X)
        
        X = self.relu3(self.linear1(X))
        X = self.relu4(self.linear2(X))
        X = self.linear3(X)
        return F.log_softmax(X, dim = 1)

net = LeNet5()

common.create_ibex_qnn(net, name, device, X_train, y_train, X_test, y_test, 
                X_val = X_val, y_val = y_val, BATCH_SIZE = BATCH_SIZE, 
                epochs = epochs, lr = lr, max_acc_drop = max_acc_drop)
