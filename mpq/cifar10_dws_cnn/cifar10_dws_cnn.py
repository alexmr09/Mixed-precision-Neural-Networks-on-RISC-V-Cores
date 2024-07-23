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

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train = np.squeeze(y_train, axis = 1)
y_test = np.squeeze(y_test, axis = 1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15)

X_train = (np.transpose(X_train, (0,3,1,2)))
X_test = (np.transpose(X_test, (0,3,1,2)))
X_val = (np.transpose(X_val, (0,3,1,2)))

BATCH_SIZE = 128
epochs = [10, 10]
lr = [0.001, 0.0001]

class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseBlock, self).__init__()
        
        layers = []
                    
        layers.append(nn.Conv2d(in_channels = in_channels, out_channels = in_channels, 
                                    kernel_size = 3, padding = 1, groups = in_channels))  # Depthwise convolution
        
        layers.append(nn.ReLU(inplace = True))
        
        layers.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                                    kernel_size = 1, padding = 0))  # Pointwise convolution
            
        layers.append(nn.ReLU(inplace = True))
                            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class Cifar10_Dws_CNN(nn.Module):
    def __init__(self):
        super(Cifar10_Dws_CNN, self).__init__()
        self.features = nn.Sequential(
            DepthwiseBlock(in_channels = 3, out_channels = 64),
            DepthwiseBlock(in_channels = 64, out_channels = 64),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            DepthwiseBlock(in_channels = 64, out_channels = 128),
            DepthwiseBlock(in_channels = 128, out_channels = 128),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            DepthwiseBlock(in_channels = 128, out_channels = 256),
            DepthwiseBlock(in_channels = 256, out_channels = 256),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 10)  # Assuming input size is (32, 32) and after 3 max pooling layers, the size is (4, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim = 1)

net = Cifar10_Dws_CNN()

common.create_ibex_qnn(net, name, device, X_train, y_train, X_test, y_test, 
                X_val = X_val, y_val = y_val, BATCH_SIZE = BATCH_SIZE, 
                epochs = epochs, lr = lr, max_acc_drop = max_acc_drop)
