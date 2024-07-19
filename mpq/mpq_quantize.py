import numpy as np 
import sys

# Import Torch 
import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, optim

import brevitas.nn as qnn
from brevitas.quant import *
from torchinfo import summary

def net_input_size(X_train):
    example = X_train[0]
    if(len(np.shape(example)) == 1 or (len(np.shape(example)) == 3)):
        example = np.expand_dims(example, axis = 0)
    else:
        example = np.expand_dims(example, axis = (0,1))
    return np.shape(example)

def display_model_info(model, input_size):
    a = summary(model, input_size, col_names = ("output_size", "num_params", "mult_adds"))
    model_params_str = str(a)
    
    lines = model_params_str.split('\n')
    lines_with_macc = []
    for line in lines:
        if(line.startswith('├─') or line.startswith('│')):
            lines_with_macc.append(line)

    type_of_layer = []
    macc_per_layer = []

    for l in lines_with_macc:
        k = l.split()[-1]
        if(k != '--'):
            type_of_layer.append(l.split()[0].replace(':','').replace('├─',''))
            macc_per_layer.append(int(l.split()[-1].replace(',','')))
        else:
            type_of_layer.append(l.split()[0].replace(':','').replace('├─',''))

    return macc_per_layer, type_of_layer

def create_dataloaders(BATCH_SIZE, X_train, y_train, X_test, y_test, 
                       X_val = None, y_val = None):

    if(X_val is None):
        torch_X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) 

        # Create feature and targets tensor for test set.

        torch_X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
        torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor) 

        train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
        test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)

        # Data Loaders
        train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
        val_loader = None

    else:
        torch_X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) 

        torch_X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
        torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor) 

        torch_X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
        torch_y_val = torch.from_numpy(y_val).type(torch.LongTensor) 

        train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
        test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)
        val = torch.utils.data.TensorDataset(torch_X_val, torch_y_val)

        # Data Loaders
        train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
        val_loader = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE, shuffle = True)
        
    
    return train_loader, val_loader, test_loader

# Function to calculate the minimum value of a DataLoader
def calculate_minimum(dataloader):
    global_min = float('inf')
    for batch in dataloader:
        inputs, _ = batch
        batch_min = inputs.min().item()
        if batch_min < global_min:
            global_min = batch_min
    return global_min

def fp_train(net, train_loader, val_loader = None, device = 'cpu', epochs = 20, lr = 0.0001):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = lr)

    patience = 10
    best_val_loss = float('inf')    
    train_losses, val_losses = [], []

    net = net.to(device)
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Prevent accumulation of gradients
            optimizer.zero_grad()
            # Make predictions
            log_ps = net(images.float())
            loss = criterion(log_ps, labels)
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        val_loss = 0
        accuracy = 0

        # Turn off gradients for validation, to save memory and computations
        with torch.no_grad():
            net.eval()
            if(val_loader != None):
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = net(images.float())
                    val_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    # Get top predictions
                    _, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

        net.train()

        train_losses.append(running_loss/len(train_loader))

        if(val_loader != None):
            val_losses.append(val_loss/len(val_loader))
            print(f"Epoch {e+1}/{epochs}.. "
              f"Train loss: {train_losses[-1]:.3f}.. "
              f"Validation loss: {val_losses[-1]:.3f}.. "
              f"Validation accuracy: {accuracy/len(val_loader):.3f}")

            # Check for early stopping
            avg_val_loss = val_loss/len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
                
            if counter >= patience:
                break
        else:
            print(f"Epoch {e+1}/{epochs}.. "
              f"Train loss: {train_losses[-1]:.3f}.. ")
        
    return net

def fp_evaluate(net, test_loader, device):
    print('\nFULL PRECISION MODEL EVALUATION ...')
    # Turn off gradients for validation
    with torch.no_grad():
        net.eval()
        correct = 0 
        y_size = 0
        for test_imgs, test_labels in test_loader:
            test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
            test_imgs = Variable(test_imgs).float()
            output = net(test_imgs)
            predicted = torch.max(output,1)[1]
            correct += (predicted == test_labels).sum()
            y_size += len(test_labels)
        print("Test accuracy: {:.3f}% ".format(100*float(correct)/(y_size)))
        floating_acc = 100*float(correct)/y_size
    return floating_acc


def generate_sequences(length, values = [2, 4, 8]):
    sequences = []

    def generate_sequence_helper(seq):
        if len(seq) == length:
            sequences.append(seq)
            return

        for value in values:
            generate_sequence_helper(seq + [value])

    generate_sequence_helper([])

    return sequences

def create_weight_confs(macc_per_layer):
    total_macc_opt = []
    weights_per_layer = generate_sequences(len(macc_per_layer))
    
    for w_conf in weights_per_layer:
        macc = 0
        for i, w in enumerate(w_conf):
            if(w == 2):
                macc += macc_per_layer[i]/16
            elif(w == 4):
                macc += macc_per_layer[i]/8
            else:
                macc += macc_per_layer[i]/4
    
        total_macc_opt.append(np.round(macc))

    # Get the indexes in descending order based on the values

    sorted_indexes = sorted(enumerate(total_macc_opt), key=lambda x: x[1])

    # Extract the sorted indexes

    ascending_indexes = [index for index, _ in sorted_indexes]

    weights_per_layer = [weights_per_layer[i] for i in ascending_indexes]  
    
    total_macc_opt_sorted = [total_macc_opt[i] for i in ascending_indexes]

    return weights_per_layer, total_macc_opt_sorted


# Define a mapping from PyTorch layers to Brevitas layers
def create_layer_mapping(bit_width):
    mapping = {
        nn.Conv2d: lambda layer, bw: qnn.QuantConv2d(in_channels = layer.in_channels, 
                                                    out_channels = layer.out_channels, 
                                                    kernel_size = layer.kernel_size, 
                                                    stride = layer.stride[0], 
                                                    padding = layer.padding,
                                                    bias = True,
                                                    cache_inference_bias = True,
                                                    bias_quant = Int32Bias,
                                                    weight_bit_width = bw,
                                                    weight_quant = Int8WeightPerTensorFloat),

        nn.Linear: lambda layer, bw: qnn.QuantLinear(in_features = layer.in_features, 
                                                    out_features = layer.out_features, 
                                                    cache_inference_bias = True, 
                                                    weight_quant = Int8WeightPerTensorFloat,
                                                    bias_quant = Int32Bias,
                                                    bias = True,
                                                    weight_bit_width = bw),

        nn.ReLU: lambda _, bw: qnn.QuantReLU(bit_width = bw, 
                                            return_quant_tensor = True),

        nn.MaxPool2d: lambda layer, _: qnn.QuantMaxPool2d(kernel_size = layer.kernel_size,
                                                        stride = layer.stride,
                                                        padding = layer.padding,
                                                        return_quant_tensor = True),

        nn.AvgPool2d: lambda layer, _: qnn.TruncAvgPool2d(kernel_size = layer.kernel_size,
                                                        stride = layer.stride,
                                                        padding = layer.padding,
                                                        return_quant_tensor = True),
    }
    
    return mapping

# Function to convert a PyTorch layer to a Brevitas layer with a specified bit width
def convert_layer(layer, bit_width, layer_mapping):
    layer_type = type(layer)
    if layer_type in layer_mapping:
        return layer_mapping[layer_type](layer, bit_width)
    else:
        return layer

# Function to convert a PyTorch model to a Brevitas model
def convert_model(module, bit_widths, layer_mapping):
    layer_idx = [0]
    brevitas_module = nn.Sequential()

    for name, layer in module.named_children():
        if list(layer.children()):  # If the layer has children, recurse
            brevitas_module.add_module(name, convert_model(layer, bit_widths, layer_mapping))
        else:
            layer_type = type(layer)
            if layer_type in [nn.Conv2d, nn.Linear]:
                bit_width = bit_widths[layer_idx[0]]
                layer_idx[0] += 1
            else:
                bit_width = 8
            brevitas_module.add_module(name, convert_layer(layer, bit_width, layer_mapping))
    return brevitas_module

class Quant_Model(nn.Module):
    def __init__(self, og_model, w, layer_mapping, input_sign = True):
        super(Quant_Model, self).__init__()
        if(input_sign):
            self.quant_inp = qnn.QuantIdentity(bit_width = 8, return_quant_tensor = True,
                         act_quant = Uint8ActPerTensorFloat)
    
        else:
            self.quant_inp = qnn.QuantIdentity(bit_width = 8, return_quant_tensor = True,
                         act_quant = Int8ActPerTensorFloat)

        self.sequential = convert_model(og_model, w, layer_mapping)
        self.o_quant =  qnn.QuantIdentity(bit_width = 8, return_quant_tensor = True)
    
    def forward(self, X):
        X = self.quant_inp(X)
        X = self.sequential(X)
        X = self.o_quant(X)
        return F.log_softmax(X, dim = 1)
        
def train_quant_model(quant_net, train_loader, val_loader = None, device = 'cpu',
                      epochs = 20, lr = 0.0001):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(quant_net.parameters(), lr = lr)
    
    patience = 10
    best_val_loss = float('inf')

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Prevent accumulation of gradients
            optimizer.zero_grad()
            # Make predictions
            log_ps = quant_net(images.float())
            loss = criterion(log_ps, labels)
            #backprop
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

            val_loss = 0
            accuracy = 0

        # Turn off gradients for validation
        with torch.no_grad():
            quant_net.eval()
            if(val_loader != None):
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = quant_net(images.float())
                    val_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    # Get our top predictions
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

        if(val_loader != None):
            # Check for early stopping
            avg_val_loss = val_loss/len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
                
            if counter >= patience:
                break

        quant_net.train()
        
    return quant_net

def quant_net_evaluation(quant_net, test_loader, device = 'cpu'):
    with torch.no_grad():
        quant_net.eval()
        correct = 0 
        y_size = 0
        for test_imgs, test_labels in test_loader:
            test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
            test_imgs = Variable(test_imgs).float()
            output = quant_net(test_imgs)
            predicted = torch.max(output, 1)[1]
            correct += (predicted == test_labels).sum()
            y_size += len(test_labels)
        print("Test accuracy: {:.3f}% ".format(100*float(correct)/y_size))
        return 100 * float(correct)/y_size
    
def dse(og_model, max_acc_drop, weights_per_layer, fp_accuracy, train_loader, test_loader, val_loader = None,
        device = 'cpu', epochs = 5, lr = 0.0001):
    
    sign = calculate_minimum(train_loader) >= 0

    if max_acc_drop is not None:
        print('\nDSE STARTING ... BINARY SEARCH')
        opt_found = 0
        low = 0
        high = len(weights_per_layer) - 1
        while low <= high:
            mid = (low + high) // 2
            w = weights_per_layer[mid]
            
            # Create and train the quantized network
            layer_mapping = create_layer_mapping(w)
            quant_net = Quant_Model(og_model, w, layer_mapping, sign)
            quant_net = quant_net.to(device)
            print(f'==========================\nEvaluating Configuration: {mid} --> Weights: {w}')

            for i in range(len(epochs)):
                quant_net = train_quant_model(quant_net, train_loader, val_loader, device,
                                      epochs = epochs[i], lr = lr[i])
            
            # Evaluate the trained quantized network
            accuracy = quant_net_evaluation(quant_net, test_loader, device)
            
            # Check if the accuracy drop is within the acceptable range
            if fp_accuracy - accuracy <= max_acc_drop:
                opt_found = 1
                optimal_quant_net = quant_net
                optimal_config = w
                high = mid - 1  # Try to find a less complex configuration that meets the criteria
            else:
                low = mid + 1  # Too much accuracy loss, look for a more complex configuration
    
        quant_net = optimal_quant_net

        if(opt_found == 0):
            print("No solution that meets user's criteria was found !!")
            optimal_config = w

        return quant_net, optimal_config
    
    else:   # Exhaustive Search for optimal solutions & to create Pareto Space for the specific Model
        print('\nDSE STARTING ... EXHAUSTIVE SEARCH')
        test_accuracy = []
        for i, w in enumerate(weights_per_layer):
            layer_mapping = create_layer_mapping(w)
            quant_net = Quant_Model(og_model, w, layer_mapping, sign)
            quant_net = quant_net.to(device)
            print(f'===================================\nModel No {i} --> {w}')
            for i in range(len(epochs)):
                quant_net = train_quant_model(quant_net, train_loader, val_loader, device,
                                      epochs = epochs[i], lr = lr[i])
            accuracy = quant_net_evaluation(quant_net, test_loader, device)
            test_accuracy.append(accuracy)
    
        return quant_net, test_accuracy
