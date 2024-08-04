import sys
import numpy as np 

# Import Torch 
import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, optim

import brevitas.nn as qnn
from brevitas.quant import *
from brevitas.core.restrict_val import RestrictValueType
from brevitas.graph.calibrate import bias_correction_mode, calibration_mode

from torchinfo import summary
import brevitas.config as config
config.IGNORE_MISSING_KEYS = True

def net_input_size(X_train):
    example = X_train[0]
    if(len(np.shape(example)) == 1 or (len(np.shape(example)) == 3)):
        example = np.expand_dims(example, axis = 0)
    else:
        example = np.expand_dims(example, axis = (0,1))
    return np.shape(example)

def display_model_info(model, input_size):
    a = summary(model, input_size, col_names = ("output_size", "num_params", "mult_adds"), verbose = 1)
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

def group_indices_by_multiple_ranges(values, ranges):
    
    ranges_dict = {f"{start}-{end}": [] for start, end in ranges}
    
    for index, value in enumerate(values):
        for start, end in ranges:
            if start <= value <= end:
                range_key = f"{start}-{end}"
                ranges_dict[range_key].append(index)

    # Function to split consecutive indices into separate lists
    def split_consecutive_indices(indices):
    
        if not indices:
            return []
    
        result = []
        current_list = [indices[0]]
    
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_list.append(indices[i])
                if len(current_list) == 3:
                    result.append(current_list)
                    current_list = []
    
            else:
                if current_list:
                    result.append(current_list)
                current_list = [indices[i]]
    
        if current_list:
            result.append(current_list)
    
        return result

    ranges_dict["0.5-0.7"] = split_consecutive_indices(ranges_dict["0.5-0.7"])
    ranges_dict["0.7-1"] = split_consecutive_indices(ranges_dict["0.7-1"])
    
    return ranges_dict

def count_total_lists(grouped_indices):
    total_lists = 0
    for key in grouped_indices:
        if grouped_indices[key]: 
            if isinstance(grouped_indices[key][0], list):
                total_lists += len(grouped_indices[key])
            else:
                total_lists += 1
    return total_lists

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
    if(len(macc_per_layer) >= 10):

        ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1)]
        values = [mpl/max(macc_per_layer) for mpl in macc_per_layer]
        grouped_lists = group_indices_by_multiple_ranges(values, ranges)
        concat_grouped_list = []

        for _, index_lists in grouped_lists.items():
            if index_lists:
                if isinstance(index_lists[0], list):
                    for sublist in index_lists:
                        concat_grouped_list.append(sublist)
                else:
                    concat_grouped_list.append(index_lists)

        weights_per_layer = generate_sequences(count_total_lists(grouped_lists))

        temp_weights = np.ndarray((len(weights_per_layer), len(macc_per_layer)), dtype = np.int8)

        for i in range(len(weights_per_layer)):
            w = weights_per_layer[i]
            for j in range(len(w)):
                indices = concat_grouped_list[j]
                for id in indices:
                    temp_weights[i][id] = w[j]
        
        temp_weights = temp_weights.tolist()
        temp_weights_f1 = []

        for el in grouped_lists["0-0.1"]:
            for w in temp_weights:
                if w[el] == 8:
                    temp_weights_f1.append(w)

        if len(temp_weights_f1) == 0:
            temp_weights_f1 = temp_weights

        temp_weights_f2 = []

        for el in grouped_lists["0.1-0.3"]:
            for w in temp_weights_f1:
                if w[el] == 8 or w[el] == 4:
                    temp_weights_f2.append(w)

        if len(temp_weights_f2) == 0:
            weights_per_layer = temp_weights_f1

        else:
            weights_per_layer = temp_weights_f2

    elif(len(macc_per_layer) >= 6):
        cc = 0 
        idx = []
        for i, mpl in enumerate(macc_per_layer):
            if(mpl/max(macc_per_layer) < 0.05):
                cc += 1
                idx.append(i)
    
        weights_per_layer = generate_sequences(len(macc_per_layer) - cc)
        
        for w in weights_per_layer:
            for i in idx:
                w.insert(i, 8)
    else:
        weights_per_layer = generate_sequences(len(macc_per_layer))
    
    total_macc_opt = []

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
        nn.Conv2d: lambda layer, bw: (qnn.QuantConv2d(in_channels=layer.in_channels, 
                                                        out_channels=layer.out_channels, 
                                                        kernel_size=layer.kernel_size, 
                                                        stride=layer.stride[0], 
                                                        padding=layer.padding,
                                                        groups=layer.groups,
                                                        bias=True,
                                                        cache_inference_bias=True,
                                                        bias_quant=Int32Bias,
                                                        weight_bit_width=bw,
                                                        weight_quant=Int8WeightPerTensorFloat,
                                                        weight_scaling_min_val=2e-16,
                                                        restrict_scaling_type=RestrictValueType.LOG_FP,
                                                        return_quant_tensor=True
                                                        ) if layer.groups != layer.in_channels or layer.groups == 1 else (
                                                            # Special case for depthwise convolutions
                                        qnn.QuantConv2d(in_channels=layer.in_channels, 
                                                                out_channels=layer.out_channels, 
                                                                kernel_size=layer.kernel_size, 
                                                                stride=layer.stride[0], 
                                                                padding=layer.padding,
                                                                groups=layer.groups,
                                                                bias=True,
                                                                cache_inference_bias=True,
                                                                bias_quant=Int32Bias,
                                                                weight_bit_width = 8,  # Fixed bit width for depthwise convolutions
                                                                weight_quant=Int8WeightPerTensorFloat,
                                                                weight_scaling_min_val=2e-16,
                                                                restrict_scaling_type=RestrictValueType.LOG_FP,
                                                                return_quant_tensor=True))),

        nn.Linear: lambda layer, bw: qnn.QuantLinear(in_features = layer.in_features, 
                                                    out_features = layer.out_features, 
                                                     
                                                    cache_inference_bias = True,
                                                    bias_quant = Int32Bias,
                                                    bias = True,
                                                    
                                                    weight_quant = Int8WeightPerTensorFloat, 
                                                    weight_bit_width = bw,
                                                    return_quant_tensor=True),

        nn.ReLU: lambda _, bw: qnn.QuantReLU(bit_width = 8, 
                                            return_quant_tensor = True),

        nn.ReLU6: lambda _, bw: qnn.QuantReLU(bit_width = 8, 
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

# Define a custom module for layers with a shortcut
class ResidualModule(nn.Module):
    def __init__(self, layer, bit_widths, layer_mapping, layer_idx, act_f, layer_residual):
        super(ResidualModule, self).__init__()

        self.layer = convert_model(layer, bit_widths, layer_mapping, layer_idx)

        self.activation = act_f

        self.residual = layer_residual

    def forward(self, x):
        res = self.layer(x)
        res = self.activation(res)
        if(self.residual):
            res += x
        return res

# Function to convert a PyTorch model to a Brevitas model
def convert_model(module, bit_widths, layer_mapping, layer_idx = [0]):
    brevitas_module = nn.Sequential()
    for name, layer in module.named_children():
        if list(layer.children()):  # If the layer has children, recurse
            if hasattr(layer, 'shortcut'):
                if(layer.shortcut == False):
                    act_f = qnn.QuantIdentity(bit_width = 8, return_quant_tensor = True,
                                    act_quant = Int8ActPerTensorFloat, scaling_min_val = 2e-16, 
                                    restrict_scaling_type = RestrictValueType.LOG_FP)
                    act_prev = act_f
                else:
                    act_f = act_prev

                brevitas_module.add_module(name, ResidualModule(layer, bit_widths, layer_mapping, layer_idx, 
                                                                act_f, layer.shortcut))
            else:
                brevitas_module.add_module(name, convert_model(layer, bit_widths, layer_mapping, layer_idx))
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
                         act_quant = Uint8ActPerTensorFloat, scaling_min_val = 2e-16, 
                                        restrict_scaling_type = RestrictValueType.LOG_FP)
    
        else:
            self.quant_inp = qnn.QuantIdentity(bit_width = 8, return_quant_tensor = True,
                         act_quant = Int8ActPerTensorFloat, scaling_min_val = 2e-16, 
                                        restrict_scaling_type = RestrictValueType.LOG_FP)

        self.sequential = convert_model(og_model, w, layer_mapping, [0])
        self.o_quant =  qnn.QuantIdentity(bit_width = 8, return_quant_tensor = True)
    
    def forward(self, X):
        X = self.quant_inp(X)
        X = self.sequential(X)
        X = self.o_quant(X)
        return F.log_softmax(X, dim = 1)

def _count_layers(submodule, sequential_counts, prefix=''):
    if isinstance(submodule, nn.Conv2d):
        # Increment the conv layer count if it's a Conv2d
        sequential_counts[-1][0] += 1
    elif isinstance(submodule, nn.Linear):
        # Increment the linear layer count if it's a Linear
        sequential_counts[-1][1] += 1
    elif isinstance(submodule, nn.Sequential):
        # Append a new tuple for this sequential block
        sequential_counts.append([0, 0])
        # Recursively count layers in this sequential block
        for name, child in submodule.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            _count_layers(child, sequential_counts, child_prefix)
    else:
        # Recursively process children of non-nn.Sequential modules
        for name, child in submodule.named_children():
            _count_layers(child, sequential_counts, prefix)

def count_layers_in_sequential(module):
    sequential_counts = [[0, 0]]  # Initialize with a count tuple for the top level
    _count_layers(module, sequential_counts)
    output = []
    for i, (c, l) in enumerate(sequential_counts[1:]): # We ignore the top level module
        if(c +l != 0):
            output.append((c, l))
    return output

def train_quant_model(quant_net, train_loader, val_loader = None, device = 'cpu',
                      epochs = 20, lr = 0.0001):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(quant_net.parameters(), lr = lr)
    
    patience = 5
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

def calibrate_model(quant_model, calibration_loader, device = 'cpu'):
    with torch.no_grad():
        # Put the model in calibration mode to collect statistics
        # Quantization is automatically disabled during the calibration, and re-enabled at the end
        with calibration_mode(quant_model):
            for i, (images, _) in enumerate(calibration_loader):
                images = images.to(device)
                quant_model(images)

        # Apply bias correction
        with bias_correction_mode(quant_model):
            for i, (images, _) in enumerate(calibration_loader):
                images = images.to(device)
                quant_model(images)
                
    return quant_model

def extract_params(module, weights = [], biases = []):
    for _, submodule in module.named_children():
        # Check if the submodule has weights and append them if present
        if hasattr(submodule, 'weight') and submodule.weight is not None:
            if not isinstance(submodule, nn.BatchNorm2d):
                weights.append(submodule.weight.cpu().clone().detach().numpy())
                biases.append(submodule.bias.cpu().clone().detach().numpy())

        # Recursively extract parameters from the children modules
        extract_params(submodule, weights, biases)

    return weights, biases

def set_ptq_net_params(trained_model, new_model):
    weights, biases = extract_params(trained_model)
    custom_model_dict = new_model.state_dict()

    for i, (name, _) in enumerate(custom_model_dict.items()):
        if(i%2 == 0):
            custom_model_dict[name] =  torch.tensor(weights[i//2])
        else:
            custom_model_dict[name] = torch.tensor(biases[i//2])
    
    new_model.load_state_dict(custom_model_dict)
    return new_model


def ptq_net(custom_model, quant_net, cal_loader, device = 'cpu'):
    quant_net = set_ptq_net_params(custom_model, quant_net)
    quant_net = quant_net.to(device)
    quant_net = calibrate_model(quant_net, cal_loader, device)
    
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
    
def dse(og_model, max_acc_drop, weights_per_layer, fp_accuracy, train_loader, test_loader, 
        val_loader = None, method = 'qat', device = 'cpu', epochs = 5, lr = 0.0001):
    
    sign = calculate_minimum(train_loader) >= 0
    seq_counts = count_layers_in_sequential(og_model)

    if max_acc_drop is not None:
        print('\nDSE STARTING ... BINARY SEARCH')
        opt_found = 0
        low = 0
        high = len(weights_per_layer) - 1
        while low <= high:
            mid = (low + high) // 2
            w = weights_per_layer[mid]
            
            f_w = []
            for j in range(len(seq_counts)):
                t_w = w[j]
                c,l = seq_counts[j]
                for _ in range(c+l):
                    f_w.append(t_w)

            if(len(seq_counts) > 0):
                w = f_w

            # Create and train the quantized network
            layer_mapping = create_layer_mapping(w)
            quant_net = Quant_Model(og_model, w, layer_mapping, sign)
            quant_net = quant_net.to(device)
            print(f'==========================\nEvaluating Configuration: {mid} --> Weights: {w}')

            if(method == 'ptq' or method == 'both'):
                if(val_loader == None):
                    cal_loader = train_loader
                else:
                    cal_loader = val_loader

                print('Starting PTQ ...')
                quant_net = ptq_net(og_model, quant_net, cal_loader, device)
                accuracy = quant_net_evaluation(quant_net, test_loader, device)

            if(method == 'qat' or method == 'both'):
                print('Starting QAT ...')
                for k in range(len(epochs)):
                    quant_net = train_quant_model(quant_net, train_loader, val_loader, device,
                                      epochs = epochs[k], lr = lr[k])
            
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
    
        if(opt_found == 0):
            print("No solution that meets user's criteria was found !!")
            optimal_config = w
            
        else:
            quant_net = optimal_quant_net

        return quant_net, optimal_config
    
    else:   # Exhaustive Search for optimal solutions & to create Pareto Space for the specific Model
        print('\nDSE STARTING ... EXHAUSTIVE SEARCH')
        test_accuracy = []
        for i, w in enumerate(weights_per_layer):
            f_w = []
            for j in range(len(seq_counts)):
                t_w = w[j]
                c,l = seq_counts[j]
                for _ in range(c+l):
                    f_w.append(t_w)

            if(len(seq_counts) > 0):
                w = f_w
                
            layer_mapping = create_layer_mapping(w)
            quant_net = Quant_Model(og_model, w, layer_mapping, sign)
            quant_net = quant_net.to(device)
            print(f'===================================\nModel No {i} --> {w}')
            
            if(method == 'ptq' or method == 'both'):
                if(val_loader == None):
                    cal_loader = train_loader
                else:
                    cal_loader = val_loader

                print('Starting PTQ ...')
                quant_net = ptq_net(og_model, quant_net, cal_loader, device)
                accuracy = quant_net_evaluation(quant_net, test_loader, device)

            if(method == 'qat' or method == 'both'):
                print('Starting QAT ...')
                for k in range(len(epochs)):
                    quant_net = train_quant_model(quant_net, train_loader, val_loader, device,
                                      epochs = epochs[k], lr = lr[k])
                    
                accuracy = quant_net_evaluation(quant_net, test_loader, device)

            test_accuracy.append(accuracy)
    
        return quant_net, test_accuracy
