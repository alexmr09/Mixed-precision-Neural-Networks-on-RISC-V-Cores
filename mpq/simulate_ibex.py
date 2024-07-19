import torch
import torch.nn as nn
from torch.autograd import Variable

class Ibex_FANN(nn.Module):
    def __init__(self, mul_vals, shift_vals):
        super(Ibex_FANN, self).__init__()
        self.m0 = mul_vals[0]
        self.m1 = mul_vals[1]
        
        self.s0 = shift_vals[0] + 7
        self.s1 = shift_vals[1] + 7
        
        self.linear1 = nn.Linear(117, 20, bias = True)
        self.linear2 = nn.Linear(20, 2, bias = True)
        
    def forward(self, X, print_out = False):
        X = self.linear1(X)
        X = torch.mul(X, self.m0)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s0 - 1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s0).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)     
        
        X = self.linear2(X)
        X = torch.mul(X, self.m1)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s1 - 1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s1)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)
        
        if(print_out):
            print(X)
        
        return X

class Ibex_UCI_MLP(nn.Module):
    def __init__(self, mul_vals, shift_vals):
        super(Ibex_UCI_MLP, self).__init__()
        self.m0 = mul_vals[0]
        self.m1 = mul_vals[1]
        self.m2 = mul_vals[2]
        self.m3 = mul_vals[3]
        
        self.s0 = shift_vals[0] + 7
        self.s1 = shift_vals[1] + 7
        self.s2 = shift_vals[2] + 7
        self.s3 = shift_vals[3] + 7
        
        self.fc0 = nn.Linear(76, 300, bias = True)
        self.fc1 = nn.Linear(300, 200, bias = True)
        self.fc2 = nn.Linear(200, 100, bias = True)
        self.fc3 = nn.Linear(100, 10, bias = True)
        
    def forward(self, X, print_out = False):
        X = self.fc0(X)
        X = torch.mul(X, self.m0)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s0 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s0).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)        
        
        X = self.fc1(X)
        X = torch.mul(X, self.m1)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1),self.s1 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s1)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)
        
        X = self.fc2(X)
        X = torch.mul(X, self.m2)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1),self.s2 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s2)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)
        
        X = self.fc3(X)
        X = torch.mul(X, self.m3)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1),self.s3 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s3)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)

        if(print_out):
            print(X[0])
        
        return X

class Ibex_Lenet5(nn.Module):
    def __init__(self, mul_vals, shift_vals):
        super(Ibex_Lenet5, self).__init__()
        
        self.m0 = mul_vals[0]
        self.m1 = mul_vals[1]
        self.m2 = mul_vals[2]
        self.m3 = mul_vals[3]
        self.m4 = mul_vals[4]
        
        self.s0 = shift_vals[0] + 7
        self.s1 = shift_vals[1] + 7
        self.s2 = shift_vals[2] + 7
        self.s3 = shift_vals[3] + 7
        self.s4 = shift_vals[4] + 7
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding= 'same')
        
        self.avg1 = nn.AvgPool2d(2,2)
        
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        self.avg2 = nn.AvgPool2d(2,2)
        
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, X, print_out = False):
        X = self.conv1(X)
        
        X = torch.mul(X, self.m0)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s0 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s0).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)
        
        X = self.avg1(X).type(torch.LongTensor)
        X = X.type(torch.FloatTensor)
                
        X = self.conv2(X)
        X = torch.mul(X, self.m1)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s1 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s1).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)
        
        X = self.avg2(X).type(torch.LongTensor)
        X = X.type(torch.FloatTensor)
        X = X.reshape(X.shape[0], -1)
        
        X = self.fc1(X)
        X = torch.mul(X, self.m2)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s2 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s2).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)
                
        X = self.fc2(X)
        X = torch.mul(X, self.m3)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s3 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s3).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)
                
        X = self.fc3(X)
        X = torch.mul(X, self.m4)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s4 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s4).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255).type(torch.FloatTensor)

        if(print_out):
            print(X)

        return X

def create_fann_model(int_weights, int_biases, mul_vals, shift_vals):
    ibex_model = Ibex_FANN(mul_vals, shift_vals)
    ibex_model_dict = ibex_model.state_dict()

    ibex_model_dict['linear1.weight'] = torch.tensor(int_weights[0])
    ibex_model_dict['linear2.weight'] = torch.tensor(int_weights[1])

    ibex_model_dict['linear1.bias'] = torch.tensor(int_biases[0])
    ibex_model_dict['linear2.bias'] = torch.tensor(int_biases[1])

    ibex_model.load_state_dict(ibex_model_dict)
    return ibex_model

def create_uci_model(int_weights, int_biases, mul_vals, shift_vals):
    ibex_model = Ibex_UCI_MLP(mul_vals, shift_vals)
    ibex_model_dict = ibex_model.state_dict()

    ibex_model_dict['fc0.weight'] = torch.tensor(int_weights[0])
    ibex_model_dict['fc1.weight'] = torch.tensor(int_weights[1])
    ibex_model_dict['fc2.weight'] = torch.tensor(int_weights[2])
    ibex_model_dict['fc3.weight'] = torch.tensor(int_weights[3])

    ibex_model_dict['fc0.bias'] = torch.tensor(int_biases[0])
    ibex_model_dict['fc1.bias'] = torch.tensor(int_biases[1])
    ibex_model_dict['fc2.bias'] = torch.tensor(int_biases[2])
    ibex_model_dict['fc3.bias'] = torch.tensor(int_biases[3])

    ibex_model.load_state_dict(ibex_model_dict)

    return ibex_model

def create_lenet_model(int_weights, int_biases, mul_vals, shift_vals):
    ibex_model = Ibex_Lenet5(mul_vals, shift_vals)
    ibex_model_dict = ibex_model.state_dict()

    ibex_model_dict['conv1.weight'] = torch.tensor(int_weights[0])
    ibex_model_dict['conv2.weight'] = torch.tensor(int_weights[1])
    ibex_model_dict['fc1.weight'] = torch.tensor(int_weights[2])
    ibex_model_dict['fc2.weight'] = torch.tensor(int_weights[3])
    ibex_model_dict['fc3.weight'] = torch.tensor(int_weights[4])

    ibex_model_dict['conv1.bias'] = torch.tensor(int_biases[0])
    ibex_model_dict['conv2.bias'] = torch.tensor(int_biases[1])
    ibex_model_dict['fc1.bias'] = torch.tensor(int_biases[2])
    ibex_model_dict['fc2.bias'] = torch.tensor(int_biases[3])
    ibex_model_dict['fc3.bias'] = torch.tensor(int_biases[4])

    ibex_model.load_state_dict(ibex_model_dict)

    return ibex_model

def eval_sim_model(quant_model, ibex_model, test_loader):
    # Turn off gradients for validation
    with torch.no_grad():
        ibex_model.eval()
        correct = 0 
        y_size = 0
        for test_imgs, test_labels in test_loader:
            test_imgs = torch.round(Variable(test_imgs).float()/quant_model.quant_inp.quant_act_scale().cpu())
            output = ibex_model(test_imgs)
            predicted = torch.max(output, 1)[1]
            correct += (predicted == test_labels).sum()
            y_size += len(test_labels)
        print("Test accuracy: {:.3f}% ".format(100*float(correct)/y_size))
    
    print(ibex_model(torch.unsqueeze(test_imgs[0], dim = 0)))
    return
