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

class Ibex_CMSIS_CNN(nn.Module):
    def __init__(self, mul_vals, shift_vals):
        super(Ibex_CMSIS_CNN, self).__init__()
        
        self.m0 = mul_vals[0]
        self.m1 = mul_vals[1]
        self.m2 = mul_vals[2]
        self.m3 = mul_vals[3]
        
        self.s0 = shift_vals[0] + 7
        self.s1 = shift_vals[1] + 7
        self.s2 = shift_vals[2] + 7
        self.s3 = shift_vals[3] + 7
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, padding = 'same')
        self.max1 = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, padding = 'same')
        self.max2 = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 'same')
        self.max3 = nn.MaxPool2d(2,2)
        
        self.linear1 = nn.Linear(1024, 10)
        
    def forward(self, X, print_out = False):
        
        X = self.conv1(X)
        X = torch.mul(X, self.m0)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s0 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s0).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255)
        
        X = self.max1(X)
        
        X = self.conv2(X)
        X = torch.mul(X, self.m1)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s1 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s1).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255)
        
        X = self.max2(X)
        
        X = self.conv3(X)
        X = torch.mul(X, self.m2)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s2 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s2).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255)
        
        X = self.max3(X)
        
        X = X.reshape(X.shape[0], -1)
        X = self.linear1(X)
        X = torch.mul(X, self.m3)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s3 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s3).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255)

        if(print_out):
            print(X)
        return X

class Ibex_DepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mul_vals, shift_vals):
        super(Ibex_DepthwiseBlock, self).__init__()
                            
        self.dw = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, 
                                    kernel_size = 3, padding = 1, groups = in_channels)
                
        self.pw = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                                    kernel_size = 1, padding = 0)
            
        self.m0 = mul_vals[0]
        self.m1 = mul_vals[1]
        
        self.s0 = shift_vals[0] + 7
        self.s1 = shift_vals[1] + 7
        
    def forward(self, X):
        X = self.dw(X)
        X = torch.mul(X, self.m0)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s0 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s0).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255)
        
        X = self.pw(X)
        X = torch.mul(X, self.m1)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s1 -1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s1).type(torch.FloatTensor)
        X = torch.clamp(X, min = 0, max = 255)
        
        return X
    
class Ibex_Cifar10_Dws_CNN(nn.Module):
    def __init__(self, mul_vals, shift_vals):
        super(Ibex_Cifar10_Dws_CNN, self).__init__()
        self.features = nn.Sequential(
            Ibex_DepthwiseBlock(3, 64, mul_vals[0:2], shift_vals[0:2]),
            Ibex_DepthwiseBlock(64, 64, mul_vals[2:4], shift_vals[2:4]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            Ibex_DepthwiseBlock(64, 128, mul_vals[4:6], shift_vals[4:6]),
            Ibex_DepthwiseBlock(128, 128, mul_vals[6:8], shift_vals[6:8]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            Ibex_DepthwiseBlock(128, 256, mul_vals[8:10], shift_vals[8:10]),
            Ibex_DepthwiseBlock(256, 256, mul_vals[10:12], shift_vals[10:12]),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 10)
        )
        
        self.m_cl = mul_vals[12]
        self.s_cl = shift_vals[12] + 7

    def forward(self, x, print_out = False):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        x = torch.mul(x, self.m_cl)
        x = torch.add(x, torch.bitwise_left_shift(torch.tensor(1), self.s_cl - 1)).type(torch.LongTensor)
        x = torch.bitwise_right_shift(x, self.s_cl).type(torch.FloatTensor)
        x = torch.clamp(x, min = 0, max = 255)
        
        if(print_out):
            print(x)
            
        return x

class Ibex_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mul_val, shift_val,
                 stride = 1, act = 'relu', padding = 0, groups = 1):
        
        super(Ibex_Conv_Block, self).__init__()

        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                             kernel_size = kernel_size, stride = stride, padding = padding,
                             groups = groups, bias = True)
        
        self.act = act

        self.m = mul_val
        
        self.s = shift_val + 7

    def forward(self, x):
        X = self.conv(x)
        X = torch.mul(X, self.m)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s-1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s).type(torch.FloatTensor)
        if(self.act == 'relu'):
            X = torch.clamp(X, min = 0, max = 255)
        else:
            X = torch.clamp(X, min = -128, max = 127)
        return X

class Ibex_MBInvertedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, out_channels_inv, kernel_size, 
                 mul_vals, shift_vals, inverted_bottleneck = False, stride = 1, padding = 0):
        
        super(Ibex_MBInvertedConvLayer, self).__init__()
        
        self.inverted_bottleneck_bool = inverted_bottleneck
        
        if(inverted_bottleneck):
            self.inverted_bottleneck = Ibex_Conv_Block(
                in_channels = in_channels,
                out_channels = out_channels_inv,
                kernel_size = 1,
                stride = 1,
                mul_val = mul_vals[0],
                shift_val = shift_vals[0]
            )
        
            self.depth_conv = Ibex_Conv_Block(
                in_channels = out_channels_inv,
                out_channels = out_channels_inv,
                groups = out_channels_inv,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                mul_val = mul_vals[1],
                shift_val = shift_vals[1]
            )
        
            self.point_linear = Ibex_Conv_Block(
                in_channels = out_channels_inv,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                act = 'not_relu',
                mul_val = mul_vals[2],
                shift_val = shift_vals[2]
            )
            
        else:
            self.depth_conv = Ibex_Conv_Block(
                in_channels = in_channels,
                out_channels = in_channels,
                groups = in_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                mul_val = mul_vals[0],
                shift_val = shift_vals[0]
            )   
        
            self.point_linear = Ibex_Conv_Block(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                act = 'not_relu',
                mul_val = mul_vals[1],
                shift_val = shift_vals[1]
            )
        
    def forward(self, x):
        if(self.inverted_bottleneck_bool):
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        
        x = self.point_linear(x)
        return x

class Ibex_MobileInvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_channels_inv, kernel_size, 
                 stride, padding, mul_vals, shift_vals, act = 'not_relu',
                 inverted_bottleneck = False, shortcut = False):
        
        super(Ibex_MobileInvertedResidualBlock, self).__init__()
        
        self.mobile_inverted_conv = Ibex_MBInvertedConvLayer(
            in_channels = in_channels,
            out_channels = out_channels,
            out_channels_inv = out_channels_inv,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            mul_vals = mul_vals,
            shift_vals = shift_vals,
            inverted_bottleneck = inverted_bottleneck,
        )
        
        self.act = act
        self.shortcut_bool = shortcut
        if(shortcut):
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        res = self.mobile_inverted_conv(x)        
        if(self.shortcut_bool):
            res += x
        
        if(self.act == 'relu'):
            res = torch.clamp(res, min = 0, max = 255)
        else:
            res = torch.clamp(res, min = -128, max = 127)

        return res

class Ibex_Linear_Layer(nn.Module):
    def __init__(self,in_features, out_features, 
                 mul_val, shift_val, bias = True, act = 'relu'):
        
        super(Ibex_Linear_Layer, self).__init__()
        
        self.s = shift_val + 7
        self.m = mul_val
        
        self.linear = nn.Linear(
            in_features = in_features,
            out_features = out_features,
            bias = bias
        )
        
        self.act = act
        
    def forward(self, x):
        X = self.linear(x)
        X = torch.mul(X, self.m)
        X = torch.add(X, torch.bitwise_left_shift(torch.tensor(1), self.s - 1)).type(torch.LongTensor)
        X = torch.bitwise_right_shift(X, self.s).type(torch.FloatTensor)
        if(self.act == 'relu'):
            X = torch.clamp(X, min = 0, max = 255)
        else:
            X = torch.clamp(X, min = -128, max = 127)
        return X

class Ibex_ProxylessNASNets(nn.Module):
    def __init__(self, ch_in, n_classes, mul_vals, shift_vals):
        super(Ibex_ProxylessNASNets, self).__init__()
        
        self.first_conv = Ibex_Conv_Block(
            in_channels = ch_in,
            out_channels = 16,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            mul_val = mul_vals[0],
            shift_val = shift_vals[0]
        )

        layers = []
        
        #0
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 16,
            out_channels = 8,
            out_channels_inv = 0,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            mul_vals = mul_vals[1:3],
            shift_vals = shift_vals[1:3],
        ))
        
        #1
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 8,
            out_channels = 16,
            out_channels_inv = 48,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            inverted_bottleneck = True,
            shortcut = False,
            mul_vals = mul_vals[3:6],
            shift_vals = shift_vals[3:6]
        ))
        
        #2
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 16,
            out_channels = 16,
            out_channels_inv = 48,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            inverted_bottleneck = True,
            shortcut = True,
            mul_vals = mul_vals[6:9],
            shift_vals = shift_vals[6:9],
        ))
        
        #3
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 16,
            out_channels = 16,
            out_channels_inv = 48,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            inverted_bottleneck = True,
            shortcut = True,
            mul_vals = mul_vals[9:12],
            shift_vals = shift_vals[9:12],
        ))
        
        #4
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 16,
            out_channels = 24,
            out_channels_inv = 48,
            kernel_size = 7,
            inverted_bottleneck = True,
            stride = 2,
            padding = 3,
            shortcut = False,
            mul_vals = mul_vals[12:15],
            shift_vals = shift_vals[12:15],
        ))
        
        #5
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 24,
            out_channels = 24,
            out_channels_inv = 144,
            kernel_size = 3,
            inverted_bottleneck = True,
            stride = 1,
            padding = 1,
            shortcut = True,
            mul_vals = mul_vals[15:18],
            shift_vals = shift_vals[15:18]
        ))
        
        #6
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 24,
            out_channels = 24,
            out_channels_inv = 120,
            kernel_size = 5,
            inverted_bottleneck = True,
            stride = 1,
            padding = 2,
            shortcut = True,
            mul_vals = mul_vals[18:21],
            shift_vals = shift_vals[18:21]
        ))
        
        #7
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 24,
            out_channels = 40,
            out_channels_inv = 144,
            kernel_size = 7,
            inverted_bottleneck = True,
            stride = 2,
            padding = 3,
            shortcut = False,
            mul_vals = mul_vals[21:24],
            shift_vals = shift_vals[21:24]
        ))
        
        #8
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 40,
            out_channels = 40,
            out_channels_inv = 240,
            kernel_size = 7,
            inverted_bottleneck = True,
            stride = 1,
            padding = 3,
            shortcut = True,
            mul_vals = mul_vals[24:27],
            shift_vals = shift_vals[24:27]
        ))
        
        #9
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 40,
            out_channels = 48,
            out_channels_inv = 240,
            kernel_size = 3,
            inverted_bottleneck = True,
            stride = 1,
            padding = 1,
            shortcut = False,
            mul_vals = mul_vals[27:30],
            shift_vals = shift_vals[27:30],
        ))
        
        #10
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 48,
            out_channels = 48,
            out_channels_inv = 192,
            kernel_size = 3,
            inverted_bottleneck = True,
            stride = 1,
            padding = 1,
            shortcut = True,
            mul_vals = mul_vals[30:33],
            shift_vals = shift_vals[30:33]
        ))
        
        #11
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 48,
            out_channels = 96,
            out_channels_inv = 240,
            kernel_size = 5,
            inverted_bottleneck = True,
            stride = 2,
            padding = 2,
            shortcut = False,
            mul_vals = mul_vals[33:36],
            shift_vals = shift_vals[33:36]
        ))
        
        #12
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 96,
            out_channels = 96,
            out_channels_inv = 480,
            kernel_size = 3,
            inverted_bottleneck = True,
            stride = 1,
            padding = 1,
            shortcut = True,
            mul_vals = mul_vals[36:39],
            shift_vals = shift_vals[36:39],
        ))
        
        #13
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 96,
            out_channels = 96,
            out_channels_inv = 384,
            kernel_size = 3,
            inverted_bottleneck = True,
            stride = 1,
            padding = 1,
            shortcut = True,
            mul_vals = mul_vals[39:42],
            shift_vals = shift_vals[39:42]
        ))
        
        #14
        layers.append(Ibex_MobileInvertedResidualBlock(
            in_channels = 96,
            out_channels = 160,
            out_channels_inv = 288,
            kernel_size = 7,
            inverted_bottleneck = True,
            stride = 1,
            padding = 3,
            shortcut = False,
            mul_vals = mul_vals[42:45],
            shift_vals = shift_vals[42:45],
        ))
        
        self.blocks = nn.Sequential(*layers)

        # Output Block
        self.classifier = Ibex_Linear_Layer(
            in_features = 160, 
            out_features = n_classes,
            mul_val = mul_vals[45],
            shift_val = shift_vals[45]
        )

    def forward(self, x, print_out = False):
        x = self.first_conv(x)
        
        x = self.blocks(x)
        x = x.mean(3).mean(2).type(torch.LongTensor)
        
        x = x.view(x.size(0), -1).type(torch.FloatTensor)        
        x = self.classifier(x)

        if(print_out):
            print(x)
            
        return x

def configure_network(ibex_model_dict, int_weights, int_biases):
    for i, (name, _) in enumerate(ibex_model_dict.items()):
        if(i%2 == 0):
            ibex_model_dict[name] =  torch.tensor(int_weights[i//2])
        else:
            ibex_model_dict[name] = torch.tensor(int_biases[i//2])

    return ibex_model_dict

def create_fann_model(int_weights, int_biases, mul_vals, shift_vals):
    ibex_model = Ibex_FANN(mul_vals, shift_vals)
    ibex_model_dict = ibex_model.state_dict()

    ibex_model_dict = configure_network(ibex_model_dict, int_weights, int_biases)

    ibex_model.load_state_dict(ibex_model_dict)
    return ibex_model

def create_uci_model(int_weights, int_biases, mul_vals, shift_vals):
    ibex_model = Ibex_UCI_MLP(mul_vals, shift_vals)
    ibex_model_dict = ibex_model.state_dict()

    ibex_model_dict = configure_network(ibex_model_dict, int_weights, int_biases)

    ibex_model.load_state_dict(ibex_model_dict)

    return ibex_model

def create_lenet_model(int_weights, int_biases, mul_vals, shift_vals):
    ibex_model = Ibex_Lenet5(mul_vals, shift_vals)
    ibex_model_dict = ibex_model.state_dict()

    ibex_model_dict = configure_network(ibex_model_dict, int_weights, int_biases)

    ibex_model.load_state_dict(ibex_model_dict)

    return ibex_model

def create_cmsis_cnn_model(int_weights, int_biases, mul_vals, shift_vals):
    ibex_model = Ibex_CMSIS_CNN(mul_vals, shift_vals)
    ibex_model_dict = ibex_model.state_dict()

    ibex_model_dict = configure_network(ibex_model_dict, int_weights, int_biases)

    ibex_model.load_state_dict(ibex_model_dict)

    return ibex_model

def create_ibex_dws_model(int_weights, int_biases, mul_vals, shift_vals):
    ibex_model = Ibex_Cifar10_Dws_CNN(mul_vals, shift_vals)
    ibex_model_dict = ibex_model.state_dict()

    ibex_model_dict = configure_network(ibex_model_dict, int_weights, int_biases)
    
    ibex_model.load_state_dict(ibex_model_dict)
    
    return ibex_model

def create_mcunet_model(int_weights, int_biases, mul_vals, shift_vals):
    ibex_model = Ibex_ProxylessNASNets(ch_in = 3, n_classes = 2, 
                                       mul_vals = mul_vals, 
                                       shift_vals = shift_vals)
        
    ibex_model_dict = ibex_model.state_dict()

    ibex_model_dict = configure_network(ibex_model_dict, int_weights, int_biases)
    
    ibex_model.load_state_dict(ibex_model_dict)
    
    return ibex_model

def eval_sim_model(quant_model, ibex_model, test_loader):
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
