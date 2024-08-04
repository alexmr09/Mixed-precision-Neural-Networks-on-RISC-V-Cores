import torch.nn as nn
from brevitas.nn.utils import merge_bn
import torch
from mpq_quantize import extract_params
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, apply_act = True,
                            padding = 0, groups = 1):
        
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                             kernel_size = kernel_size, stride = stride, padding = padding,
                             groups = groups, bias = True)
        
        if(apply_act):
            self.act = nn.ReLU6(inplace = True)

        else: 
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x
    
class MBInvertedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, out_channels_inv, kernel_size, 
                 inverted_bottleneck = False, stride = 1, padding = 0):
        
        super(MBInvertedConvLayer, self).__init__()
        
        layers = []

        if(inverted_bottleneck):
            layers.append(ConvLayer(
                in_channels = in_channels,
                out_channels = out_channels_inv,
                kernel_size = 1,
                stride = 1
            ))
        
            layers.append(ConvLayer(
                in_channels = out_channels_inv,
                out_channels = out_channels_inv,
                groups = out_channels_inv,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
            ))
        
            layers.append(ConvLayer(
                in_channels = out_channels_inv,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                apply_act = False
            ))
            
        else:
            layers.append(ConvLayer(
                in_channels = in_channels,
                out_channels = in_channels,
                groups = in_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
            ))
        
            layers.append(ConvLayer(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                apply_act = False
            ))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class MobileInvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_channels_inv, kernel_size, 
                 stride, padding, inverted_bottleneck = False, shortcut = False):
        
        super(MobileInvertedResidualBlock, self).__init__()
        
        self.mobile_inverted_conv = MBInvertedConvLayer(
            in_channels = in_channels,
            out_channels = out_channels,
            out_channels_inv = out_channels_inv,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            inverted_bottleneck = inverted_bottleneck
        )
        
        self.shortcut = shortcut
        
    def forward(self, x):
        res = self.mobile_inverted_conv(x)
        if(self.shortcut):
            res += x
        return res
    
class ProxylessNASNets_No_Bn(nn.Module):
    def __init__(self, ch_in = 3, n_classes = 2):
        super(ProxylessNASNets_No_Bn, self).__init__()
        
        self.first_conv = nn.Sequential(
            ConvLayer(
                in_channels = ch_in,
                out_channels = 16,
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        )
        
        layers = []

        #0
        layers.append(MobileInvertedResidualBlock(
            in_channels = 16,
            out_channels = 8,
            out_channels_inv = 0,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ))
        
        #1
        layers.append(MobileInvertedResidualBlock(
            in_channels = 8,
            out_channels = 16,
            out_channels_inv = 48,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            inverted_bottleneck = True,
            shortcut = False
        ))
        
        #2
        layers.append(MobileInvertedResidualBlock(
            in_channels = 16,
            out_channels = 16,
            out_channels_inv = 48,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            inverted_bottleneck = True,
            shortcut = True
        ))
        
        #3
        layers.append(MobileInvertedResidualBlock(
            in_channels = 16,
            out_channels = 16,
            out_channels_inv = 48,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            inverted_bottleneck = True,
            shortcut = True
        ))
        
        #4
        layers.append(MobileInvertedResidualBlock(
            in_channels = 16,
            out_channels = 24,
            out_channels_inv = 48,
            kernel_size = 7,
            inverted_bottleneck = True,
            stride = 2,
            padding = 3,
            shortcut = False
        ))
        
        #5
        layers.append(MobileInvertedResidualBlock(
            in_channels = 24,
            out_channels = 24,
            out_channels_inv = 144,
            kernel_size = 3,
            inverted_bottleneck = True,
            stride = 1,
            padding = 1,
            shortcut = True
        ))
        
        #6
        layers.append(MobileInvertedResidualBlock(
            in_channels = 24,
            out_channels = 24,
            out_channels_inv = 120,
            kernel_size = 5,
            inverted_bottleneck = True,
            stride = 1,
            padding = 2,
            shortcut = True
        ))
        
        #7
        layers.append(MobileInvertedResidualBlock(
            in_channels = 24,
            out_channels = 40,
            out_channels_inv = 144,
            kernel_size = 7,
            inverted_bottleneck = True,
            stride = 2,
            padding = 3,
            shortcut = False
        ))
        
        #8
        layers.append(MobileInvertedResidualBlock(
            in_channels = 40,
            out_channels = 40,
            out_channels_inv = 240,
            kernel_size = 7,
            inverted_bottleneck = True,
            stride = 1,
            padding = 3,
            shortcut = True
        ))
        
        #9
        layers.append(MobileInvertedResidualBlock(
            in_channels = 40,
            out_channels = 48,
            out_channels_inv = 240,
            kernel_size = 3,
            inverted_bottleneck = True,
            stride = 1,
            padding = 1,
            shortcut = False
        ))
        
        #10
        layers.append(MobileInvertedResidualBlock(
            in_channels = 48,
            out_channels = 48,
            out_channels_inv = 192,
            kernel_size = 3,
            inverted_bottleneck = True,
            stride = 1,
            padding = 1,
            shortcut = True
        ))
        
        #11
        layers.append(MobileInvertedResidualBlock(
            in_channels = 48,
            out_channels = 96,
            out_channels_inv = 240,
            kernel_size = 5,
            inverted_bottleneck = True,
            stride = 2,
            padding = 2,
            shortcut = False
        ))
        
        #12
        layers.append(MobileInvertedResidualBlock(
            in_channels = 96,
            out_channels = 96,
            out_channels_inv = 480,
            kernel_size = 3,
            inverted_bottleneck = True,
            stride = 1,
            padding = 1,
            shortcut = True
        ))
        
        #13
        layers.append(MobileInvertedResidualBlock(
            in_channels = 96,
            out_channels = 96,
            out_channels_inv = 384,
            kernel_size = 3,
            inverted_bottleneck = True,
            stride = 1,
            padding = 1,
            shortcut = True
        ))
        
        #14
        layers.append(MobileInvertedResidualBlock(
            in_channels = 96,
            out_channels = 160,
            out_channels_inv = 288,
            kernel_size = 7,
            inverted_bottleneck = True,
            stride = 1,
            padding = 3,
            shortcut = False
        ))

        self.features = nn.Sequential(*layers)
        
        self.avg = nn.AvgPool2d(kernel_size = 3, stride = 1)

        self.flatten = nn.Flatten()
        
        # Output Block
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features = 160, 
                out_features = n_classes
            )
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.avg(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim = 1)

def merge_bn_network(bn_net, no_bn_net):
    for _, module in bn_net.named_modules():
        if hasattr(module, 'conv') and hasattr(module, 'bn'):
            merge_bn(module.conv, module.bn)

    weights, biases = extract_params(bn_net)
    
    custom_model_dict = no_bn_net.state_dict()

    for i, (name, _) in enumerate(custom_model_dict.items()):
        if(i%2 == 0):
            custom_model_dict[name] =  torch.tensor(weights[i//2])
        else:
            custom_model_dict[name] = torch.tensor(biases[i//2])
    
    no_bn_net.load_state_dict(custom_model_dict)

    return no_bn_net

