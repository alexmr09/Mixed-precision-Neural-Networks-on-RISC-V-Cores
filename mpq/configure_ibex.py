import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import shutil

def quantize_multiplier(real_multiplier):
    s = 0
    while real_multiplier < 0.5:
        real_multiplier *= 2.0
        s += 1

    q = int(round(real_multiplier * (1 << 7)))

    # Handle the special case when the real multiplier was so close to 1
    # that its fixed-point approximation was undistinguishable from 1.
    # We handle this by dividing it by two, and remembering to decrement
    # the right shift amount.

    if q == (1 << 7):
        q //= 2
        s -= 1

    quantized_multiplier = int(q)
    right_shift = s

    return quantized_multiplier, right_shift

def get_int_params(quant_net):
    
    int_weights = []
    int_bias = []
    in_scales = []
    act_scales = []
    
    def extract_quant_params(module):
        for name, submodule in module.named_children():
            # Check if the submodule has weights and append them if present
            if hasattr(submodule, 'weight') and submodule.weight is not None:
                int_weights.append(submodule.int_weight().cpu().detach().numpy())
                int_bias.append(submodule.int_bias().cpu().detach().numpy())
                in_scales.append(submodule.quant_bias_scale().cpu().detach().numpy())

            # Check if the submodule has activation scale and append it if present
            if hasattr(submodule, 'quant_act_scale') and submodule.quant_act_scale is not None:
                act_scales.append(submodule.quant_act_scale().cpu().detach().numpy())

            # Recursively extract parameters from the children modules
            extract_quant_params(submodule)

    # Start extraction from the top-level module
    extract_quant_params(quant_net)
    
    mul_vals, shift_vals = [], []
    
    for i in range(len(act_scales)-1):
        M = in_scales[i]/act_scales[i+1]
        mul, shift = quantize_multiplier(M[0])
        mul_vals.append(mul)
        shift_vals.append(shift)
      
    int_biases = []
    f_int_biases = []
    shift_biases = []

    for int_b in int_bias:
        shift_bias = np.clip(np.log2(abs(int_b + 1e-10)).astype(np.int32) - 6, a_max = None, a_min = 0)
        r_bias = np.right_shift(int_b, shift_bias)
        f_int_biases.append(r_bias)
        l_bias = np.left_shift(r_bias, shift_bias)
        shift_biases.append(shift_bias)
        int_biases.append(l_bias)

    return int_weights, int_biases, f_int_biases, shift_biases, mul_vals, shift_vals

def decide_mode(network, weight_bit_width, input_uint8 = True):

    VALID_BIT_WIDTH_VALS = {2, 4, 8}
    
    # Checking the values of the arrays
    unique_weight_bit_width = np.unique(weight_bit_width)
    
    for i in range(len(unique_weight_bit_width)):
        if unique_weight_bit_width[i] not in VALID_BIT_WIDTH_VALS:
            raise ValueError("Wrong bit width selected {0}. Please choose values 2, 4 or 8".format(unique_weight_bit_width[i]))
    
    input_sign = [int(not input_uint8)] + (len(weight_bit_width)-1)*[1]
    ins = 1

    mode_per_layer = []
    layer_type = []

    layer_types_py = tuple(cls for name, cls in nn.__dict__.items() if isinstance(cls, type) and issubclass(cls, nn.Module))

    for name, module in network.named_modules():
        if isinstance(module, layer_types_py):
            layer_type_name = module.__class__.__name__
            if(layer_type_name == 'Linear'):
                layer_type.append(layer_type_name)
            if(layer_type_name == 'Conv2d'):
                if(module.groups == module.in_channels and module.groups != 1):
                    layer_type.append('DepthwiseConv2d')
                else:
                    layer_type.append(layer_type_name)
            else:
                if(layer_type_name == 'ReLU' or layer_type_name == 'Sigmoid'):
                    input_sign[ins] = 0
                    ins += 1
        
    for i in range(len(weight_bit_width)):
        signed_input = 4 * input_sign[i]
        if(layer_type[i] == 'DepthwiseConv2d'):
                mode_per_layer.append(signed_input + 1)
        else:
            if(weight_bit_width[i] == 2):
                mode_per_layer.append(signed_input + 3)
            elif(weight_bit_width[i] == 4):
                mode_per_layer.append(signed_input + 2)
            else:
                mode_per_layer.append(signed_input)

    return mode_per_layer, layer_type

def pad_inputs_weights(quant_net, test_loader, mode_per_layer, 
                       int_weights, int_biases, shift_biases,
                       mul_vals, shift_vals):
    
    for test_imgs, _ in test_loader:
        t = (torch.round(Variable(test_imgs).float()/quant_net.quant_inp.quant_act_scale().cpu()))
        t = t.detach().cpu().numpy().astype(np.int16)[0]

    isPaddingNeeded = t.shape[0]%4
    a = t.shape[0]//4
    if(isPaddingNeeded != 0):
        new_size = (a+1)*4
    else:
        new_size = a*4

    new_shape = [new_size]
    for sh in np.shape(t)[1:]:
        new_shape.append(sh)

    new_shape = tuple(new_shape)

    padded_input = np.zeros(shape = new_shape).astype(np.int16)
    padded_input[:t.shape[0], ...] = t

    padded_int_weights = []

    for i, w in enumerate(int_weights):
        if(len(w.shape) == 2):
            nodes_per_layer = w.shape[0]
            a = nodes_per_layer//4
        
            if(nodes_per_layer%4 != 0):
                new_size_0 = (a+1)*4
            else:
                new_size_0 = a*4
        
            if(i == 0):
                new_size_1 = padded_input.shape[0]
                new_w = np.zeros((new_size_0, new_size_1)).astype(np.int8)
                new_w[:w.shape[0], :w.shape[1]] = w
            else:
                new_w = np.zeros((new_size_0, w.shape[1])).astype(np.int8)
                new_w[:w.shape[0], :] = w    
    
        elif(len(w.shape) == 4):
            filters_per_layer = w.shape[0]
            a = filters_per_layer//4
        
            if(filters_per_layer % 4 != 0):
                new_size_0 = (a + 1) * 4
            else:
                new_size_0 = a * 4
        
            if((mode_per_layer[i] != 1) and (mode_per_layer[i] != 5)):
                b = w.shape[1] // 4
                if(w.shape[1] % 4 != 0):
                    new_size_1 = (b + 1) * 4
                else:
                    new_size_1 = b * 4
                    
                new_w = np.zeros((new_size_0, new_size_1, w.shape[2], w.shape[3])).astype(np.int8)
                new_w[:w.shape[0], :w.shape[1], :, :] = w
            
            else:
                new_size_1 = 1
                new_w = np.zeros((new_size_0, new_size_1, w.shape[2], w.shape[3])).astype(np.int8)
                new_w[:w.shape[0], :w.shape[1], :, :] = w
                new_w = np.squeeze(new_w, axis = 1)
                
        padded_int_weights.append(new_w)

    padded_int_biases = []

    for i, b in enumerate(int_biases):
        nodes_per_layer = b.shape[0]
        a = nodes_per_layer//4
        if(nodes_per_layer%4 != 0):
            new_size_b = (a+1)*4
        else:
            new_size_b = a*4
        new_b = np.zeros(new_size_b).astype(np.int8)
        new_b[:b.shape[0]] = b
        padded_int_biases.append(new_b)

    padded_shift_biases = []

    for i, b in enumerate(shift_biases):
        nodes_per_layer = b.shape[0]
        a = nodes_per_layer//4
        if(nodes_per_layer%4 != 0):
            new_size_b = (a+1)*4
        else:
            new_size_b = a*4
        new_b = np.zeros(new_size_b).astype(np.int8)
        new_b[:b.shape[0]] = b
        padded_shift_biases.append(new_b)

    padded_mul_vals = []

    for i, mul_v in enumerate(mul_vals):
        m = np.array(mul_v)
        if(len(np.shape(m)) > 0):
            nodes_per_layer = m.shape[0]
            a = nodes_per_layer//4
            if(nodes_per_layer%4 != 0):
                new_size_m = (a+1)*4
            else:
                new_size_m = a*4
            new_m = np.zeros(new_size_m).astype(np.int8)
            new_m[:m.shape[0]] = m
            padded_mul_vals.append(new_m)
        else:
            padded_mul_vals.append(mul_v)

    padded_shift_vals = []

    for i, sh_v in enumerate(shift_vals):
        s = np.array(sh_v)
        if(len(np.shape(s)) > 0):
            nodes_per_layer = s.shape[0]
            a = nodes_per_layer//4
            if(nodes_per_layer%4 != 0):
                new_size_s = (a+1)*4
            else:
                new_size_s = a*4
            new_s = np.zeros(new_size_s).astype(np.int8)
            new_s[:s.shape[0]] = m
            padded_shift_vals.append(new_s)
        else:
            padded_shift_vals.append(sh_v)
    
    t = np.expand_dims(t, axis = 0)

    return t, padded_input, padded_int_weights, padded_int_biases, padded_shift_biases, padded_mul_vals, padded_shift_vals

def combine_values(vec):
    combined_value = 0

    dims = np.shape(vec)
    if(dims == (2,2)):
        vec = [vec[0][0], vec[1][0], vec[0][1], vec[1][1]]

    elif(dims == (2,4)):
        vec = [vec[0][0], vec[1][0], vec[0][1], vec[1][1],
               vec[0][2], vec[1][2], vec[0][3], vec[1][3]]

    elif(dims == (4,2)):
        vec = [vec[0][0], vec[1][0], vec[2][0], vec[3][0],
               vec[0][1], vec[1][1], vec[2][1], vec[3][1]]

    elif(dims == (4,4)):
        vec = [vec[0][0], vec[1][0], vec[2][0], vec[3][0],
               vec[0][1], vec[1][1], vec[2][1], vec[3][1],
               vec[0][2], vec[1][2], vec[2][2], vec[3][2],
               vec[0][3], vec[1][3], vec[2][3], vec[3][3]]

    if len(vec) not in [1, 2, 4, 8, 16]:
        raise ValueError("The input vector 'a' must have 1, 2, 4, 8 or 16 values")
        
    else:
        div_s = int(32/len(vec))
        keep_lsb = (1 << div_s) - 1
        for value in vec:
            value = int(value)
            combined_value = (combined_value << div_s) | (value & keep_lsb)
            
    return combined_value

def concat_inputs_weights(mode_per_layer, padded_input, padded_int_weights, padded_int_biases, 
                          padded_shift_biases, padded_mul_vals, padded_shift_vals):

    padded_input = np.expand_dims(padded_input, axis = 0)
    combined_input_data = []

    if(len(np.shape(padded_input)) == 2):
        for data in padded_input:
            size = len(data)
            new_mat = np.zeros(int(size//4), dtype = np.int64)
            for i in range(int(size//4)):
                vector = data[4*i : 4*(i+1)]
                comb = combine_values(vector)
                new_mat[i] = comb
            combined_input_data.append(new_mat)
    else:
        for data in padded_input:
            new_mat = np.zeros((data.shape[0] // 4, data.shape[1], data.shape[2]), dtype = np.int64)
            for i in range(data.shape[0]//4):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        vector = data[4 * i : 4 * (i + 1), j, k]
                        comb = combine_values(vector)
                        new_mat[i][j][k] = comb
            combined_input_data.append(new_mat)
    
    new_int_weights = []

    for iter, layer_weight in enumerate(padded_int_weights):
        dims = layer_weight.shape

        if(len(dims) == 2):
            if((mode_per_layer[iter] == 0) | (mode_per_layer[iter] == 4)): 
                new_mat = np.zeros((int(dims[0]/4), int(dims[1])), dtype = np.int64)
                for i in range(int(dims[0]/4)):
                    for j in range(int(dims[1])):
                        vector = layer_weight[4*i : 4*(i+1),j]
                        comb = combine_values(vector)
                        new_mat[i][j] = comb

            elif((mode_per_layer[iter] == 2) | (mode_per_layer[iter] == 6)):
                new_mat = np.zeros((int(dims[0]/4), int(dims[1]/2)), dtype = np.int64)
                for i in range(int(dims[0]/4)):
                    for j in range(int(dims[1]/2)):
                        vector = layer_weight[4*i : 4*(i+1), 2*j : 2*(j+1)]
                        comb = combine_values(vector)
                        new_mat[i][j] = comb

            elif((mode_per_layer[iter] == 3) | (mode_per_layer[iter] == 7)): 
                new_mat = np.zeros((int(dims[0]/4), int(dims[1]/4)), dtype = np.int64)
                for i in range(int(dims[0]/4)):
                    for j in range(int(dims[1]/4)):
                        vector = layer_weight[4*i : 4*(i+1), 4*j : 4*(j+1)]
                        comb = combine_values(vector)
                        new_mat[i][j] = comb

        elif(len(dims) == 3):
            new_mat = np.zeros((int(dims[0]//4), dims[1], dims[2]), dtype = np.int64)
            for i in range(int(dims[0]//4)):
                    for j in range(dims[1]):
                        for k in range(dims[2]):
                            vector = layer_weight[4*i : 4*(i+1), j, k]
                            comb = combine_values(vector)
                            new_mat[i][j][k] = comb
                            
        elif(len(dims) == 4):
            if((mode_per_layer[iter] == 0) | (mode_per_layer[iter] == 4)):
                new_mat = np.zeros((int(dims[0]//4), dims[1], dims[2], dims[3]), dtype = np.int64)
                for i in range(int(dims[0]//4)):
                    for j in range(dims[1]):
                        for k in range(dims[2]):
                            for l in range(dims[3]):
                                vector = layer_weight[4*i : 4*(i+1), j, k, l]
                                comb = combine_values(vector)
                                new_mat[i][j][k][l] = comb

            elif((mode_per_layer[iter] == 2) | (mode_per_layer[iter] == 6)):
                new_mat = np.zeros((int(dims[0]//4), int(dims[1]//2), dims[2], dims[3]), dtype = np.int64)
                for i in range(int(dims[0]//4)):
                    for j in range(int(dims[1]//2)):
                        for k in range(dims[2]):
                            for l in range(dims[3]):
                                vector = layer_weight[4*i : 4*(i+1), 2*j : 2*(j+1), k, l]
                                comb = combine_values(vector)
                                new_mat[i][j][k][l] = comb
                            
            elif((mode_per_layer[iter] == 3) | (mode_per_layer[iter] == 7)):
                new_mat = np.zeros((int(dims[0]//4), int(dims[1]//4), dims[2], dims[3]), dtype = np.int64)
                for i in range(int(dims[0]//4)):
                    for j in range(int(dims[1]//4)):
                        for k in range(dims[2]):
                            for l in range(dims[3]):
                                vector = layer_weight[4*i : 4*(i+1), 4*j : 4*(j+1), k, l]
                                comb = combine_values(vector)
                                new_mat[i][j][k][l] = comb
                            
        new_int_weights.append(new_mat)

    new_int_biases = []

    for iter, layer_biases in enumerate(padded_int_biases):
        dims = np.shape(layer_biases)
        new_mat = np.zeros(int(dims[0]/4), dtype = np.int64)
        for j in range(int(dims[0]/4)):
            comb = combine_values(layer_biases[4 * j : 4 * (j+1)])
            new_mat[j] = comb
        new_int_biases.append(new_mat)

    shift_biases = []

    for iter, layer_shift_biases in enumerate(padded_shift_biases):
        dims = np.shape(layer_shift_biases)
        new_mat = np.zeros(int(dims[0]/4), dtype = np.int64)
        for j in range(int(dims[0]/4)):
            sh1 = layer_shift_biases[4*j]
            sh2 = layer_shift_biases[4*j+1]
            sh3 = layer_shift_biases[4*j+2]
            sh4 = layer_shift_biases[4*j+3]
            s = (sh1 << 27) | (sh2 << 20) | (sh3 << 13) | (sh4 << 6)
            s += mode_per_layer[iter]
            new_mat[j] = s
        shift_biases.append(new_mat)
    
    mul_vals = []

    for iter, layer_muls in enumerate(padded_mul_vals):
        if(len(np.shape(layer_muls)) > 0):
            dims = np.shape(layer_muls)
            new_mat = np.zeros(int(dims[0]/4), dtype = np.int64)
            for j in range(int(dims[0]/4)):
                m1 = layer_muls[4*j]
                m2 = layer_muls[4*j+1]
                m3 = layer_muls[4*j+2]
                m4 = layer_muls[4*j+3]
                m = (m1 << 24) | (m2 << 16) | (m3 << 8) | (m4)
                new_mat[j] = m
        else:
            vec = [layer_muls, layer_muls, layer_muls, layer_muls]
            new_mat = combine_values(vec)
        
        mul_vals.append(new_mat)
        
    shift_vals = []

    for iter, layer_shifts in enumerate(padded_shift_vals):
        if(len(np.shape(layer_shifts)) > 0):
            dims = np.shape(layer_shifts)
            new_mat = np.zeros(int(dims[0]/4), dtype = np.int64)
            for j in range(int(dims[0]/4)):
                sh1 = layer_shifts[4*j] + 7
                sh2 = layer_shifts[4*j+1] + 7
                sh3 = layer_shifts[4*j+2] + 7
                sh4 = layer_shifts[4*j+3] + 7
                s = (sh1 << 27) | (sh2 << 20) | (sh3 << 13) | (sh4 << 6)
                if(iter + 1 == len(padded_shift_vals)):
                    s += 1
                else:
                    if(mode_per_layer[iter+1] < 4):
                        s += 1
                new_mat[j] = s
        else:
            sh1 = layer_shifts + 7
            new_mat = (sh1 << 27) | (sh1 << 20) | (sh1 << 13) | (sh1 << 6)
            if(iter + 1 == len(padded_shift_vals)):
                new_mat += 1
            else:
                if(mode_per_layer[iter+1] < 4):
                    new_mat += 1
                
        shift_vals.append(new_mat)

    return combined_input_data, new_int_weights, new_int_biases, shift_biases, mul_vals, shift_vals

def save_1d_inputs(path, input):
    with open(path + '/ibex_inputs.h', 'w') as f:
        f.write('#ifndef MLP_INPUTS_H\n#define MLP_INPUTS_H\n\n')
        dims = np.shape(input)
        st = 'static const int input[' + str(dims[0]) + '][' + str(dims[1]) + '] = {\n'
        f.write(st)
        for n in range(dims[0]):
            f.write('\t{')

            for m in range(dims[1] - 1):
                f.write(str(input[n][m]) + ', ')
            f.write(str(input[n][m+1]) + '}')
            if(n != dims[0]-1):
                f.write(',')
            f.write('\n')

        f.write('};\n\n')

        f.write('#endif /* IBEX_MLP_INPUTS_H */')
    return

def save_2d_inputs(path, input):
    with open(path + '/ibex_inputs.h', 'w') as f:
        f.write('#ifndef IBEX_INPUTS_H\n#define IBEX_INPUTS_H\n\n')
        test_batch_X_cnn_new = np.transpose(input, (2, 3, 1, 0))
        dims = np.shape(test_batch_X_cnn_new)
        st = 'static const int input[' + str(dims[0]) + '][' + str(dims[1]) + '][' + str(dims[2]) + ']['
        st += str(dims[3]) + '] = {\n'
        f.write(st)
        for n in range(dims[0]):
            f.write('\t{\n')

            for m in range(dims[1]):
                f.write('\t\t{\n')

                for k in range(dims[2]):
                    f.write('\t\t\t{')
                    for l in range(dims[3]-1):
                        f.write(str(test_batch_X_cnn_new[n][m][k][l]) + ', ')
                    if(dims[3] != 1):
                        f.write(str(test_batch_X_cnn_new[n][m][k][l+1]) + '}')
                    else:
                        f.write(str(test_batch_X_cnn_new[n][m][k][0]) + '}')

                    if(k != dims[2]-1):
                        f.write(',')
                f.write('\n')

                f.write('\t\t}')
                if(m != dims[1]-1):
                    f.write(',')

                f.write('\n')

            f.write('\t}')
            if(n != dims[0]-1):
                f.write(',')
            f.write('\n')

        f.write('};\n\n\n')
        f.write('#endif /* IBEX_INPUTS_H */')
    
    return 

def save_mlp_net_params(path, int_weights, int_biases, mul_vals, shift_vals, shift_biases = None):
    i = 0
    j = 0

    # Open a text file for writing
    with open(path + '/mlp_weights.h', 'w') as f:
        f.write('#ifndef MLP_WEIGHTS_H\n#define MLP_WEIGHTS_H\n\n')
        for k in range(len(int_weights)):
            dims = np.shape(int_weights[k])
            mat = int_weights[k]
            i += 1
            st = 'static const int W' + str(i) + '[' + str(dims[0]) + ']' + '[' + str(dims[1]) + '] = {\n'
            f.write(st)

            for n in range(dims[0]):
                f.write('\t{')
                for m in range(dims[1] - 1):
                    f.write(str(mat[n][m]) + ', ')
                if(dims[1] == 1):
                    f.write(str(mat[n][0]) + '}')
                else:
                    f.write(str(mat[n][m+1]) + '}')
                if(n != dims[0]-1):
                    f.write(',')
                f.write('\n')
            f.write('};\n\n')
            
        for k in range(len(int_biases)):
            dims = np.shape(int_biases[k])
            mat = int_biases[k]
            j += 1
            st = 'static const int B' + str(j) + '[' + str(dims[0]) + '] = {\n\t'
            f.write(st)

            for n in range(dims[0]):
                f.write(str(mat[n]))
                if(n != dims[0] - 1):
                    f.write(', ')
            f.write('\n};\n\n')

        f.write('\n')
        f.write('#endif /* MLP_WEIGHTS_H */')

    if('original' in path):
        with open(path + '/ibex_mlp_params.h', 'w') as f:
            f.write('#ifndef IBEX_MLP_PARAMS_H\n#define IBEX_MLP_PARAMS_H\n\n')
            for i, mul_v in enumerate(mul_vals):
                f.write('#define MV' + str(i+1) + ' ' + str(mul_v) + '\n')
            
            f.write('\n')
            
            for i, shift_v in enumerate(shift_vals):
                f.write('#define SV' + str(i+1) + ' ' + str(shift_v+7) + '\n')

            f.write('\n')

            for i, mul_v in enumerate(mul_vals):
                f.write('#define SB' + str(i+1) + ' ' + str(0) + '\n')
            
            f.write('\n#endif /* IBEX_MLP_PARAMS_H */')

    else:
        bi = 0
        with open(path + '/ibex_mlp_params.h', 'w') as f:
            f.write('#ifndef IBEX_MLP_PARAMS_H\n#define IBEX_MLP_PARAMS_H\n\n')

            for i, mul_v in enumerate(mul_vals):
                f.write('#define MV' + str(i+1) + ' ' + str(mul_v) + '\n')
            
            f.write('\n')
            
            for i, shift_v in enumerate(shift_vals):
                f.write('#define SV' + str(i+1) + ' ' + str(shift_v) + '\n')
            
            f.write('\n')
            
            for k in range(len(shift_biases)):
                dims = np.shape(shift_biases[k])
                mat = shift_biases[k]
                bi += 1
                st = 'static const int SB' + str(bi) + '[' + str(dims[0]) + '] = {\n\t'
                f.write(st)

                for n in range(dims[0]):
                    f.write(str(mat[n]))
                    if(n != dims[0] - 1):
                        f.write(', ')
                f.write('\n};\n\n')
                
            f.write('#endif /* IBEX_MLP_PARAMS_H */')

    return

def save_cnn_net_params(path, int_weights, int_biases, mul_vals, shift_vals, shift_biases = None):
    wi = 0
    bi = 0
    fi = 0

    # Open a text file for writing
    with open(path + '/cnn_weights.h', 'w') as f:
        f.write('#ifndef CNN_WEIGHTS_H\n#define CNN_WEIGHTS_H\n\n')
        for k in range(len(int_weights)):
            dims = np.shape(int_weights[k])
            mat = int_weights[k]   
            
            if(len(dims) == 2 or ((len(dims) == 4) and dims[2] == dims[3] == 1)):
                f.write('static const int ')
                if(len(dims) == 2):
                    wi += 1
                    f.write('W' + str(wi))                
                else:
                    mat = np.squeeze(mat, axis = (2,3))
                    fi += 1
                    f.write('F' + str(fi))
                    
                st = '[' + str(dims[0]) + ']' + '[' + str(dims[1]) + '] = {\n'
                f.write(st)
                for n in range(dims[0]):
                    f.write('\t{')
                    for m in range(dims[1] - 1):
                        f.write(str(mat[n][m]) + ', ')
                    if(dims[1] == 1):
                        f.write(str(mat[n][0]) + '}')
                    else:
                        f.write(str(mat[n][m+1]) + '}')
                    if(n != dims[0]-1):
                        f.write(',')
                    f.write('\n')
                f.write('};\n\n')
            
            elif (len(dims) == 3):
                dims = np.shape(mat)
                fi += 1
                st = 'static const int F' + str(fi) + '[' + str(dims[0]) + '][' + str(dims[1])
                st += '][' + str(dims[2]) + '] = {\n'
                f.write(st)

                for n in range(dims[0]):
                    f.write('\t{\n')
                    for l in range(dims[1]):
                        f.write('\t\t{')
                        for h in range(dims[2] - 1):
                            f.write(str(mat[n][l][h]) + ', ')
                        if dims[2] != 1:
                            f.write(str(mat[n][l][dims[2] - 1]) + '}')
                        else:
                            f.write(str(mat[n][l][0]) + '}')
                        if (l != dims[1] - 1):
                            f.write(',')
                        f.write('\n')
                    f.write('\t}')
                    if n != dims[0] - 1:
                        f.write(',')
                    f.write('\n')
                f.write('};\n\n')
            
            elif(len(dims) == 4):
                mat = np.transpose(mat, (0, 2, 3, 1))
                dims = np.shape(mat)
                fi += 1
                st = 'static const int F' + str(fi) + '[' + str(dims[0]) + '][' + str(dims[1])
                st += '][' + str(dims[2]) + '][' + str(dims[3]) + '] = {\n'
                f.write(st)

                for n in range(dims[0]):
                    f.write('\t{\n')
                    for m in range(dims[1]):
                        f.write('\t\t{\n')
                        for l in range(dims[2]):
                            f.write('\t\t\t{')
                            for h in range(dims[3] - 1):
                                f.write(str(mat[n][m][l][h]) + ', ')
                            if(dims[3] != 1):
                                f.write(str(mat[n][m][l][h+1]) + '}')
                            else:
                                f.write(str(mat[n][m][l][0]) + '}')
                            if (l != dims[2]-1):
                                f.write(',')
                            f.write('\n')
                        f.write('\t\t}')
                        if (m != dims[1] - 1):
                            f.write(',')
                        f.write('\n')
                    f.write('\t}')
                    if (n != dims[0] - 1):
                        f.write(',')
                    f.write('\n')
                f.write('};\n\n')
                
        for k in range(len(int_biases)):
            dims = np.shape(int_biases[k])
            mat = int_biases[k]
            bi += 1
            st = 'static const int B' + str(bi) + '[' + str(dims[0]) + '] = {\n\t'
            f.write(st)

            for n in range(dims[0]):
                f.write(str(mat[n]))
                if(n != dims[0] - 1):
                    f.write(', ')
            f.write('\n};\n\n')

        f.write('\n')
        f.write('#endif /* CNN_WEIGHTS_H */')

    if('original' in path):
        with open(path + '/ibex_cnn_params.h', 'w') as f:
            f.write('#ifndef IBEX_CNN_PARAMS_H\n#define IBEX_CNN_PARAMS_H\n\n')
            for i, mul_v in enumerate(mul_vals):
                f.write('#define MV' + str(i+1) + ' ' + str(mul_v) + '\n')
            
            f.write('\n')
            
            for i, shift_v in enumerate(shift_vals):
                f.write('#define SV' + str(i+1) + ' ' + str(shift_v+7) + '\n')

            f.write('\n')

            for i, mul_v in enumerate(mul_vals):
                f.write('#define SB' + str(i+1) + ' ' + str(0) + '\n')
            
            f.write('\n#endif /* IBEX_CNN_PARAMS_H */')

    else:
        bi = 0
        with open(path + '/ibex_cnn_params.h', 'w') as f:
            f.write('#ifndef IBEX_CNN_PARAMS_H\n#define IBEX_CNN_PARAMS_H\n\n')

            for i, mul_v in enumerate(mul_vals):
                f.write('#define MV' + str(i+1) + ' ' + str(mul_v) + '\n')
            
            f.write('\n')
            
            for i, shift_v in enumerate(shift_vals):
                f.write('#define SV' + str(i+1) + ' ' + str(shift_v) + '\n')
            
            f.write('\n')
            
            for k in range(len(shift_biases)):
                dims = np.shape(shift_biases[k])
                mat = shift_biases[k]
                bi += 1
                st = 'static const int SB' + str(bi) + '[' + str(dims[0]) + '] = {\n\t'
                f.write(st)

                for n in range(dims[0]):
                    f.write(str(mat[n]))
                    if(n != dims[0] - 1):
                        f.write(', ')
                f.write('\n};\n\n')
                
            f.write('#endif /* IBEX_CNN_PARAMS_H */')

    return

def generate_Makefile(path, name):
    with open(path + '/Makefile', 'w') as f:
        f.write('# Copyright lowRISC contributors.\n')
        f.write('# Licensed under the Apache License, Version 2.0, see LICENSE for details.\n')
        f.write('# SPDX-License-Identifier: Apache-2.0\n')
        f.write('#\n# Generate a baremetal application\n\n')
        f.write('# Name of the program $(PROGRAM).c will be added as a source file\n\n')
        
        f.write('PROGRAM = ' + name + '\n')
        f.write('PROGRAM_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))\n')
        f.write('# Any extra source files to include in the build. Use the upper case .S\n')
        f.write('# extension for assembly files\nEXTRA_SRCS :=\n\n')
        f.write('include ${PROGRAM_DIR}/../../common/common.mk')
    
    shutil.copy(path + '/Makefile', path + '/../optimized')

    return

def generate_og_c_code_mlp(path, name, int_weights, optimal_config, type_of_layer):
    with open(path + '/' + name + '.c', 'w') as f:
        f.write('#include "simple_system_common.h"\n')
        f.write('#include "fully_connected.h"\n')
        f.write('#include "ibex_mlp_params.h"\n')
        f.write('#include "mlp_weights.h"\n')
        f.write('#include "ibex_inputs.h"\n\n')
        
        f.write('#define IN_DIM ' + str(int_weights[0].shape[1]))
        for i in range(1, len(int_weights)):
            f.write('\n#define HIDDEN_DIM' + str(i) + ' ' + str(int_weights[i].shape[1]))
        f.write('\n#define OUT_DIM ' + str(int_weights[-1].shape[0]))
        
        f.write('\n#define SAMPLES 1\n\n')
        f.write('int outs[SAMPLES][OUT_DIM];\n\n')
        
        f.write('void ' + name + '() {\n\n')
        f.write('\tint inp[IN_DIM];\n')
        for i in range(1, len(int_weights)):
            f.write('\tint y' + str(i) + '[HIDDEN_DIM' + str(i) + '];\n')
        f.write('\tint out[OUT_DIM];\n')
        
        f.write('\n\tfor (int iter = 0; iter < SAMPLES; iter ++){\n')
        f.write('\t\tfor(int i = 0; i < IN_DIM; i++) inp[i] = input[iter][i];\n\n')
        f.write('\t\tpcount_enable(1);\n\n')
        
        if(type_of_layer[0] == 'Linear'):
            f.write('\t\tmlp_layer(inp, y1, IN_DIM,') 
            f.write(' HIDDEN_DIM1, W1, B1, SB1, MV1, SV1);\n')
        
        for i, b_w in enumerate(optimal_config[1:-1], start = 1):
            if(type_of_layer[i] == 'Linear'):
                f.write('\t\tmlp_layer(y' + str(i) + ', y' + str(i+1))
                f.write(', HIDDEN_DIM' + str(i) + ', HIDDEN_DIM' + str(i+1) + ', W' + str(i+1))
                f.write(', B' + str(i+1) + ', SB' + str(i+1) + ', MV' + str(i+1) + ', SV' + str(i+1) + ');\n')
        
        if(type_of_layer[-1] == 'Linear'):
            f.write('\t\tmlp_layer(y')
            f.write(str(len(int_weights)-1)+', out, HIDDEN_DIM'+str(len(int_weights)-1))
            f.write(', OUT_DIM, W' + str(len(int_weights)) + ', B')
            f.write(str(len(int_weights)) + ', SB' + str(len(int_weights)) + ', MV')
            f.write(str(len(int_weights)) + ', SV' + str(len(int_weights)) + ');\n\n')
        
        f.write('\t\tpcount_enable(0);\n\n')
        f.write('\t\tputs("Output Layer Values:\\n");\n')
        f.write('\t\tfor(int i = 0; i < OUT_DIM; i++) {\n')
        f.write('\t\t\tputhex(out[i]);\n')
        f.write('\t\t\tputs("\\n");\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('}\n\n')
        
        f.write('int main(void) {\n\n')
        f.write('\tpcount_enable(0);\n\n')
        f.write('\t' + name + '();\n\n')
        f.write('\treturn 0;\n}')
    return

def generate_opt_c_code_mlp(path, name, int_weights, optimal_config, type_of_layer):
    with open(path + '/' + name + '.c', 'w') as f:
        f.write('#include "simple_system_common.h"\n')
        f.write('#include "fully_connected_opt.h"\n')
        f.write('#include "ibex_mlp_params.h"\n')
        f.write('#include "mlp_weights.h"\n')
        f.write('#include "ibex_inputs.h"\n\n')
        f.write('#define IN_DIM ' + str((8//optimal_config[0]) * int_weights[0].shape[1]))
        for i in range(1, len(int_weights)):
            f.write('\n#define HIDDEN_DIM' + str(i) + ' ' + str(4 * int_weights[i-1].shape[0]))
        f.write('\n#define OUT_DIM ' + str(4 * int_weights[-1].shape[0]))
        
        f.write('\n#define SAMPLES 1\n\n')
        f.write('int outs[SAMPLES][OUT_DIM >> 2];\n\n')
        
        f.write('void ' + name + '() {\n\n')
        f.write('\tint inp[IN_DIM >> 2];\n')
        for i in range(1, len(int_weights)):
            f.write('\tint y' + str(i) + '[HIDDEN_DIM' + str(i) + ' >> 2];\n')
        f.write('\tint out[OUT_DIM >> 2];\n')
        
        f.write('\n\tfor (int iter = 0; iter < SAMPLES; iter ++){\n')
        f.write('\t\tfor(int i = 0; i < IN_DIM >> 2; i++) inp[i] = input[iter][i];\n\n')
        f.write('\t\tpcount_enable(1);\n\n')
        
        if(type_of_layer[0] == 'Linear'):
            f.write('\t\tmlp_layer_' + str(optimal_config[0]) + 'bits(inp, y1, IN_DIM >> 2,') 
            f.write(' HIDDEN_DIM1 >> 2, W1, B1, SB1, MV1, SV1);\n')
        
        for i, b_w in enumerate(optimal_config[1:-1], start = 1):
            if(type_of_layer[i] == 'Linear'):
                f.write('\t\tmlp_layer_' + str(b_w) + 'bits(y' + str(i) + ', y' + str(i+1))
                f.write(', HIDDEN_DIM' + str(i) + ' >> 2, HIDDEN_DIM' + str(i+1) + ' >> 2, W' + str(i+1))
                f.write(', B' + str(i+1) + ', SB' + str(i+1) + ', MV' + str(i+1) + ', SV' + str(i+1) + ');\n')
        
        if(type_of_layer[-1] == 'Linear'):
            f.write('\t\tmlp_layer_' + str(optimal_config[-1]) + 'bits(y')
            f.write(str(len(int_weights)-1)+', out, HIDDEN_DIM'+str(len(int_weights)-1))
            f.write(' >> 2, OUT_DIM >> 2, W' + str(len(int_weights)) + ', B')
            f.write(str(len(int_weights)) + ', SB' + str(len(int_weights)) + ', MV')
            f.write(str(len(int_weights)) + ', SV' + str(len(int_weights)) + ');\n\n')
        
        f.write('\t\tpcount_enable(0);\n\n')
        f.write('\t\tputs("Output Layer Values:\\n");\n')
        f.write('\t\tfor(int i = 0; i < OUT_DIM >> 2; i++) {\n')
        f.write('\t\t\tputhex((out[i] & 0xFF000000) >> 24);\n')
        f.write('\t\t\tputs(" ");\n')
        f.write('\t\t\tputhex((out[i] & 0xFF0000) >> 16);\n')
        f.write('\t\t\tputs(" ");\n')
        f.write('\t\t\tputhex((out[i] & 0xFF00) >> 8);\n')
        f.write('\t\t\tputs(" ");\n')
        f.write('\t\t\tputhex(out[i] & 0xFF);\n')
        f.write('\t\t\tputs("\\n");\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('}\n\n')
        
        f.write('int main(void) {\n\n')
        f.write('\tpcount_enable(0);\n\n')
        f.write('\t' + name + '();\n\n')
        f.write('\treturn 0;\n}')

def get_cnn_details(module, details = None):
    if details is None:
        details = []

    for layer in module.children():
        if isinstance(layer, nn.Conv2d):
            details.append({
                "layer_type": "Conv2d",
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
                "padding": layer.padding,
                "groups": layer.groups
            })

        elif isinstance(layer, nn.MaxPool2d):
            details.append({
                "layer_type": "MaxPool2d",
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
                "padding": layer.padding
            })

        elif isinstance(layer, nn.AvgPool2d):
            details.append({
                "layer_type": "AvgPool2d",
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
                "padding": layer.padding
            })

        elif isinstance(layer, nn.Linear):
            details.append({
                "layer_type": "Linear",
                "in_features": layer.in_features,
                "out_features": layer.out_features
            })

        # Recursively apply to children modules
        get_cnn_details(layer, details)

    return details

def generate_og_c_code_cnn(path, name, input, cnn_details, int_weights):
    with open(path + '/' + name + '.c', 'w') as f:
        f.write('#include "simple_system_common.h"\n')
        f.write('#include "cnn_weights.h"\n')
        f.write('#include "fully_connected.h"\n')
        f.write('#include "ibex_cnn_params.h"\n')
        f.write('#include "ibex_inputs.h"\n')
        f.write('#include "conv2d.h"\n')

        for detail in cnn_details[:-1]:
            if detail["layer_type"] == "Conv2d":
                if(detail["in_channels"] == detail["out_channels"] == detail["groups"] != 1):
                    f.write('#include "dws_conv.h"\n')
                    break
        
        f.write('\n')
        f.write('#define IMG_SZ ' + str(np.shape(input)[2]) + '\n')
        f.write('#define NUM_FIL0 ' + str(np.shape(input)[1]) + '\n\n')
        i = 1
        for w in int_weights:
            if(len(np.shape(w)) == 4):
                f.write('#define FILTER' + str(i) + ' ' + str(w.shape[2]) + '\n')
                i += 1

        f.write('\n')
        
        i = 1
        for w in int_weights:
            if(len(np.shape(w)) == 4):
                f.write('#define NUM_FIL' + str(i) + ' ' + str(w.shape[0]) + '\n')
                i += 1

        f.write('\n')

        i = 1
        for detail in cnn_details:
           if detail["layer_type"] == "Conv2d":
                f.write('#define STRIDE' + str(i) + ' ' + str(detail["stride"][0]) + '\n')
                i += 1

        f.write('\n')

        i = 1
        for detail in cnn_details:
           if detail["layer_type"] == "Conv2d":
                if(detail["padding"] == 'same'):
                    f.write('#define PAD_TB' + str(i) + ' ' + str((detail["kernel_size"][0] - 1)//2) + '\n')
                    f.write('#define PAD_LR' + str(i) + ' ' + str((detail["kernel_size"][0] - 1)//2) + '\n')
                
                elif(detail["padding"] == 'valid'):
                    f.write('#define PAD_TB' + str(i) + ' 0\n')
                    f.write('#define PAD_LR' + str(i) + ' 0\n')
                
                else:
                    f.write('#define PAD_TB' + str(i) + ' ' + str(detail["padding"][0]) + '\n')
                    f.write('#define PAD_LR' + str(i) + ' ' + str(detail["padding"][0]) + '\n')

                f.write('\n')
                i += 1

        i = 1
        for detail in cnn_details:
           if ((detail["layer_type"] == "MaxPool2d") or (detail["layer_type"] == "AvgPool2d")):
                f.write('#define POOL_STRIDE' + str(i) + ' ' + str(detail["stride"]) + '\n')
                f.write('#define POOL_SIZE' + str(i) + ' ' + str(detail["kernel_size"]) + '\n')
                f.write('\n')
                i += 1

        i = 1
        for w in int_weights[:-1]:
            if(len(np.shape(w)) == 2):
                f.write('#define DENSE_DIM' + str(i) + ' ' + str(w.shape[0]) + '\n')
                i += 1
        
        f.write('#define OUT_DIM ' + str(int_weights[-1].shape[0]) + '\n\n')
        f.write('#define SAMPLES 1\nint outs[SAMPLES][OUT_DIM];\n\n')
        f.write('void ' + name + '() {\n\n')

        i = 1
        fi = 1
        st = 1
        flatten = 0

        for detail in cnn_details:
            if detail["layer_type"] == "Conv2d":
                f.write('\tint dout' + str(i) + ' = NUM_FIL' + str(fi) + ';\n')
                if(i == 1):
                   f.write('\tint hout' + str(i) + ' = ((IMG_SZ - FILTER1 + 2 * PAD_TB1)/STRIDE1) + 1;\n')
                   f.write('\tint wout' + str(i) + ' = ((IMG_SZ - FILTER1 + 2 * PAD_LR1)/STRIDE1) + 1;\n')
                else:
                   f.write('\tint hout' + str(i) + ' = ((hout' + str(i-1) + ' - FILTER' + str(fi))
                   f.write('+ 2 * PAD_TB' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')

                   f.write('\tint wout' + str(i) + ' = ((wout' + str(i-1) + ' - FILTER' + str(fi))
                   f.write('+ 2 * PAD_LR' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                fi += 1
            
            elif ((detail["layer_type"] == "MaxPool2d") or (detail["layer_type"] == "AvgPool2d")):
                f.write('\tint dout' + str(i) + ' = dout' + str(i-1) + ';\n')
                f.write('\tint hout' + str(i) + ' = hout' + str(i-1) + '/POOL_STRIDE' + str(st) + ';\n')
                f.write('\tint wout' + str(i) + ' = wout' + str(i-1) + '/POOL_STRIDE' + str(st) + ';\n')
                st += 1
            
            elif detail["layer_type"] == "Linear":
                if flatten == 0:
                    f.write('\tint flatten_dim = dout' + str(i-1) + ' * hout' + str(i-1) + ' * wout' + str(i-1) + ';\n')
                    flatten = 1
                break

            f.write('\n')
            i += 1

        f.write('\n')
        i = 1
        fi = 1
        dn = 1
        flatten = 0

        f.write('\tint in[IMG_SZ][IMG_SZ][NUM_FIL0];\n')
        f.write('\tint inp_dim[3] = {IMG_SZ, IMG_SZ, NUM_FIL0};\n\n')

        for detail in cnn_details:
            if detail["layer_type"] == "Conv2d":
                f.write('\tint out' + str(i) + '[hout' + str(i) + '][wout' + str(i) + '][dout' + str(i) + '];\n')
                f.write('\tint pad_' + str(i) + '[4] = {PAD_TB' + str(fi) + ', PAD_TB' + str(fi))
                f.write(', PAD_LR' + str(fi) + ', PAD_LR' + str(fi) + '};\n')
                f.write('\tint outp_dim' + str(i) + '[3] = {hout' + str(i) + ', wout' + str(i))
                f.write(', dout' + str(i) + '};\n')
                f.write('\tint f_dim' + str(i) + '[4] = {NUM_FIL' + str(fi) + ', FILTER' + str(fi))
                f.write(', FILTER' + str(fi) + ', NUM_FIL' + str(fi-1) + '};\n')
                fi += 1
            
            elif ((detail["layer_type"] == "MaxPool2d") or (detail["layer_type"] == "AvgPool2d")):
                f.write('\tint out' + str(i) + '[hout' + str(i) + '][wout' + str(i) + '][dout' + str(i) + '];\n')
                f.write('\tint outp_dim' + str(i) + '[3] = {hout' + str(i) + ', wout' + str(i))
                f.write(', dout' + str(i) + '};\n')
            
            elif detail["layer_type"] == "Linear":
                if flatten == 0:
                    f.write('\tint out' + str(i) + '[flatten_dim];\n')
                    flatten = 1
                else:
                    f.write('\tint out' + str(i) + '[DENSE_DIM' + str(dn) + '];')
                    dn += 1

            f.write('\n')
            i += 1

        f.write('\n\tint out[OUT_DIM];\n\n\tfor (int iter = 0; iter < SAMPLES; iter++){\n\n')
       
        f.write('\t\tfor(int i = 0; i < IMG_SZ; i++){\n')
        f.write('\t\t\tfor(int j = 0; j < IMG_SZ; j++){\n')
        f.write('\t\t\t\tfor(int k = 0; k < NUM_FIL0; k++){\n')
        f.write('\t\t\t\t\tin[i][j][k] = input[i][j][k][iter];\n')
        f.write('\t\t\t\t}\n\t\t\t}\n\t\t}\n\n\t\tpcount_enable(1);\n\n')

        i = 1
        fi = 1
        st = 1
        dn = 1
        flatten = 0

        for detail in cnn_details[:-1]:
            if detail["layer_type"] == "Conv2d":
                if(detail["in_channels"] == detail["out_channels"] == detail["groups"] != 1):
                    conv_type = 'dw_conv'
                elif(detail["kernel_size"][0] == 1):
                    conv_type = 'pw_conv'
                else:
                    conv_type = "conv2"
                if(i == 1):
                    f.write('\t\t' + conv_type + '(inp_dim, f_dim1, outp_dim1, in, F1, B1, ')
                    f.write('out1, STRIDE1, pad_1, SB1, MV1, SV1);')
                else:
                    f.write('\t\t' + conv_type + '(outp_dim' + str(i-1) + ', f_dim' + str(i) + ', outp_dim' + str(i))
                    f.write(', out' + str(i-1) + ', F' + str(fi) + ', B' + str(fi) + ', out' + str(i))
                    f.write(', STRIDE' + str(fi) + ', pad_' + str(i) + ', SB' + str(fi))
                    f.write(', MV' + str(fi) + ', SV' + str(fi) + ');')
                fi += 1
            
            elif detail["layer_type"] == "MaxPool2d":
                f.write('\t\tmaxpool2(outp_dim' + str(i-1) + ', outp_dim' + str(i))
                f.write(', out' + str(i-1) + ', out' + str(i) + ', POOL_SIZE' + str(st) + ', POOL_STRIDE')
                f.write(str(st) + ');\n')
                st += 1

            elif(detail["layer_type"] == "AvgPool2d"):
                f.write('\t\tavgpool2(outp_dim' + str(i-1) + ', outp_dim' + str(i))
                f.write(', out' + str(i-1) + ', out' + str(i) + ', POOL_SIZE' + str(st) + ', POOL_STRIDE')
                f.write(str(st) + ');\n')
                st += 1

            elif detail["layer_type"] == "Linear":
                if flatten == 0:
                    f.write('\t\tflatten(outp_dim' + str(i-1) + ', out' + str(i-1) + ', out' + str(i) + ');\n\n')
                    i += 1
                    f.write('\t\tmlp_layer(out' + str(i-1) + ', out' + str(i) + ', flatten_dim, DENSE_DIM1')
                    f.write(', W1, B' + str(fi + dn - 1) +  ', SB' + str(fi + dn - 1) + ', MV' + str(fi + dn - 1))
                    f.write(', SV' + str(fi + dn - 1) + ');')
                    dn += 1
                    flatten = 1
                else:
                    f.write('\t\tmlp_layer(out' + str(i-1) + ', out' + str(i) + ', DENSE_DIM' + str(dn-1))
                    f.write(', DENSE_DIM' + str(dn) + ', W' + str(dn) + ', B' + str(fi + dn - 1))
                    f.write(', SB' + str(fi + dn - 1) + ', MV' + str(fi + dn - 1))
                    f.write(', SV' + str(fi + dn - 1) + ');')
                    dn += 1

            f.write('\n')
            i += 1
        
        if flatten == 0:
            f.write('\t\tflatten(outp_dim' + str(i-1) + ', out' + str(i-1) + ', out' + str(i) + ');\n\n')
            i += 1
            f.write('\t\tmlp_layer(out' + str(i-1) + ', out, flatten_dim, OUT_DIM, ')
            f.write('W1, B' + str(fi + dn - 1) +  ', SB' + str(fi + dn - 1) + ', MV' + str(fi + dn - 1))
            f.write(', SV' + str(fi + dn - 1) + ');')
        else:
            f.write('\t\tmlp_layer(out' + str(i-1) + ', out, DENSE_DIM' + str(dn-1))
            f.write(', OUT_DIM, W' + str(dn) + ', B' + str(fi + dn - 1))
            f.write(', SB' + str(fi + dn - 1) + ', MV' + str(fi + dn - 1))
            f.write(', SV' + str(fi + dn - 1) + ');\n')

        f.write('\n\t\tpcount_enable(0);\n\n')
        f.write('\t\tputs("Output Layer Values:\\n");\n')
        f.write('\t\tfor(int i = 0; i < OUT_DIM; i++) {\n')
        f.write('\t\t\tputhex(out[i]);\n')
        f.write('\t\t\tputs("\\n");\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('}\n\n')
        
        f.write('int main(void) {\n\n')
        f.write('\tpcount_enable(0);\n\n')
        f.write('\t' + name + '();\n\n')
        f.write('\treturn 0;\n}')

    return

def generate_opt_c_code_cnn(path, name, input, cnn_details, int_weights, optimal_config):
    with open(path + '/' + name + '.c', 'w') as f:
        f.write('#include "simple_system_common.h"\n')
        f.write('#include "cnn_weights.h"\n')
        f.write('#include "fully_connected_opt.h"\n')
        f.write('#include "ibex_cnn_params.h"\n')
        f.write('#include "ibex_inputs.h"\n')
        f.write('#include "conv2d_opt.h"\n')
        
        for detail in cnn_details[:-1]:
            if detail["layer_type"] == "Conv2d":
                if(detail["in_channels"] == detail["out_channels"] == detail["groups"] != 1):
                    f.write('#include "dws_conv_opt.h"\n')
                    break
                
        f.write('\n')
        
        f.write('#define IMG_SZ ' + str(np.shape(input)[2]) + '\n')
        f.write('#define NUM_FIL0 ' + str(np.shape(input)[1]) + '\n\n')
        i = 1
        for w in int_weights:
            if(len(np.shape(w)) == 4 or len(np.shape(w)) == 3):
                f.write('#define FILTER' + str(i) + ' ' + str(w.shape[2]) + '\n')
                i += 1

        f.write('\n')
        
        i = 1
        for w in int_weights:
            if(len(np.shape(w)) == 4 or len(np.shape(w)) == 3):
                f.write('#define NUM_FIL' + str(i) + ' ' + str(w.shape[0]) + '\n')
                i += 1

        f.write('\n')

        i = 1
        for detail in cnn_details:
           if detail["layer_type"] == "Conv2d":
                f.write('#define STRIDE' + str(i) + ' ' + str(detail["stride"][0]) + '\n')
                i += 1

        f.write('\n')

        i = 1
        for detail in cnn_details:
           if detail["layer_type"] == "Conv2d":
                if(detail["padding"] == 'same'):
                    f.write('#define PAD_TB' + str(i) + ' ' + str((detail["kernel_size"][0] - 1)//2) + '\n')
                    f.write('#define PAD_LR' + str(i) + ' ' + str((detail["kernel_size"][0] - 1)//2) + '\n')
                
                elif(detail["padding"] == 'valid'):
                    f.write('#define PAD_TB' + str(i) + ' 0\n')
                    f.write('#define PAD_LR' + str(i) + ' 0\n')
                
                else:
                    f.write('#define PAD_TB' + str(i) + ' ' + str(detail["padding"][0]) + '\n')
                    f.write('#define PAD_LR' + str(i) + ' ' + str(detail["padding"][0]) + '\n')

                f.write('\n')
                i += 1

        i = 1
        for detail in cnn_details:
           if ((detail["layer_type"] == "MaxPool2d") or (detail["layer_type"] == "AvgPool2d")):
                f.write('#define POOL_STRIDE' + str(i) + ' ' + str(detail["stride"]) + '\n')
                f.write('#define POOL_SIZE' + str(i) + ' ' + str(detail["kernel_size"]) + '\n')
                f.write('\n')
                i += 1

        i = 1
        for w in int_weights[:-1]:
            if(len(np.shape(w)) == 2):
                f.write('#define DENSE_DIM' + str(i) + ' ' + str(w.shape[0]) + '\n')
                i += 1
        
        f.write('#define OUT_DIM ' + str(int_weights[-1].shape[0]) + '\n\n')
        f.write('#define SAMPLES 1\nint outs[SAMPLES][OUT_DIM];\n\n')
        f.write('void ' + name + '() {\n\n')

        i = 1
        fi = 1
        st = 1
        flatten = 0

        for detail in cnn_details:
            if detail["layer_type"] == "Conv2d":
                f.write('\tint dout' + str(i) + ' = NUM_FIL' + str(fi) + ';\n')
                if(i == 1):
                   f.write('\tint hout' + str(i) + ' = ((IMG_SZ - FILTER1 + 2 * PAD_TB1)/STRIDE1) + 1;\n')
                   f.write('\tint wout' + str(i) + ' = ((IMG_SZ - FILTER1 + 2 * PAD_LR1)/STRIDE1) + 1;\n')
                else:
                   f.write('\tint hout' + str(i) + ' = ((hout' + str(i-1) + ' - FILTER' + str(fi))
                   f.write('+ 2 * PAD_TB' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')

                   f.write('\tint wout' + str(i) + ' = ((wout' + str(i-1) + ' - FILTER' + str(fi))
                   f.write('+ 2 * PAD_LR' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                fi += 1
            
            elif ((detail["layer_type"] == "MaxPool2d") or (detail["layer_type"] == "AvgPool2d")):
                f.write('\tint dout' + str(i) + ' = dout' + str(i-1) + ';\n')
                f.write('\tint hout' + str(i) + ' = hout' + str(i-1) + '/POOL_STRIDE' + str(st) + ';\n')
                f.write('\tint wout' + str(i) + ' = wout' + str(i-1) + '/POOL_STRIDE' + str(st) + ';\n')
                st += 1
            
            elif detail["layer_type"] == "Linear":
                if flatten == 0:
                    f.write('\tint flatten_dim = dout' + str(i-1) + ' * hout' + str(i-1) + ' * wout' + str(i-1) + ';\n')
                    flatten = 1
                break

            f.write('\n')
            i += 1

        f.write('\n')
        i = 1
        fi = 1
        dn = 1
        flatten = 0

        f.write('\tint in[IMG_SZ][IMG_SZ][NUM_FIL0];\n')
        f.write('\tint inp_dim[3] = {IMG_SZ, IMG_SZ, NUM_FIL0};\n\n')

        for detail in cnn_details:
            if detail["layer_type"] == "Conv2d":
                f.write('\tint out' + str(i) + '[hout' + str(i) + '][wout' + str(i) + '][dout' + str(i) + '];\n')
                f.write('\tint pad_' + str(i) + '[4] = {PAD_TB' + str(fi) + ', PAD_TB' + str(fi))
                f.write(', PAD_LR' + str(fi) + ', PAD_LR' + str(fi) + '};\n')
                f.write('\tint outp_dim' + str(i) + '[3] = {hout' + str(i) + ', wout' + str(i))
                f.write(', dout' + str(i) + '};\n')
                f.write('\tint f_dim' + str(i) + '[4] = {NUM_FIL' + str(fi) + ', FILTER' + str(fi))
                f.write(', FILTER' + str(fi) + ', NUM_FIL' + str(fi-1) + '};\n')
                fi += 1
            
            elif ((detail["layer_type"] == "MaxPool2d") or (detail["layer_type"] == "AvgPool2d")):
                f.write('\tint out' + str(i) + '[hout' + str(i) + '][wout' + str(i) + '][dout' + str(i) + '];\n')
                f.write('\tint outp_dim' + str(i) + '[3] = {hout' + str(i) + ', wout' + str(i))
                f.write(', dout' + str(i) + '};\n')
            
            elif detail["layer_type"] == "Linear":
                if flatten == 0:
                    f.write('\tint out' + str(i) + '[flatten_dim];')
                    flatten = 1
                else:
                    f.write('\tint out' + str(i) + '[DENSE_DIM' + str(dn) + '];')
                    dn += 1

            f.write('\n')
            i += 1

        f.write('\n\tint out[OUT_DIM];\n\n\tfor (int iter = 0; iter < SAMPLES; iter++){\n\n')
       
        f.write('\t\tfor(int i = 0; i < IMG_SZ; i++){\n')
        f.write('\t\t\tfor(int j = 0; j < IMG_SZ; j++){\n')
        f.write('\t\t\t\tfor(int k = 0; k < NUM_FIL0; k++){\n')
        f.write('\t\t\t\t\tin[i][j][k] = input[i][j][k][iter];\n')
        f.write('\t\t\t\t}\n\t\t\t}\n\t\t}\n\n\t\tpcount_enable(1);\n\n')

        i = 1
        j = 0
        fi = 1
        st = 1
        dn = 1
        flatten = 0

        for detail in cnn_details[:-1]:
            if detail["layer_type"] == "Conv2d":
                if(detail["in_channels"] == detail["out_channels"] == detail["groups"] != 1):
                    conv_type = 'dw_conv_opt'
                elif(detail["kernel_size"][0] == 1):
                    conv_type = 'pw_conv_' + str(optimal_config[j]) + 'bits'
                else:
                    conv_type = 'conv2_' + str(optimal_config[j]) + 'bits'
                    
                if(i == 1):
                    f.write('\t\t' + conv_type)
                    if(np.shape(input)[1] == 1):
                        f.write('_1ch')
                    f.write('(inp_dim, f_dim1, outp_dim1, in, F1, B1, ')
                    f.write('out1, STRIDE1, pad_1, SB1, MV1, SV1);')
                else:
                    f.write('\t\t' + conv_type + '(outp_dim' + str(i-1) + ', f_dim' + str(i))
                    f.write(', outp_dim' + str(i) + ', out' + str(i-1) + ', F' + str(fi) + ', B' + str(fi) + ', out')
                    f.write(str(i) + ', STRIDE' + str(fi) + ', pad_' + str(i) + ', SB' + str(fi))
                    f.write(', MV' + str(fi) + ', SV' + str(fi) + ');')
                j += 1
                fi += 1
            
            elif detail["layer_type"] == "MaxPool2d":
                f.write('\t\tmaxpool2_compressed(outp_dim' + str(i-1) + ', outp_dim' + str(i))
                f.write(', out' + str(i-1) + ', out' + str(i) + ', POOL_SIZE' + str(st) + ', POOL_STRIDE')
                f.write(str(st) + ');\n')
                st += 1

            elif(detail["layer_type"] == "AvgPool2d"):
                f.write('\t\tavgpool2_compressed(outp_dim' + str(i-1) + ', outp_dim' + str(i))
                f.write(', out' + str(i-1) + ', out' + str(i) + ', POOL_SIZE' + str(st) + ', POOL_STRIDE')
                f.write(str(st) + ');\n')
                st += 1

            elif detail["layer_type"] == "Linear":
                if flatten == 0:
                    f.write('\t\tflatten(outp_dim' + str(i-1) + ', out' + str(i-1) + ', out' + str(i) + ');\n\n')
                    i += 1
                    f.write('\t\tmlp_layer_' + str(optimal_config[j]) + 'bits(out' + str(i-1) + ', out' + str(i) + ', ')
                    f.write('flatten_dim, DENSE_DIM1, W1, B' + str(fi + dn - 1) +  ', SB' + str(fi + dn - 1) + ', MV')
                    f.write(str(fi + dn - 1) + ', SV' + str(fi + dn - 1) + ');\n')
                    flatten = 1
                else:
                    f.write('\t\tmlp_layer_' + str(optimal_config[j]) + 'bits(out' + str(i-1) + ', out' + str(i) + ', ')
                    f.write('DENSE_DIM' + str(dn-1) + ', DENSE_DIM' + str(dn) + ', W' + str(dn) + ', B')
                    f.write(str(fi + dn - 1) + ', SB' + str(fi + dn - 1) + ', MV' + str(fi + dn - 1))
                    f.write(', SV' + str(fi + dn - 1) + ');\n')
                j += 1
                dn += 1
            f.write('\n')
            i += 1
        
        if flatten == 0:
            f.write('\t\tflatten(outp_dim' + str(i-1) + ', out' + str(i-1) + ', out' + str(i) + ');\n\n')
            i += 1
            f.write('\t\tmlp_layer_' + str(optimal_config[j]) + 'bits(out' + str(i-1) + ', out, ')
            f.write('flatten_dim, OUT_DIM, W1, B' + str(fi + dn - 1) +  ', SB' + str(fi + dn - 1) + ', MV')
            f.write(str(fi + dn - 1) + ', SV' + str(fi + dn - 1) + ');\n')
        else:
            f.write('\t\tmlp_layer_' + str(optimal_config[-1]) + 'bits(out' + str(i-1) + ', out, DENSE_DIM' + str(dn-1))
            f.write(', OUT_DIM, W' + str(dn) + ', B' + str(fi + dn - 1))
            f.write(', SB' + str(fi + dn - 1) + ', MV' + str(fi + dn - 1))
            f.write(', SV' + str(fi + dn - 1) + ');\n')

        f.write('\n\t\tpcount_enable(0);\n\n')
        f.write('\t\tputs("Output Layer Values:\\n");\n')
        f.write('\t\tfor(int i = 0; i < OUT_DIM; i++) {\n')
        f.write('\t\t\tputhex((out[i] & 0xFF000000) >> 24);\n')
        f.write('\t\t\tputs(" ");\n')
        f.write('\t\t\tputhex((out[i] & 0xFF0000) >> 16);\n')
        f.write('\t\t\tputs(" ");\n')
        f.write('\t\t\tputhex((out[i] & 0xFF00) >> 8);\n')
        f.write('\t\t\tputs(" ");\n')
        f.write('\t\t\tputhex(out[i] & 0xFF);\n')
        f.write('\t\t\tputs("\\n");\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('}\n\n')
        
        f.write('int main(void) {\n\n')
        f.write('\tpcount_enable(0);\n\n')
        f.write('\t' + name + '();\n\n')
        f.write('\treturn 0;\n}')

    return
