import mpq_quantize
import pareto_sols
import configure_ibex
import simulate_ibex
import os
import sys
import numpy as np

def create_ibex_qnn(net, name, device, X_train, y_train, X_test, y_test, X_val = None, y_val = None, 
         BATCH_SIZE = 32, epochs = 20, lr = 0.0001, max_acc_drop = None):
    
    net = net.to(device)

    in_shape = mpq_quantize.net_input_size(X_train)

    macc_per_layer, _ = mpq_quantize.display_model_info(net, in_shape)

    train_loader, val_loader, test_loader = mpq_quantize.create_dataloaders(BATCH_SIZE, X_train, y_train, X_test, y_test,
                                                                            X_val, y_val)

    if isinstance(epochs, (int, float)):
        tr_epochs = [epochs]
    else:
        tr_epochs = epochs

    if isinstance(lr, (int, float)):
        tr_lr = [lr]
    else:
        tr_lr = lr
    
    if len(tr_epochs) != len(tr_lr):
        print("Error: The lengths of 'epochs' and 'lr' are not equal.")
        sys.exit(1)

    print('\nFULL PRECISION MODEL TRAINING ...')

    for i in range(len(tr_epochs)):
        print(f'Round No{i+1}')
        fp_net = mpq_quantize.fp_train(net, train_loader, val_loader, device, 
                                       epochs = tr_epochs[i], lr = tr_lr[i])

    fp_accuracy = mpq_quantize.fp_evaluate(net, test_loader, device)

    weights_per_layer, total_macc_opt_sorted = mpq_quantize.create_weight_confs(macc_per_layer)

    if(max_acc_drop is None):
        
        quant_net, accuracy = mpq_quantize.dse(fp_net, max_acc_drop, weights_per_layer, fp_accuracy, train_loader,
                                        test_loader, val_loader, device, tr_epochs, tr_lr)
        
        optimal_config = pareto_sols.pareto_space(fp_accuracy, accuracy, weights_per_layer, macc_per_layer,
                                            total_macc_opt_sorted, name)
        
        layer_mapping = mpq_quantize.create_layer_mapping(optimal_config)
        
        quant_net = mpq_quantize.Quant_Model(fp_net, optimal_config, layer_mapping, 
                                            mpq_quantize.calculate_minimum(train_loader) >= 0)
        
        quant_net = mpq_quantize.train_quant_model(quant_net, train_loader, val_loader, device, epochs, lr)
        
        mpq_quantize.quant_net_evaluation(quant_net, test_loader, device)

    else:
        quant_net, optimal_config = mpq_quantize.dse(fp_net, max_acc_drop, weights_per_layer, fp_accuracy, train_loader, 
                                        test_loader, val_loader, device, tr_epochs, tr_lr)

    print('\nCREATING FILES FOR SIMULATION ON IBEX CORE ...')
    mode_per_layer, layer_type = configure_ibex.decide_mode(fp_net, 
                                    optimal_config, mpq_quantize.calculate_minimum(train_loader) >= 0)

    int_weights, int_og_bias, int_biases, shift_biases, mul_vals, shift_vals = configure_ibex.get_int_params(quant_net)

    input, padded_input, padded_w, padded_b, padded_sb_v, padded_mv, padded_sv = configure_ibex.pad_inputs_weights(quant_net, 
                                                    test_loader, mode_per_layer, int_weights, 
                                                    int_biases, shift_biases, mul_vals, shift_vals)

    combined_input, new_int_w, new_int_b, shift_b, mul_v, shift_v = configure_ibex.concat_inputs_weights(mode_per_layer, 
                                                    padded_input, padded_w, padded_b, 
                                                    padded_sb_v, padded_mv, padded_sv)

    optimized_path = '../inference_codes/' + name + '/optimized'
    original_path = '../inference_codes/' + name + '/original'

    os.makedirs(optimized_path, exist_ok = True)
    os.makedirs(original_path,  exist_ok = True)

    configure_ibex.generate_Makefile(original_path, name)

    if np.unique(layer_type)[0] =='Linear':
        configure_ibex.save_1d_inputs(original_path, input)
        configure_ibex.save_1d_inputs(optimized_path, combined_input)
        
        configure_ibex.save_mlp_net_params(original_path, int_weights, int_og_bias, mul_vals, shift_vals)
        configure_ibex.save_mlp_net_params(optimized_path, new_int_w, new_int_b, mul_v, shift_v, shift_b)
        
        configure_ibex.generate_og_c_code_mlp(original_path, name, int_weights, optimal_config, layer_type)  
        configure_ibex.generate_opt_c_code_mlp(optimized_path, name, new_int_w, optimal_config, layer_type)

    else:
        configure_ibex.save_2d_inputs(original_path, input)
        configure_ibex.save_2d_inputs(optimized_path, combined_input)

        configure_ibex.save_cnn_net_params(original_path, int_weights, int_og_bias, mul_vals, shift_vals)
        configure_ibex.save_cnn_net_params(optimized_path, new_int_w, new_int_b, mul_v, shift_v, shift_b)

        cnn_details = configure_ibex.get_cnn_details(fp_net)

        configure_ibex.generate_og_c_code_cnn(original_path, name, input, cnn_details, int_weights)
        configure_ibex.generate_opt_c_code_cnn(optimized_path, name, combined_input, cnn_details, 
                                            new_int_w, optimal_config)

    print('FINISHED ...')

    if(name == 'elderly_fall'):
        print('\nSIMULATING MODEL ON IBEX CORE\nUSE THE OUTPUTS TO VERIFY THAT THE RESULTS ARE CORRECT !!')
        ibex_model = simulate_ibex.create_fann_model(int_weights, int_og_bias, mul_vals, shift_vals)
        simulate_ibex.eval_sim_model(quant_net, ibex_model, test_loader)
    
    elif(name == 'uci_fann_net'):
        print('\nSIMULATING MODEL ON IBEX CORE\nUSE THE OUTPUTS TO VERIFY THAT THE RESULTS ARE CORRECT !!')
        ibex_model = simulate_ibex.create_uci_model(int_weights, int_og_bias, mul_vals, shift_vals)
        simulate_ibex.eval_sim_model(quant_net, ibex_model, test_loader)
    
    elif(name == 'lenet5_mnist'):
        print('\nSIMULATING MODEL ON IBEX CORE\nUSE THE OUTPUTS TO VERIFY THAT THE RESULTS ARE CORRECT !!')
        ibex_model = simulate_ibex.create_lenet_model(int_weights, int_og_bias, mul_vals, shift_vals)
        simulate_ibex.eval_sim_model(quant_net, ibex_model, test_loader)
