#ifndef FULLY_CONNECTED_OPT_H
#define FULLY_CONNECTED_OPT_H

#include <stdint.h>

void mlp_layer_2bits(int input[], int output[], int num_inputs, int num_outputs, const int weights[][num_inputs], const int bias[], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){

	// Compute the output for each neuron
    	int z, bias_val, w, inp, temp;
    	
    	for (int i = 0; i < num_outputs; i++) {
    		bias_val = bias[i];
    		asm volatile("neur_init %0, %1, %2\n":"=r"(z):"r"(bias_val),"r"(bias_shift_mode[i]):);
        	
        	for (int j = 0; j < num_inputs; j++) {
        		w = weights[i][j];
        		inp = input[j];
        		asm volatile("nn_mac_2b %0, %1,%2\n":"=r"(temp):"r"(w),"r"(inp):);
        	}
        	asm volatile("neur_res %0, %1, %2\n":"=r"(z):"r"(quantized_multiplier),"r"(out_shift_rl):);
        	output[i] = z;
 	}
}

void mlp_layer_4bits(int input[], int output[], int num_inputs, int num_outputs, const int weights[][num_inputs << 1], const int bias[], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){

	// Compute the output for each neuron
    	int z, bias_val, w, inp, temp;
    	
    	for (int i = 0; i < num_outputs; i++) {
    		bias_val = bias[i];
    		asm volatile("neur_init %0, %1, %2\n":"=r"(z):"r"(bias_val),"r"(bias_shift_mode[i]):);
        	
        	for (int j = 0; j < num_inputs; j++) {
        		w = weights[i][2*j];
        		inp = input[j];
        		asm volatile("nn_mac_4b %0, %1,%2\n":"=r"(temp):"r"(w),"r"(inp):);
        		
        		w = weights[i][2*j+1];
        		asm volatile("nn_mac_4b %0, %1,%2\n":"=r"(temp):"r"(w),"r"(inp):);
        		
        	}
        	asm volatile("neur_res %0, %1, %2\n":"=r"(z):"r"(quantized_multiplier),"r"(out_shift_rl):);
        	output[i] = z;
 	}
}

void mlp_layer_8bits(int input[], int output[], int num_inputs, int num_outputs, const int weights[][num_inputs << 2], const int bias[], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){

	// Compute the output for each neuron
    	int z, bias_val, w, inp, temp;
    	
    	for (int i = 0; i < num_outputs; i++) {
    		bias_val = bias[i];
    		asm volatile("neur_init %0, %1, %2\n":"=r"(z):"r"(bias_val),"r"(bias_shift_mode[i]):);
        	
        	for (int j = 0; j < num_inputs; j++) {
        		w = weights[i][4*j];
        		inp = input[j];
        		asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(temp):"r"(w),"r"(inp):);
        		
        		w = weights[i][4*j+1];
        		asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(temp):"r"(w),"r"(inp):);
        		
        		w = weights[i][4*j+2];
        		asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(temp):"r"(w),"r"(inp):);
        		
        		w = weights[i][4*j+3];
        		asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(temp):"r"(w),"r"(inp):);        		
        	}
        	
        	asm volatile("neur_res %0, %1, %2\n":"=r"(z):"r"(quantized_multiplier),"r"(out_shift_rl):);
        	output[i] = z;
 	}
}

#endif /* FULLY_CONNECTED_OPT_H */
