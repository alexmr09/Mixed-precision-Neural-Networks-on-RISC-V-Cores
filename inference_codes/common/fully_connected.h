#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <stdint.h>

void mlp_layer(int input[], int output[], int num_inputs, int num_outputs, const int weights[][num_inputs], const int bias[], const int bias_shift_mode, const int quantized_multiplier, const int out_shift_rl){

	// Compute the output for each neuron
    	int z, w, inp, quant_prod;
    	
    	for (int i = 0; i < num_outputs; i++) {
    		z = bias[i];
        	
        	for (int j = 0; j < num_inputs; j++) {
        		w = weights[i][j];
        		inp = input[j];
        		z += w*inp;
        	}
        	quant_prod = quantized_multiplier * z + (1 << (out_shift_rl-1));
        	quant_prod = quant_prod >> out_shift_rl;
        
        	if(quant_prod < 0) quant_prod = 0;
        	if(quant_prod > 255) quant_prod = 255;
        	
        	output[i] = quant_prod;
 	}
}
#endif /* FULLY_CONNECTED_H */
