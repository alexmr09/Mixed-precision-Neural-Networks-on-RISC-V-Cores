#ifndef CONV2D_H
#define CONV2D_H

void conv2(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], const int fil[fil_dim[0]][fil_dim[1]][fil_dim[2]][fil_dim[3]], const int bias[fil_dim[0]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[4], const int bias_shift_mode, const int quantized_multiplier, const int out_shift_rl){

	int i, j, k, m, n, p, res, k1, k2, str1, str2, quant_prod;
	
	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0] - strides;
		for (j = 0; j < out_dim[0]; j++) {  // output height
			str1 += strides;
			str2 = -pad[2] - strides;
	        	for (k = 0; k < out_dim[1]; k++) {  // output width
                    		res = bias[i];
                    		str2 += strides;
                    		for (p = 0; p < fil_dim[1]; p++) {  // filters height
                            		for (n = 0; n < fil_dim[2]; n++) {  // filters width
                            			k1 = str1 + p; 
                            			k2 = str2 + n;
                            			if (k1 < in_dim[0] && k1 >= 0 && k2 >= 0 && k2 < in_dim[1]) { 
							for (m = 0; m < fil_dim[3]; m++) {   // filters depth
                            					res += inp[k1][k2][m] * fil[i][p][n][m];
                            				}
                                		}
                        		}
                    		}
                    		quant_prod = quantized_multiplier * res + (1 << (out_shift_rl-1));
        			quant_prod = quant_prod >> out_shift_rl;
        
        			if(quant_prod < 0) quant_prod = 0;
        			if(quant_prod > 255) quant_prod = 255;
                    		out[j][k][i] = quant_prod;
            		}
        	}
    	}
}

void maxpool2(int in_dim[3], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int pool_size, int strides) {

	int i, j, m, n, d, max_value, value, k1, k2, str1, str2;

    	for (d = 0; d < out_dim[2]; d++) {
    		str1 = 0;
        	for (i = 0; i < out_dim[0]; i++) {
        		if (i != 0) str1 += strides;
        		str2 = 0;
            		for (j = 0; j < out_dim[1]; j++) {
            			if (j != 0) str2 += strides;
                		max_value = 0;
                		
                		for (m = 0; m < pool_size; m++) {
                    			for (n = 0; n < pool_size; n++) {
                    				k1 = str1 + m;
                    				k2 = str2 + n;
                    				if (k1 >= 0 && k2 >=0 && k1 < in_dim[0] && k2 < in_dim[1]){
                         				value = inp[k1][k2][d];
                        				if (value > max_value) max_value = value;
                        			}
                    			}
                		}
                		out[i][j][d] = max_value;
            		}
        	}
    	}
}

void avgpool2(int in_dim[3], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int pool_size, int strides) {

	int i, j, m, n, d, avg_value, value, k1, k2, str1, str2;
	
    	for (d = 0; d < out_dim[2]; d++) {
    		str1 = 0;
        	for (i = 0; i < out_dim[0]; i++) {
        		if (i != 0) str1 += strides;
        		str2 = 0;
            		for (j = 0; j < out_dim[1]; j++) {
            			if (j != 0) str2 += strides;
                		avg_value = 0;

                		for (m = 0; m < pool_size; m++) {
                    			for (n = 0; n < pool_size; n++) {
                    				k1 = str1 + m;
                    				k2 = str2 + n;
                    				if (k1 >= 0 && k2 >=0 && k1 < in_dim[0] && k2 < in_dim[1]){
                         				value = inp[k1][k2][d];
                          				avg_value += value;
                        			}
                    			}
                		}
                		avg_value = avg_value / (pool_size * pool_size);
                		out[i][j][d] = avg_value;
            		}
        	}
    	}
}

void flatten(int in_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], int out[]){

	int index = 0;
	for (int i = 0; i < in_dim[2]; i++){
		for(int j = 0; j < in_dim[0]; j++){
			for(int k = 0; k < in_dim[1]; k++){
				out[index++] = inp[j][k][i];
			}
		}
	}
}

#endif  /* CONV2D_H */
