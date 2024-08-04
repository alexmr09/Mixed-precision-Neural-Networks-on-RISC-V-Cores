#ifndef CONV2D_OPT_H
#define CONV2D_OPT_H

void conv2_8bits(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], const int fil[fil_dim[0]][fil_dim[1]][fil_dim[2]][fil_dim[3] << 2], const int bias[fil_dim[0]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[4], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){

	int i, j, k, m, n, p, res, k1, k2, str1, str2, w, in_cnn, bias_val;
	
	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0] - strides;
		for (j = 0; j < out_dim[0]; j++) {  // output height
			str1 += strides;
			str2 = -pad[2] - strides;
	        	for (k = 0; k < out_dim[1]; k++) {  // output width
                    		bias_val = bias[i];
                    		asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[i]):);
                    		str2 += strides;
                    		for (p = 0; p < fil_dim[1]; p++) {  // filters height
                            		for (n = 0; n < fil_dim[2]; n++) {  // filters width
                            			k1 = str1 + p; 
                            			k2 = str2 + n;
                            				
                            			if (k1 < in_dim[0] && k1 >= 0 && k2 >= 0 && k2 < in_dim[1]) { 
							for (m = 0; m < fil_dim[3]; m++) {   // filters depth
                            					in_cnn = inp[k1][k2][m];
                            					w = fil[i][p][n][4*m];
                            					asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            					w = fil[i][p][n][4*m+1];
                            					asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            					w = fil[i][p][n][4*m+2];
                            					asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            					w = fil[i][p][n][4*m+3];
                            					asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            				}
                                		}
                        		}
                    		}
                    		asm volatile("neur_res %0, %1, %2\n":"=r"(res):"r"(quantized_multiplier),"r"(out_shift_rl):);
                    		out[j][k][i] = res;
            		}
        	}
    	}
}

void conv2_8bits_1ch(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], const int fil[fil_dim[0]][fil_dim[1]][fil_dim[2]][fil_dim[3] << 2], const int bias[fil_dim[0]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[4], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){

	int i, j, k, n, p, res, k1, k2, str1, str2, w, in_cnn, bias_val;
	
	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0] - strides;
		for (j = 0; j < out_dim[0]; j++) {  // output height
			str1 += strides;
			str2 = -pad[2] - strides;
	        	for (k = 0; k < out_dim[1]; k++) {  // output width
                    		bias_val = bias[i];
                    		asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[i]):);
                    		str2 += strides;
                    		for (p = 0; p < fil_dim[1]; p++) {  // filters height
                            		for (n = 0; n < fil_dim[2]; n++) {  // filters width
                            			k1 = str1 + p; 
                            			k2 = str2 + n;
                            				
                            			if (k1 < in_dim[0] && k1 >= 0 && k2 >= 0 && k2 < in_dim[1]) { 
                            			          in_cnn = inp[k1][k2][0];
                            				  w = fil[i][p][n][0];
                            				  asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            				  w = fil[i][p][n][1];
                            				  asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            				  w = fil[i][p][n][2];
                            				  asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            				  w = fil[i][p][n][3];
                            				  asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                                		}
                        		}
                    		}
                    		asm volatile("neur_res %0, %1, %2\n":"=r"(res):"r"(quantized_multiplier),"r"(out_shift_rl):);
                    		out[j][k][i] = res;
            		}
        	}
    	}
}

void conv2_4bits(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], const int fil[fil_dim[0]][fil_dim[1]][fil_dim[2]][fil_dim[3] << 1], const int bias[fil_dim[0]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[4], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){

	int i, j, k, m, n, p, res, k1, k2, str1, str2, w, in_cnn, bias_val;
	
	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0];
		for (j = 0; j < out_dim[0]; j++) {  // output height
			if (j != 0) str1 += strides;
			str2 = -pad[2];
	        	for (k = 0; k < out_dim[1]; k++) {  // output width
                    		bias_val = bias[i];
                    		asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[i]):);
                    		if (k != 0) str2 += strides;
                    		for (p = 0; p < fil_dim[1]; p++) {  // filters height
                            		for (n = 0; n < fil_dim[2]; n++) {  // filters width
                            			k1 = str1 + p; 
                            			k2 = str2 + n;
                            				
                            			if (k1 < in_dim[0] && k1 >= 0 && k2 >= 0 && k2 < in_dim[1]) { 
							for (m = 0; m < fil_dim[3]; m++) {   // filters depth
                            					in_cnn = inp[k1][k2][m];
                            					w = fil[i][p][n][2*m];
                            					asm volatile("nn_mac_4b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            					w = fil[i][p][n][2*m+1];
                            					asm volatile("nn_mac_4b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            				}
                                		}
                        		}
                    		}
                    		asm volatile("neur_res %0, %1, %2\n":"=r"(res):"r"(quantized_multiplier),"r"(out_shift_rl):);
                    		out[j][k][i] = res;
            		}
        	}
    	}    	
}

void conv2_4bits_1ch(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], const int fil[fil_dim[0]][fil_dim[1]][fil_dim[2]][fil_dim[3] << 1], const int bias[fil_dim[0]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[4], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){

	int i, j, k, n, p, res, k1, k2, str1, str2, w, in_cnn, bias_val;
	
	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0] - strides;
		for (j = 0; j < out_dim[0]; j++) {  // output height
			str1 += strides;
			str2 = -pad[2] - strides;
	        	for (k = 0; k < out_dim[1]; k++) {  // output width
                    		bias_val = bias[i];
                    		asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[i]):);
                    		str2 += strides;
                    		for (p = 0; p < fil_dim[1]; p++) {  // filters height
                            		for (n = 0; n < fil_dim[2]; n++) {  // filters width
                            			k1 = str1 + p; 
                            			k2 = str2 + n;
                            				
                            			if (k1 < in_dim[0] && k1 >= 0 && k2 >= 0 && k2 < in_dim[1]) {
                            				in_cnn = inp[k1][k2][0];
                            				w = fil[i][p][n][0];
                            				asm volatile("nn_mac_2b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            				
                            				w = fil[i][p][n][1];
                            				asm volatile("nn_mac_2b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                                		}
                        		}
                    		}
                    		asm volatile("neur_res %0, %1, %2\n":"=r"(res):"r"(quantized_multiplier),"r"(out_shift_rl):);
                    		out[j][k][i] = res;
            		}
        	}
    	}    	
}

void conv2_2bits(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], const int fil[fil_dim[0]][fil_dim[1]][fil_dim[2]][fil_dim[3]], const int bias[fil_dim[0]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[4], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){

	int i, j, k, m, n, p, res, k1, k2, str1, str2, w, in_cnn, bias_val;
	
	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0] - strides;
		for (j = 0; j < out_dim[0]; j++) {  // output height
			str1 += strides;
			str2 = -pad[2] - strides;
	        	for (k = 0; k < out_dim[1]; k++) {  // output width
                    		bias_val = bias[i];
                    		asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[i]):);
                    		str2 += strides;
                    		for (p = 0; p < fil_dim[1]; p++) {  // filters height
                            		for (n = 0; n < fil_dim[2]; n++) {  // filters width
                            			k1 = str1 + p; 
                            			k2 = str2 + n;
                            				
                            			if (k1 < in_dim[0] && k1 >= 0 && k2 >= 0 && k2 < in_dim[1]) { 
							for (m = 0; m < fil_dim[3]; m++) {   // filters depth
                            					in_cnn = inp[k1][k2][m];
                            					w = fil[i][p][n][m];
                            					asm volatile("nn_mac_2b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            				}
                                		}
                        		}
                    		}
                    		asm volatile("neur_res %0, %1, %2\n":"=r"(res):"r"(quantized_multiplier),"r"(out_shift_rl):);
                    		out[j][k][i] = res;
            		}
        	}
    	}
}


void conv2_2bits_1ch(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][1], const int fil[fil_dim[0]][fil_dim[1]][fil_dim[2]][1], const int bias[fil_dim[0]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[4], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){

	int i, j, k, n, p, res, k1, k2, str1, str2, w, in_cnn, bias_val;
	
	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0] -strides;
		for (j = 0; j < out_dim[0]; j++) {  // output height
			str1 += strides;
			str2 = -pad[2] - strides;
	        	for (k = 0; k < out_dim[1]; k++) {  // output width
                    		bias_val = bias[i];
                    		asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[i]):);
                    		str2 += strides;
                    		for (p = 0; p < fil_dim[1]; p++) {  // filters height
                            		for (n = 0; n < fil_dim[2]; n++) {  // filters width
                            			k1 = str1 + p; 
                            			k2 = str2 + n;
                            				
                            			if (k1 < in_dim[0] && k1 >= 0 && k2 >= 0 && k2 < in_dim[1]) { 
                            				in_cnn = inp[k1][k2][0];
                            				w = fil[i][p][n][0];
                            				asm volatile("nn_mac_2b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                                		}
                        		}
                    		}
    				
                    		asm volatile("neur_res %0, %1, %2\n":"=r"(res):"r"(quantized_multiplier),"r"(out_shift_rl):);
                    		out[j][k][i] = res;
            		}
        	}
    	}
}

void maxpool2_compressed_signed(int in_dim[3], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], 
                                    int out[out_dim[0]][out_dim[1]][out_dim[2]], int pool_size, int strides) {

	int i, j, m, n, d, k1, k2, str1, str2, c;
	int8_t value1, value2, value3, value4;
	int8_t max_value1, max_value2, max_value3, max_value4;
	
    	for (d = 0; d < out_dim[2]; d++) {
    		str1 = 0;
        	for (i = 0; i < out_dim[0]; i++) {
        		if (i != 0) str1 += strides;
        		str2 = 0;
            		for (j = 0; j < out_dim[1]; j++) {
            			if (j != 0) str2 += strides;
                		max_value1 = -128;
                		max_value2 = -128;
                		max_value3 = -128;
                		max_value4 = -128;

                		for (m = 0; m < pool_size; m++) {
                    			for (n = 0; n < pool_size; n++) {
                    				k1 = str1 + m;
                    				k2 = str2 + n;
                    				if (k1 >= 0 && k2 >=0 && k1 < in_dim[0] && k2 < in_dim[1]){
                         				value1 = (inp[k1][k2][d] & 0xFF000000) >> 24;
                         				value2 = (inp[k1][k2][d] & 0x00FF0000) >> 16;
                         				value3 = (inp[k1][k2][d] & 0x0000FF00) >> 8;
                         				value4 = inp[k1][k2][d] & 0x000000FF;
                         				
                        				if (value1 > max_value1) {
                            					max_value1 = value1;
                        				}
                        				
                        				if (value2 > max_value2) {
                        					max_value2 = value2;
                        				}
                        				
                        				if (value3 > max_value3) {
                            					max_value3 = value3;
                        				}
                        				
                        				if (value4 > max_value4) {
                        					max_value4 = value4;
                        				}
                        				
                        			}
                    			}
                		}
                		
                		c = ((max_value1 & 0xFF) << 24)|((max_value2 & 0xFF) << 16)|((max_value3 & 0xFF) << 8)|((max_value4 & 0xFF));
                		out[i][j][d] = c;
            		}
        	}
    	}	
}

void maxpool2_compressed_unsigned(int in_dim[3], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], 
                                int out[out_dim[0]][out_dim[1]][out_dim[2]], int pool_size, int strides) {

	int i, j, m, n, d, k1, k2, str1, str2;
	uint32_t value1, value2, value3, value4;
	uint32_t max_value1, max_value2, max_value3, max_value4, c;
	
    	for (d = 0; d < out_dim[2]; d++) {
    		str1 = 0;
        	for (i = 0; i < out_dim[0]; i++) {
        		if (i != 0) str1 += strides;
        		str2 = 0;
            		for (j = 0; j < out_dim[1]; j++) {
            			if (j != 0) str2 += strides;
                		max_value1 = 0;
                		max_value2 = 0;
                		max_value3 = 0;
                		max_value4 = 0;

                		for (m = 0; m < pool_size; m++) {
                    			for (n = 0; n < pool_size; n++) {
                    				k1 = str1 + m;
                    				k2 = str2 + n;
                    				if (k1 >= 0 && k2 >=0 && k1 < in_dim[0] && k2 < in_dim[1]){
                         				value1 = inp[k1][k2][d] & 0xFF000000;
                         				value2 = inp[k1][k2][d] & 0x00FF0000;
                         				value3 = inp[k1][k2][d] & 0x0000FF00;
                         				value4 = inp[k1][k2][d] & 0x000000FF;
                         				
                        				if (value1 > max_value1) {
                            					max_value1 = value1;
                        				}
                        				
                        				if (value2 > max_value2) {
                        					max_value2 = value2;
                        				}
                        				
                        				if (value3 > max_value3) {
                            					max_value3 = value3;
                        				}
                        				
                        				if (value4 > max_value4) {
                        					max_value4 = value4;
                        				}
                        				
                        			}
                    			}
                		}
                		
                		c = max_value1 | max_value2 | max_value3 | max_value4;
                		out[i][j][d] = c;
            		}
        	}
    	}	
}

void avgpool2_compressed_signed(int in_dim[3], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int pool_size, int strides) {

	int i, j, m, n, d, k1, k2, str1, str2;
	int avg_value1, avg_value2, avg_value3, avg_value4;
        int8_t v1, v2, v3, v4;

    	for (d = 0; d < out_dim[2]; d++) {
    		str1 = -strides;
        	for (i = 0; i < out_dim[0]; i++) {
        		str1 += strides;
        		str2 = -strides;
            		for (j = 0; j < out_dim[1]; j++) {
            			str2 += strides;
                		avg_value1 = 0;
                		avg_value2 = 0;
                		avg_value3 = 0;
                		avg_value4 = 0;

                		for (m = 0; m < pool_size; m++) {
                    			for (n = 0; n < pool_size; n++) {
                    				k1 = str1 + m;
                    				k2 = str2 + n;
                    				if (k1 >= 0 && k2 >=0 && k1 < in_dim[0] && k2 < in_dim[1]){
                    				        v1 = (inp[k1][k2][d] & 0xFF000000) >> 24;
                         				avg_value1 += v1;
                         				
                         				v2 = (inp[k1][k2][d] & 0x00FF0000) >> 16;
                         				avg_value2 += v2;
                         				
                         				v3 = (inp[k1][k2][d] & 0x0000FF00) >> 8;
                         				avg_value3 += v3;
                         				
                         				v4 = inp[k1][k2][d]  & 0x000000FF;
                         				avg_value4 += v4;
                         				
                        			}
                    			}
                		}
                		
                		avg_value1 = avg_value1 / (pool_size * pool_size);
                		avg_value2 = avg_value2 / (pool_size * pool_size);
                		avg_value3 = avg_value3 / (pool_size * pool_size);
                		avg_value4 = avg_value4 / (pool_size * pool_size);
                		
                		out[i][j][d] = (((avg_value1 & 0xFF) << 24) | ((avg_value2 & 0xFF)  << 16) | ((avg_value3 & 0xFF)  << 8) | ((avg_value4 & 0xFF)));
            		}
        	}
    	}
}

void avgpool2_compressed_unsigned(int in_dim[3], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], int out[out_dim[0]][out_dim[1]][out_dim[2]], int pool_size, int strides) {

	int i, j, m, n, d, k1, k2, str1, str2;
	int avg_value1, avg_value2, avg_value3, avg_value4;

    	for (d = 0; d < out_dim[2]; d++) {
    		str1 = -strides;
        	for (i = 0; i < out_dim[0]; i++) {
        		str1 += strides;
        		str2 = -strides;
            		for (j = 0; j < out_dim[1]; j++) {
            			str2 += strides;
                		avg_value1 = 0;
                		avg_value2 = 0;
                		avg_value3 = 0;
                		avg_value4 = 0;

                		for (m = 0; m < pool_size; m++) {
                    			for (n = 0; n < pool_size; n++) {
                    				k1 = str1 + m;
                    				k2 = str2 + n;
                    				if (k1 >= 0 && k2 >=0 && k1 < in_dim[0] && k2 < in_dim[1]){
                         				avg_value1 += ((inp[k1][k2][d] & 0xFF000000) >> 24);
                         				avg_value2 += ((inp[k1][k2][d] & 0x00FF0000) >> 16);
                         				avg_value3 += ((inp[k1][k2][d] & 0x0000FF00) >> 8);
                         				avg_value4 += (inp[k1][k2][d]  & 0x000000FF);
                         				
                        			}
                    			}
                		}
                		
                		avg_value1 = avg_value1 / (pool_size * pool_size);
                		avg_value2 = avg_value2 / (pool_size * pool_size);
                		avg_value3 = avg_value3 / (pool_size * pool_size);
                		avg_value4 = avg_value4 / (pool_size * pool_size);
                		
                		out[i][j][d] = ((avg_value1 << 24) | (avg_value2 << 16) | (avg_value3 << 8) | (avg_value4)) ;
            		}
        	}
    	}
}

void flatten(int in_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], int out[]){

	int index = 0;
	
	int values[in_dim[0]][in_dim[1]][in_dim[2] << 2];
	
	for (int i = 0; i < in_dim[0]; i++){
		for(int j = 0; j < in_dim[1]; j++){
			for(int k = 0; k < in_dim[2]; k++){
				values[i][j][4*k]   = (inp[i][j][k] & 0xFF000000) >> 24;
				values[i][j][4*k+1] = (inp[i][j][k] & 0x00FF0000) >> 16;
				values[i][j][4*k+2] = (inp[i][j][k] & 0x0000FF00) >> 8;
				values[i][j][4*k+3] =  inp[i][j][k] & 0x000000FF;
			}
		}
	}
	
	int out_dim = (in_dim[0] * in_dim[1] * in_dim[2]) << 2;
	int flatten_matrix[out_dim];
	
	for (int k = 0; k < in_dim[2] << 2; k++){
		for(int j = 0; j < in_dim[0]; j++){
			for(int i = 0; i < in_dim[1]; i++){
				flatten_matrix[index++] = values[j][i][k];
			}
		}
	}
	
	for(int i = 0; i < out_dim >> 2; i++){
		out[i] = (flatten_matrix[4*i] << 24 | flatten_matrix[4*i+1] << 16 | flatten_matrix[4*i+2] << 8 | flatten_matrix[4*i+3]);
	}	
}

#endif  /* CONV2D_OPT_H */
