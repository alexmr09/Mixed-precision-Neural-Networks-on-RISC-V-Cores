#ifndef DWS_CONV_H
#define DWS_CONV_H

void pw_conv(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], 
	      const int fil[fil_dim[0]][fil_dim[3]], const int bias[], 
	      int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[], 
              const int bias_shift_mode, const int quantized_multiplier, const int out_shift_rl, int relu){

     int i, j, k, m, res, str1, str2, quant_prod;

     for (i = 0; i < out_dim[2]; i++) {   // output depth
           str1 = -pad[0] - strides;
           for (j = 0; j < out_dim[0]; j++) {  // output height
	        str1 += strides;
	        str2 = -pad[2] - strides;
	        for (k = 0; k < out_dim[1]; k++) {  // output width
                    res = bias[i];
                      str2 += strides;
                      if (str1 < in_dim[0] && str1 >= 0 && str2 >= 0 && str2 < in_dim[1]) { 
		           for (m = 0; m < fil_dim[3]; m++) {   // filters depth
                                res += inp[str1][str2][m] * fil[i][m];
                          }
                      }
                      quant_prod = quantized_multiplier * res + (1 << (out_shift_rl -1));
        	      quant_prod = quant_prod >> (out_shift_rl);
        	      
        	      if(relu == 1){
          	          if(quant_prod < 0) quant_prod = 0;
        	          if(quant_prod > 255) quant_prod = 255;
                      }
                
                      else{
                          if(quant_prod < -128) quant_prod = -128;
        	          if(quant_prod > 127) quant_prod = 127;
                      }
                      
                      out[j][k][i] = quant_prod;
	       }
          }
     }
}

void dw_conv(int in_dim[3], int depthwise_fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], 
            const int depthwise_fil[depthwise_fil_dim[0]][depthwise_fil_dim[1]][depthwise_fil_dim[2]][1], 
            const int bias[], int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[], 
            const int bias_shift_mode, const int depthwise_multiplier, const int depthwise_out_shift_rl, int relu){
    
	int i, j, k, n, p, res, k1, k2, str1, str2, quant_prod;

	// Depthwise convolution
	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0] - strides;
		for (j = 0; j < out_dim[0]; j++) {  // output height
			str1 += strides;
			str2 = -pad[2] - strides;
			for (k = 0; k < out_dim[1]; k++) {  // output width
				res = bias[i];
				str2 += strides;
				for (p = 0; p < depthwise_fil_dim[1]; p++){  // depthwise filter height
					for (n = 0; n < depthwise_fil_dim[2]; n++) {  // depthwise filter width
						k1 = str1 + p; 
						k2 = str2 + n;
                        
						if (k1 < in_dim[0] && k1 >= 0 && k2 >= 0 && k2 < in_dim[1]) { 
							res += inp[k1][k2][i] * depthwise_fil[i][p][n][0];
						}
					}
				}
				quant_prod = depthwise_multiplier * res + (1 << (depthwise_out_shift_rl -1));
		                quant_prod = quant_prod >> (depthwise_out_shift_rl);
				
				if(relu == 1){
          	                    if(quant_prod < 0) quant_prod = 0;
        	                    if(quant_prod > 255) quant_prod = 255;
                                }
                
                                else{
                                    if(quant_prod < -128) quant_prod = -128;
        	                    if(quant_prod > 127) quant_prod = 127;
                                }
                                
                                out[j][k][i] = quant_prod;
                        }
		}
	}
}


void shortcut(int mat_dim[3], int inA[mat_dim[0]][mat_dim[1]][mat_dim[2]], int inB[mat_dim[0]][mat_dim[1]][mat_dim[2]], 
		int out[mat_dim[0]][mat_dim[1]][mat_dim[2]]){
	
	int sum;
	for(int i = 0; i < mat_dim[0]; i++){
		for(int j = 0; j < mat_dim[1]; j++){
			for(int k = 0; k < mat_dim[2]; k++){
				sum = inA[i][j][k] + inB[i][j][k];
				if(sum < -128) sum = -128;
        		if(sum > 127) sum = 127;
                out[i][j][k] = sum;
			}
		}
	}
}

#endif  /* DWS_CONV_H */
