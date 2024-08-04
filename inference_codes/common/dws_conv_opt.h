#ifndef DWS_CONV_OPT_H
#define DWS_CONV_OPT_H

#include <stdint.h>

void pw_conv_8bits(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], 
	const int fil[fil_dim[0]][fil_dim[3] << 2], const int bias[fil_dim[0]], 
	int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[], const int bias_shift_mode[],
	const int quantized_multiplier, const int out_shift_rl){

	int i, j, k, m, res, str1, str2, bias_val, w, in_cnn;

	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0] - strides;
		for (j = 0; j < out_dim[0]; j++) {  // output height
			str1 += 1;
			str2 = -pad[2] - strides;
	        	for (k = 0; k < out_dim[1]; k++) {  // output width
                		bias_val = bias[i];
                		str2 += 1;
                		asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[i]):);
				
				for (m = 0; m < fil_dim[3]; m++) {   // filters depth
                        		in_cnn = inp[str1][str2][m];
                            		w = fil[i][4*m];
                            		asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            		w = fil[i][4*m+1];
                            		asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            		w = fil[i][4*m+2];
                            		asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            		w = fil[i][4*m+3];
                            		asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                		}
                		asm volatile("neur_res %0, %1, %2\n":"=r"(res):"r"(quantized_multiplier),"r"(out_shift_rl):);
                    		out[j][k][i] = res;
			}
        	}
	}
}

void pw_conv_4bits(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], 
	const int fil[fil_dim[0]][fil_dim[3] << 1], const int bias[fil_dim[0]], 
	int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[], const int bias_shift_mode[],
	const int quantized_multiplier, const int out_shift_rl){

	int i, j, k, m, res, str1, str2, bias_val, w, in_cnn;

	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0] - strides;
		for (j = 0; j < out_dim[0]; j++) {  // output height
			str1 += 1;
			str2 = -pad[2] - strides;
	        	for (k = 0; k < out_dim[1]; k++) {  // output width
                		bias_val = bias[i];
                		str2 += 1;
                		asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[i]):);
				for (m = 0; m < fil_dim[3]; m++) {   // filters depth
                        		in_cnn = inp[str1][str2][m];
                            		w = fil[i][2*m];
                            		asm volatile("nn_mac_4b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                            					
                            		w = fil[i][2*m+1];
                            		asm volatile("nn_mac_4b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                    		}
                		asm volatile("neur_res %0, %1, %2\n":"=r"(res):"r"(quantized_multiplier),"r"(out_shift_rl):);
                    		out[j][k][i] = res;
			}
        	}
	}
}

void pw_conv_2bits(int in_dim[3], int fil_dim[4], int out_dim[3], int inp[in_dim[0]][in_dim[1]][in_dim[2]], 
	const int fil[fil_dim[0]][fil_dim[3]], const int bias[fil_dim[0]], 
	int out[out_dim[0]][out_dim[1]][out_dim[2]], int strides, int pad[], const int bias_shift_mode[],
	const int quantized_multiplier, const int out_shift_rl){

	int i, j, k, m, res, str1, str2, bias_val, w, in_cnn;

	for (i = 0; i < out_dim[2]; i++) {   // output depth
		str1 = -pad[0] - strides;
		for (j = 0; j < out_dim[0]; j++) {  // output height
			str1 += 1;
			str2 = -pad[2] - strides;
	        	for (k = 0; k < out_dim[1]; k++) {  // output width
                		bias_val = bias[i];
                		str2 += 1;
                		asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[i]):);
				for (m = 0; m < fil_dim[3]; m++) {   // filters depth
                        		in_cnn = inp[str1][str2][m];
                            		w = fil[i][m];
                            		asm volatile("nn_mac_2b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                    		}
                		asm volatile("neur_res %0, %1, %2\n":"=r"(res):"r"(quantized_multiplier),"r"(out_shift_rl):);
                    		out[j][k][i] = res;
			}
        	}
	}
}

void dw_conv_opt(int in_dim[3], int depthwise_fil_dim[4], int out_dim[3],
	int inp[in_dim[0]][in_dim[1]][in_dim[2]], const int depthwise_fil[depthwise_fil_dim[0]][depthwise_fil_dim[1]][depthwise_fil_dim[2]],
	const int bias[depthwise_fil_dim[0]], int out[out_dim[0]][out_dim[1]][out_dim[2]],
	int strides, int pad[], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){
    
	int i, j, k, n, p, res, k1, k2, str1, str2, bias_val, in_cnn, w;

    	// Depthwise convolution
    	for (i = 0; i < out_dim[2]; i++){   // output depth
        	str1 = -pad[0] - strides;
        	for (j = 0; j < out_dim[0]; j++) {  // output height
            		str1 += strides;
            		str2 = -pad[2] - strides;
            		for (k = 0; k < out_dim[1]; k++) {  // output width
                		bias_val = bias[i];
                		str2 += strides;
                		asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[i]):);
                		for (p = 0; p < depthwise_fil_dim[1]; p++) {  // depthwise filter height
                    			for (n = 0; n < depthwise_fil_dim[2]; n++) {  // depthwise filter width
                        			k1 = str1 + p; 
                        			k2 = str2 + n;
                        
                        			if (k1 < in_dim[0] && k1 >= 0 && k2 >= 0 && k2 < in_dim[1]) { 
                            				in_cnn = inp[k1][k2][i];
                            				w = depthwise_fil[i][p][n];
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

void dw_conv_opt_1ch(int in_dim[3], int depthwise_fil_dim[4], int out_dim[3],
	int inp[in_dim[0]][in_dim[1]][in_dim[2]], const int depthwise_fil[depthwise_fil_dim[0]][depthwise_fil_dim[1]][depthwise_fil_dim[2]],
	const int bias[depthwise_fil_dim[0]], int out[out_dim[0]][out_dim[1]][out_dim[2]],
	int strides, int pad[], const int bias_shift_mode[], const int quantized_multiplier, const int out_shift_rl){
    
        int j, k, n, p, res, k1, k2, str1, str2, bias_val, in_cnn, w;

    	// Depthwise convolution
        str1 = -pad[0] - strides;
        for (j = 0; j < out_dim[0]; j++) {  // output height
             str1 += strides;
             str2 = -pad[2] - strides;
             for (k = 0; k < out_dim[1]; k++) {  // output width
                  bias_val = bias[0];
                  str2 += strides;
                  asm volatile("neur_init %0, %1, %2\n":"=r"(res):"r"(bias_val),"r"(bias_shift_mode[0]):);
                  for (p = 0; p < depthwise_fil_dim[1]; p++) {  // depthwise filter height
                       for (n = 0; n < depthwise_fil_dim[2]; n++) {  // depthwise filter width
                            k1 = str1 + p; 
                            k2 = str2 + n;
                        
                            if (k1 < in_dim[0] && k1 >= 0 && k2 >= 0 && k2 < in_dim[1]) { 
                                 in_cnn = inp[k1][k2][0];
                            	 w = depthwise_fil[0][p][n];
                            	 asm volatile("nn_mac_8b %0, %1,%2\n":"=r"(res):"r"(w),"r"(in_cnn):);
                             }
                        }
                   }
                   asm volatile("neur_res %0, %1, %2\n":"=r"(res):"r"(quantized_multiplier),"r"(out_shift_rl):);
                   out[j][k][0] = res;
           }
      }
}

void shortcut(int mat_dim[3], int inA[mat_dim[0]][mat_dim[1]][mat_dim[2]], int inB[mat_dim[0]][mat_dim[1]][mat_dim[2]], 
	int out[mat_dim[0]][mat_dim[1]][mat_dim[2]]){
	
	int8_t va1,va2,va3,va4;
	int8_t vb1,vb2,vb3,vb4;
	int sum1, sum2, sum3, sum4;

	for(int i = 0; i < mat_dim[0]; i++){
		for(int j = 0; j < mat_dim[1]; j++){
			for(int k = 0; k < mat_dim[2]; k++){
				va1 = (inA[i][j][k] & 0xff000000) >> 24;
				va2 = (inA[i][j][k] & 0xff0000) >> 16;
				va3 = (inA[i][j][k] & 0xff00) >> 8;
				va4 = inA[i][j][k] & 0xff;
				
				vb1 = (inB[i][j][k] & 0xff000000) >> 24;
				vb2 = (inB[i][j][k] & 0xff0000) >> 16;
				vb3 = (inB[i][j][k] & 0xff00) >> 8;
				vb4 = inB[i][j][k] & 0xff;
				
				sum1 = va1 + vb1;
				if(sum1 < -128) sum1 = -128;
        			if(sum1 > 127) sum1 = 127;
				
				sum2 = va2 + vb2;
				if(sum2 < -128) sum2 = -128;
        			if(sum2 > 127) sum2 = 127;
        			
				sum3 = va3 + vb3;
				if(sum3 < -128) sum3 = -128;
        			if(sum3 > 127) sum3 = 127;
        			
				sum4 = va4 + vb4;
        			if(sum4 < -128) sum4 = -128;
        			if(sum4 > 127) sum4 = 127;
        			
        			out[i][j][k] = (((sum1 & 0xff) << 24) | ((sum2 & 0xff) << 16) | ((sum3 & 0xff) << 8) | (sum4 & 0xff));
			}
		}
	}

}

#endif  /* DWS_CONV_OPT_H */
