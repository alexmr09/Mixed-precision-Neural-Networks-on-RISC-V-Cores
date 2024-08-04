#include "simple_system_common.h"
#include "cnn_weights.h"
#include "fully_connected.h"
#include "ibex_cnn_params.h"
#include "ibex_inputs.h"
#include "conv2d.h"
#include "dws_conv.h"

#define IMG_SZ 32
#define NUM_FIL0 3

#define FILTER1 3
#define FILTER2 1
#define FILTER3 3
#define FILTER4 1
#define FILTER5 3
#define FILTER6 1
#define FILTER7 3
#define FILTER8 1
#define FILTER9 3
#define FILTER10 1
#define FILTER11 3
#define FILTER12 1

#define NUM_FIL1 3
#define NUM_FIL2 64
#define NUM_FIL3 64
#define NUM_FIL4 64
#define NUM_FIL5 64
#define NUM_FIL6 128
#define NUM_FIL7 128
#define NUM_FIL8 128
#define NUM_FIL9 128
#define NUM_FIL10 256
#define NUM_FIL11 256
#define NUM_FIL12 256

#define STRIDE1 1
#define STRIDE2 1
#define STRIDE3 1
#define STRIDE4 1
#define STRIDE5 1
#define STRIDE6 1
#define STRIDE7 1
#define STRIDE8 1
#define STRIDE9 1
#define STRIDE10 1
#define STRIDE11 1
#define STRIDE12 1

#define PAD_TB1 1
#define PAD_LR1 1

#define PAD_TB2 0
#define PAD_LR2 0

#define PAD_TB3 1
#define PAD_LR3 1

#define PAD_TB4 0
#define PAD_LR4 0

#define PAD_TB5 1
#define PAD_LR5 1

#define PAD_TB6 0
#define PAD_LR6 0

#define PAD_TB7 1
#define PAD_LR7 1

#define PAD_TB8 0
#define PAD_LR8 0

#define PAD_TB9 1
#define PAD_LR9 1

#define PAD_TB10 0
#define PAD_LR10 0

#define PAD_TB11 1
#define PAD_LR11 1

#define PAD_TB12 0
#define PAD_LR12 0

#define POOL_STRIDE1 2
#define POOL_SIZE1 2

#define POOL_STRIDE2 2
#define POOL_SIZE2 2

#define POOL_STRIDE3 2
#define POOL_SIZE3 2

#define OUT_DIM 10

#define SAMPLES 1
int outs[SAMPLES][OUT_DIM];

void cifar10_dws_cnn() {

	int dout1 = NUM_FIL1;
	int hout1 = ((IMG_SZ - FILTER1 + 2 * PAD_TB1)/STRIDE1) + 1;
	int wout1 = ((IMG_SZ - FILTER1 + 2 * PAD_LR1)/STRIDE1) + 1;

	int dout2 = NUM_FIL2;
	int hout2 = ((hout1 - FILTER2+ 2 * PAD_TB2)/STRIDE2)+1;
	int wout2 = ((wout1 - FILTER2+ 2 * PAD_LR2)/STRIDE2)+1;

	int dout3 = NUM_FIL3;
	int hout3 = ((hout2 - FILTER3+ 2 * PAD_TB3)/STRIDE3)+1;
	int wout3 = ((wout2 - FILTER3+ 2 * PAD_LR3)/STRIDE3)+1;

	int dout4 = NUM_FIL4;
	int hout4 = ((hout3 - FILTER4+ 2 * PAD_TB4)/STRIDE4)+1;
	int wout4 = ((wout3 - FILTER4+ 2 * PAD_LR4)/STRIDE4)+1;

	int dout5 = dout4;
	int hout5 = hout4/POOL_STRIDE1;
	int wout5 = wout4/POOL_STRIDE1;

	int dout6 = NUM_FIL5;
	int hout6 = ((hout5 - FILTER5+ 2 * PAD_TB5)/STRIDE5)+1;
	int wout6 = ((wout5 - FILTER5+ 2 * PAD_LR5)/STRIDE5)+1;

	int dout7 = NUM_FIL6;
	int hout7 = ((hout6 - FILTER6+ 2 * PAD_TB6)/STRIDE6)+1;
	int wout7 = ((wout6 - FILTER6+ 2 * PAD_LR6)/STRIDE6)+1;

	int dout8 = NUM_FIL7;
	int hout8 = ((hout7 - FILTER7+ 2 * PAD_TB7)/STRIDE7)+1;
	int wout8 = ((wout7 - FILTER7+ 2 * PAD_LR7)/STRIDE7)+1;

	int dout9 = NUM_FIL8;
	int hout9 = ((hout8 - FILTER8+ 2 * PAD_TB8)/STRIDE8)+1;
	int wout9 = ((wout8 - FILTER8+ 2 * PAD_LR8)/STRIDE8)+1;

	int dout10 = dout9;
	int hout10 = hout9/POOL_STRIDE2;
	int wout10 = wout9/POOL_STRIDE2;

	int dout11 = NUM_FIL9;
	int hout11 = ((hout10 - FILTER9+ 2 * PAD_TB9)/STRIDE9)+1;
	int wout11 = ((wout10 - FILTER9+ 2 * PAD_LR9)/STRIDE9)+1;

	int dout12 = NUM_FIL10;
	int hout12 = ((hout11 - FILTER10+ 2 * PAD_TB10)/STRIDE10)+1;
	int wout12 = ((wout11 - FILTER10+ 2 * PAD_LR10)/STRIDE10)+1;

	int dout13 = NUM_FIL11;
	int hout13 = ((hout12 - FILTER11+ 2 * PAD_TB11)/STRIDE11)+1;
	int wout13 = ((wout12 - FILTER11+ 2 * PAD_LR11)/STRIDE11)+1;

	int dout14 = NUM_FIL12;
	int hout14 = ((hout13 - FILTER12+ 2 * PAD_TB12)/STRIDE12)+1;
	int wout14 = ((wout13 - FILTER12+ 2 * PAD_LR12)/STRIDE12)+1;

	int dout15 = dout14;
	int hout15 = hout14/POOL_STRIDE3;
	int wout15 = wout14/POOL_STRIDE3;

	int flatten_dim = dout15 * hout15 * wout15;

	int in[IMG_SZ][IMG_SZ][NUM_FIL0];
	int inp_dim[3] = {IMG_SZ, IMG_SZ, NUM_FIL0};

	int out1[hout1][wout1][dout1];
	int pad_1[4] = {PAD_TB1, PAD_TB1, PAD_LR1, PAD_LR1};
	int outp_dim1[3] = {hout1, wout1, dout1};
	int f_dim1[4] = {NUM_FIL1, FILTER1, FILTER1, NUM_FIL0};

	int out2[hout2][wout2][dout2];
	int pad_2[4] = {PAD_TB2, PAD_TB2, PAD_LR2, PAD_LR2};
	int outp_dim2[3] = {hout2, wout2, dout2};
	int f_dim2[4] = {NUM_FIL2, FILTER2, FILTER2, NUM_FIL1};

	int out3[hout3][wout3][dout3];
	int pad_3[4] = {PAD_TB3, PAD_TB3, PAD_LR3, PAD_LR3};
	int outp_dim3[3] = {hout3, wout3, dout3};
	int f_dim3[4] = {NUM_FIL3, FILTER3, FILTER3, NUM_FIL2};

	int out4[hout4][wout4][dout4];
	int pad_4[4] = {PAD_TB4, PAD_TB4, PAD_LR4, PAD_LR4};
	int outp_dim4[3] = {hout4, wout4, dout4};
	int f_dim4[4] = {NUM_FIL4, FILTER4, FILTER4, NUM_FIL3};

	int out5[hout5][wout5][dout5];
	int outp_dim5[3] = {hout5, wout5, dout5};

	int out6[hout6][wout6][dout6];
	int pad_6[4] = {PAD_TB5, PAD_TB5, PAD_LR5, PAD_LR5};
	int outp_dim6[3] = {hout6, wout6, dout6};
	int f_dim6[4] = {NUM_FIL5, FILTER5, FILTER5, NUM_FIL4};

	int out7[hout7][wout7][dout7];
	int pad_7[4] = {PAD_TB6, PAD_TB6, PAD_LR6, PAD_LR6};
	int outp_dim7[3] = {hout7, wout7, dout7};
	int f_dim7[4] = {NUM_FIL6, FILTER6, FILTER6, NUM_FIL5};

	int out8[hout8][wout8][dout8];
	int pad_8[4] = {PAD_TB7, PAD_TB7, PAD_LR7, PAD_LR7};
	int outp_dim8[3] = {hout8, wout8, dout8};
	int f_dim8[4] = {NUM_FIL7, FILTER7, FILTER7, NUM_FIL6};

	int out9[hout9][wout9][dout9];
	int pad_9[4] = {PAD_TB8, PAD_TB8, PAD_LR8, PAD_LR8};
	int outp_dim9[3] = {hout9, wout9, dout9};
	int f_dim9[4] = {NUM_FIL8, FILTER8, FILTER8, NUM_FIL7};

	int out10[hout10][wout10][dout10];
	int outp_dim10[3] = {hout10, wout10, dout10};

	int out11[hout11][wout11][dout11];
	int pad_11[4] = {PAD_TB9, PAD_TB9, PAD_LR9, PAD_LR9};
	int outp_dim11[3] = {hout11, wout11, dout11};
	int f_dim11[4] = {NUM_FIL9, FILTER9, FILTER9, NUM_FIL8};

	int out12[hout12][wout12][dout12];
	int pad_12[4] = {PAD_TB10, PAD_TB10, PAD_LR10, PAD_LR10};
	int outp_dim12[3] = {hout12, wout12, dout12};
	int f_dim12[4] = {NUM_FIL10, FILTER10, FILTER10, NUM_FIL9};

	int out13[hout13][wout13][dout13];
	int pad_13[4] = {PAD_TB11, PAD_TB11, PAD_LR11, PAD_LR11};
	int outp_dim13[3] = {hout13, wout13, dout13};
	int f_dim13[4] = {NUM_FIL11, FILTER11, FILTER11, NUM_FIL10};

	int out14[hout14][wout14][dout14];
	int pad_14[4] = {PAD_TB12, PAD_TB12, PAD_LR12, PAD_LR12};
	int outp_dim14[3] = {hout14, wout14, dout14};
	int f_dim14[4] = {NUM_FIL12, FILTER12, FILTER12, NUM_FIL11};

	int out15[hout15][wout15][dout15];
	int outp_dim15[3] = {hout15, wout15, dout15};

	int out16[flatten_dim];


	int out[OUT_DIM];

	for (int iter = 0; iter < SAMPLES; iter++){

		for(int i = 0; i < IMG_SZ; i++){
			for(int j = 0; j < IMG_SZ; j++){
				for(int k = 0; k < NUM_FIL0; k++){
					in[i][j][k] = input[i][j][k][iter];
				}
			}
		}

		pcount_enable(1);

		dw_conv(inp_dim, f_dim1, outp_dim1, in, F1, B1, out1, STRIDE1, pad_1, SB1, MV1, SV1, 1);
		pw_conv(outp_dim1, f_dim2, outp_dim2, out1, F2, B2, out2, STRIDE2, pad_2, SB2, MV2, SV2, 1);
		dw_conv(outp_dim2, f_dim3, outp_dim3, out2, F3, B3, out3, STRIDE3, pad_3, SB3, MV3, SV3, 1);
		pw_conv(outp_dim3, f_dim4, outp_dim4, out3, F4, B4, out4, STRIDE4, pad_4, SB4, MV4, SV4, 1);
		maxpool2(outp_dim4, outp_dim5, out4, out5, POOL_SIZE1, POOL_STRIDE1);

		dw_conv(outp_dim5, f_dim6, outp_dim6, out5, F5, B5, out6, STRIDE5, pad_6, SB5, MV5, SV5, 1);
		pw_conv(outp_dim6, f_dim7, outp_dim7, out6, F6, B6, out7, STRIDE6, pad_7, SB6, MV6, SV6, 1);
		dw_conv(outp_dim7, f_dim8, outp_dim8, out7, F7, B7, out8, STRIDE7, pad_8, SB7, MV7, SV7, 1);
		pw_conv(outp_dim8, f_dim9, outp_dim9, out8, F8, B8, out9, STRIDE8, pad_9, SB8, MV8, SV8, 1);
		maxpool2(outp_dim9, outp_dim10, out9, out10, POOL_SIZE2, POOL_STRIDE2);

		dw_conv(outp_dim10, f_dim11, outp_dim11, out10, F9, B9, out11, STRIDE9, pad_11, SB9, MV9, SV9, 1);
		pw_conv(outp_dim11, f_dim12, outp_dim12, out11, F10, B10, out12, STRIDE10, pad_12, SB10, MV10, SV10, 1);
		dw_conv(outp_dim12, f_dim13, outp_dim13, out12, F11, B11, out13, STRIDE11, pad_13, SB11, MV11, SV11, 1);
		pw_conv(outp_dim13, f_dim14, outp_dim14, out13, F12, B12, out14, STRIDE12, pad_14, SB12, MV12, SV12, 1);
		maxpool2(outp_dim14, outp_dim15, out14, out15, POOL_SIZE3, POOL_STRIDE3);

		flatten(outp_dim15, out15, out16);

		mlp_layer(out16, out, flatten_dim, OUT_DIM, W1, B13, SB13, MV13, SV13, 1);
		pcount_enable(0);

		puts("Output Layer Values:\n");
		for(int i = 0; i < OUT_DIM; i++) {
			puthex(out[i]);
			puts("\n");
		}
	}
}

int main(void) {

	pcount_enable(0);

	cifar10_dws_cnn();

	return 0;
}
