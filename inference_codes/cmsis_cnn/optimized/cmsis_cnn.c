#include "simple_system_common.h"
#include "cnn_weights.h"
#include "fully_connected_opt.h"
#include "ibex_cnn_params.h"
#include "ibex_inputs.h"
#include "conv2d_opt.h"

#define IMG_SZ 32
#define NUM_FIL0 1

#define FILTER1 5
#define FILTER2 5
#define FILTER3 5

#define NUM_FIL1 8
#define NUM_FIL2 8
#define NUM_FIL3 16

#define STRIDE1 1
#define STRIDE2 1
#define STRIDE3 1

#define PAD_TB1 2
#define PAD_LR1 2

#define PAD_TB2 2
#define PAD_LR2 2

#define PAD_TB3 2
#define PAD_LR3 2

#define POOL_STRIDE1 2
#define POOL_SIZE1 2

#define POOL_STRIDE2 2
#define POOL_SIZE2 2

#define POOL_STRIDE3 2
#define POOL_SIZE3 2

#define OUT_DIM 3

#define SAMPLES 1
int outs[SAMPLES][OUT_DIM];

void cmsis_cnn() {

	int dout1 = NUM_FIL1;
	int hout1 = ((IMG_SZ - FILTER1 + 2 * PAD_TB1)/STRIDE1) + 1;
	int wout1 = ((IMG_SZ - FILTER1 + 2 * PAD_LR1)/STRIDE1) + 1;

	int dout2 = dout1;
	int hout2 = hout1/POOL_STRIDE1;
	int wout2 = wout1/POOL_STRIDE1;

	int dout3 = NUM_FIL2;
	int hout3 = ((hout2 - FILTER2+ 2 * PAD_TB2)/STRIDE2)+1;
	int wout3 = ((wout2 - FILTER2+ 2 * PAD_LR2)/STRIDE2)+1;

	int dout4 = dout3;
	int hout4 = hout3/POOL_STRIDE2;
	int wout4 = wout3/POOL_STRIDE2;

	int dout5 = NUM_FIL3;
	int hout5 = ((hout4 - FILTER3+ 2 * PAD_TB3)/STRIDE3)+1;
	int wout5 = ((wout4 - FILTER3+ 2 * PAD_LR3)/STRIDE3)+1;

	int dout6 = dout5;
	int hout6 = hout5/POOL_STRIDE3;
	int wout6 = wout5/POOL_STRIDE3;

	int flatten_dim = dout6 * hout6 * wout6;

	int in[IMG_SZ][IMG_SZ][NUM_FIL0];
	int inp_dim[3] = {IMG_SZ, IMG_SZ, NUM_FIL0};

	int out1[hout1][wout1][dout1];
	int pad_1[4] = {PAD_TB1, PAD_TB1, PAD_LR1, PAD_LR1};
	int outp_dim1[3] = {hout1, wout1, dout1};
	int f_dim1[4] = {NUM_FIL1, FILTER1, FILTER1, NUM_FIL0};

	int out2[hout2][wout2][dout2];
	int outp_dim2[3] = {hout2, wout2, dout2};

	int out3[hout3][wout3][dout3];
	int pad_3[4] = {PAD_TB2, PAD_TB2, PAD_LR2, PAD_LR2};
	int outp_dim3[3] = {hout3, wout3, dout3};
	int f_dim3[4] = {NUM_FIL2, FILTER2, FILTER2, NUM_FIL1};

	int out4[hout4][wout4][dout4];
	int outp_dim4[3] = {hout4, wout4, dout4};

	int out5[hout5][wout5][dout5];
	int pad_5[4] = {PAD_TB3, PAD_TB3, PAD_LR3, PAD_LR3};
	int outp_dim5[3] = {hout5, wout5, dout5};
	int f_dim5[4] = {NUM_FIL3, FILTER3, FILTER3, NUM_FIL2};

	int out6[hout6][wout6][dout6];
	int outp_dim6[3] = {hout6, wout6, dout6};

	int out7[flatten_dim];

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

		conv2_2bits_1ch(inp_dim, f_dim1, outp_dim1, in, F1, B1, out1, STRIDE1, pad_1, SB1, MV1, SV1);
		maxpool2_compressed(outp_dim1, outp_dim2, out1, out2, POOL_SIZE1, POOL_STRIDE1);

		conv2_2bits(outp_dim2, f_dim3, outp_dim3, out2, F2, B2, out3, STRIDE2, pad_3, SB2, MV2, SV2);
		maxpool2_compressed(outp_dim3, outp_dim4, out3, out4, POOL_SIZE2, POOL_STRIDE2);

		conv2_8bits(outp_dim4, f_dim5, outp_dim5, out4, F3, B3, out5, STRIDE3, pad_5, SB3, MV3, SV3);
		maxpool2_compressed(outp_dim5, outp_dim6, out5, out6, POOL_SIZE3, POOL_STRIDE3);

		flatten(outp_dim6, out6, out7);

		mlp_layer_2bits(out7, out, flatten_dim, OUT_DIM, W1, B4, SB4, MV4, SV4);

		pcount_enable(0);

		puts("Output Layer Values:\n");
		for(int i = 0; i < OUT_DIM; i++) {
			puthex((out[i] & 0xFF000000) >> 24);
			puts(" ");
			puthex((out[i] & 0xFF0000) >> 16);
			puts(" ");
			puthex((out[i] & 0xFF00) >> 8);
			puts(" ");
			puthex(out[i] & 0xFF);
			puts("\n");
		}
	}
}

int main(void) {

	pcount_enable(0);

	cmsis_cnn();

	return 0;
}
