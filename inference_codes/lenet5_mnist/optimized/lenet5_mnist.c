#include "simple_system_common.h"
#include "cnn_weights.h"
#include "fully_connected_opt.h"
#include "ibex_cnn_params.h"
#include "ibex_inputs.h"
#include "conv2d_opt.h"

#define IMG_SZ 28
#define NUM_FIL0 1

#define FILTER1 5
#define FILTER2 5

#define NUM_FIL1 2
#define NUM_FIL2 4

#define STRIDE1 1
#define STRIDE2 1

#define PAD_TB1 2
#define PAD_LR1 2

#define PAD_TB2 0
#define PAD_LR2 0

#define POOL_STRIDE1 2
#define POOL_SIZE1 2

#define POOL_STRIDE2 2
#define POOL_SIZE2 2

#define DENSE_DIM1 30
#define DENSE_DIM2 21
#define OUT_DIM 3

#define SAMPLES 1
int outs[SAMPLES][OUT_DIM];

void lenet5_mnist() {

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

	int flatten_dim = dout4 * hout4 * wout4;

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

	int out5[flatten_dim];
	int out6[DENSE_DIM1];
	int out7[DENSE_DIM2];

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

		conv2_4bits_1ch(inp_dim, f_dim1, outp_dim1, in, F1, B1, out1, STRIDE1, pad_1, SB1, MV1, SV1);
		avgpool2_compressed_unsigned(outp_dim1, outp_dim2, out1, out2, POOL_SIZE1, POOL_STRIDE1);

		conv2_8bits(outp_dim2, f_dim3, outp_dim3, out2, F2, B2, out3, STRIDE2, pad_3, SB2, MV2, SV2);
		avgpool2_compressed_unsigned(outp_dim3, outp_dim4, out3, out4, POOL_SIZE2, POOL_STRIDE2);

		flatten(outp_dim4, out4, out5);

		mlp_layer_2bits(out5, out6, flatten_dim, DENSE_DIM1, W1, B3, SB3, MV3, SV3);

		mlp_layer_8bits(out6, out7, DENSE_DIM1, DENSE_DIM2, W2, B4, SB4, MV4, SV4);

		mlp_layer_8bits(out7, out, DENSE_DIM2, OUT_DIM, W3, B5, SB5, MV5, SV5);

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

	lenet5_mnist();

	return 0;
}
