#include "simple_system_common.h"
#include "fully_connected_opt.h"
#include "ibex_mlp_params.h"
#include "mlp_weights.h"
#include "ibex_inputs.h"

#define IN_DIM 120
#define HIDDEN_DIM1 20
#define OUT_DIM 4
#define SAMPLES 1

int outs[SAMPLES][OUT_DIM >> 2];

void elderly_fall() {

	int inp[IN_DIM >> 2];
	int y1[HIDDEN_DIM1 >> 2];
	int out[OUT_DIM >> 2];

	for (int iter = 0; iter < SAMPLES; iter ++){
		for(int i = 0; i < IN_DIM >> 2; i++) inp[i] = input[iter][i];

		pcount_enable(1);

		mlp_layer_2bits(inp, y1, IN_DIM >> 2, HIDDEN_DIM1 >> 2, W1, B1, SB1, MV1, SV1);
		mlp_layer_2bits(y1, out, HIDDEN_DIM1 >> 2, OUT_DIM >> 2, W2, B2, SB2, MV2, SV2);

		pcount_enable(0);

		puts("Output Layer Values:\n");
		for(int i = 0; i < OUT_DIM >> 2; i++) {
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

	elderly_fall();

	return 0;
}