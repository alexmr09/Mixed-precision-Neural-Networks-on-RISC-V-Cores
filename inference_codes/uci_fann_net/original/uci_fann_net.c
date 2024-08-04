#include "simple_system_common.h"
#include "fully_connected.h"
#include "ibex_mlp_params.h"
#include "mlp_weights.h"
#include "ibex_inputs.h"

#define IN_DIM 76
#define HIDDEN_DIM1 300
#define HIDDEN_DIM2 200
#define HIDDEN_DIM3 100
#define OUT_DIM 10
#define SAMPLES 1

int outs[SAMPLES][OUT_DIM];

void uci_fann_net() {

	int inp[IN_DIM];
	int y1[HIDDEN_DIM1];
	int y2[HIDDEN_DIM2];
	int y3[HIDDEN_DIM3];
	int out[OUT_DIM];

	for (int iter = 0; iter < SAMPLES; iter ++){
		for(int i = 0; i < IN_DIM; i++) inp[i] = input[iter][i];

		pcount_enable(1);

		mlp_layer(inp, y1, IN_DIM, HIDDEN_DIM1, W1, B1, SB1, MV1, SV1, 1);
		mlp_layer(y1, y2, HIDDEN_DIM1, HIDDEN_DIM2, W2, B2, SB2, MV2, SV2, 1);
		mlp_layer(y2, y3, HIDDEN_DIM2, HIDDEN_DIM3, W3, B3, SB3, MV3, SV3, 1);
		mlp_layer(y3, out, HIDDEN_DIM3, OUT_DIM, W4, B4, SB4, MV4, SV4, 1);

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

	uci_fann_net();

	return 0;
}