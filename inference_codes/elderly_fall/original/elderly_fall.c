#include "simple_system_common.h"
#include "fully_connected.h"
#include "ibex_mlp_params.h"
#include "mlp_weights.h"
#include "ibex_inputs.h"

#define IN_DIM 117
#define HIDDEN_DIM1 20
#define OUT_DIM 2
#define SAMPLES 1

int outs[SAMPLES][OUT_DIM];

void elderly_fall() {

	int inp[IN_DIM];
	int y1[HIDDEN_DIM1];
	int out[OUT_DIM];

	for (int iter = 0; iter < SAMPLES; iter ++){
		for(int i = 0; i < IN_DIM; i++) inp[i] = input[iter][i];

		pcount_enable(1);

		mlp_layer(inp, y1, IN_DIM, HIDDEN_DIM1, W1, B1, SB1, MV1, SV1);
		mlp_layer(y1, out, HIDDEN_DIM1, OUT_DIM, W2, B2, SB2, MV2, SV2);

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

	elderly_fall();

	return 0;
}
