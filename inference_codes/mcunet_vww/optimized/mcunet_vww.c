#include "simple_system_common.h"
#include "cnn_weights.h"
#include "fully_connected_opt.h"
#include "ibex_cnn_params.h"
#include "ibex_inputs.h"
#include "conv2d_opt.h"
#include "dws_conv_opt.h"

#define IMG_SZ 80
#define NUM_FIL0 1

#define FILTER1 3
#define FILTER2 3
#define FILTER3 1
#define FILTER4 1
#define FILTER5 3
#define FILTER6 1
#define FILTER7 1
#define FILTER8 3
#define FILTER9 1
#define FILTER10 1
#define FILTER11 3
#define FILTER12 1
#define FILTER13 1
#define FILTER14 7
#define FILTER15 1
#define FILTER16 1
#define FILTER17 3
#define FILTER18 1
#define FILTER19 1
#define FILTER20 5
#define FILTER21 1
#define FILTER22 1
#define FILTER23 7
#define FILTER24 1
#define FILTER25 1
#define FILTER26 7
#define FILTER27 1
#define FILTER28 1
#define FILTER29 3
#define FILTER30 1
#define FILTER31 1
#define FILTER32 3
#define FILTER33 1
#define FILTER34 1
#define FILTER35 5
#define FILTER36 1
#define FILTER37 1
#define FILTER38 3
#define FILTER39 1
#define FILTER40 1
#define FILTER41 3
#define FILTER42 1
#define FILTER43 1
#define FILTER44 7
#define FILTER45 1

#define NUM_FIL1 4
#define NUM_FIL2 4
#define NUM_FIL3 2
#define NUM_FIL4 12
#define NUM_FIL5 12
#define NUM_FIL6 4
#define NUM_FIL7 12
#define NUM_FIL8 12
#define NUM_FIL9 4
#define NUM_FIL10 12
#define NUM_FIL11 12
#define NUM_FIL12 4
#define NUM_FIL13 12
#define NUM_FIL14 12
#define NUM_FIL15 6
#define NUM_FIL16 36
#define NUM_FIL17 36
#define NUM_FIL18 6
#define NUM_FIL19 30
#define NUM_FIL20 30
#define NUM_FIL21 6
#define NUM_FIL22 36
#define NUM_FIL23 36
#define NUM_FIL24 10
#define NUM_FIL25 60
#define NUM_FIL26 60
#define NUM_FIL27 10
#define NUM_FIL28 60
#define NUM_FIL29 60
#define NUM_FIL30 12
#define NUM_FIL31 48
#define NUM_FIL32 48
#define NUM_FIL33 12
#define NUM_FIL34 60
#define NUM_FIL35 60
#define NUM_FIL36 24
#define NUM_FIL37 120
#define NUM_FIL38 120
#define NUM_FIL39 24
#define NUM_FIL40 96
#define NUM_FIL41 96
#define NUM_FIL42 24
#define NUM_FIL43 72
#define NUM_FIL44 72
#define NUM_FIL45 40

#define STRIDE1 2
#define STRIDE2 1
#define STRIDE3 1
#define STRIDE4 1
#define STRIDE5 2
#define STRIDE6 1
#define STRIDE7 1
#define STRIDE8 1
#define STRIDE9 1
#define STRIDE10 1
#define STRIDE11 1
#define STRIDE12 1
#define STRIDE13 1
#define STRIDE14 2
#define STRIDE15 1
#define STRIDE16 1
#define STRIDE17 1
#define STRIDE18 1
#define STRIDE19 1
#define STRIDE20 1
#define STRIDE21 1
#define STRIDE22 1
#define STRIDE23 2
#define STRIDE24 1
#define STRIDE25 1
#define STRIDE26 1
#define STRIDE27 1
#define STRIDE28 1
#define STRIDE29 1
#define STRIDE30 1
#define STRIDE31 1
#define STRIDE32 1
#define STRIDE33 1
#define STRIDE34 1
#define STRIDE35 2
#define STRIDE36 1
#define STRIDE37 1
#define STRIDE38 1
#define STRIDE39 1
#define STRIDE40 1
#define STRIDE41 1
#define STRIDE42 1
#define STRIDE43 1
#define STRIDE44 1
#define STRIDE45 1

#define PAD_TB1 1
#define PAD_LR1 1

#define PAD_TB2 1
#define PAD_LR2 1

#define PAD_TB3 0
#define PAD_LR3 0

#define PAD_TB4 0
#define PAD_LR4 0

#define PAD_TB5 1
#define PAD_LR5 1

#define PAD_TB6 0
#define PAD_LR6 0

#define PAD_TB7 0
#define PAD_LR7 0

#define PAD_TB8 1
#define PAD_LR8 1

#define PAD_TB9 0
#define PAD_LR9 0

#define PAD_TB10 0
#define PAD_LR10 0

#define PAD_TB11 1
#define PAD_LR11 1

#define PAD_TB12 0
#define PAD_LR12 0

#define PAD_TB13 0
#define PAD_LR13 0

#define PAD_TB14 3
#define PAD_LR14 3

#define PAD_TB15 0
#define PAD_LR15 0

#define PAD_TB16 0
#define PAD_LR16 0

#define PAD_TB17 1
#define PAD_LR17 1

#define PAD_TB18 0
#define PAD_LR18 0

#define PAD_TB19 0
#define PAD_LR19 0

#define PAD_TB20 2
#define PAD_LR20 2

#define PAD_TB21 0
#define PAD_LR21 0

#define PAD_TB22 0
#define PAD_LR22 0

#define PAD_TB23 3
#define PAD_LR23 3

#define PAD_TB24 0
#define PAD_LR24 0

#define PAD_TB25 0
#define PAD_LR25 0

#define PAD_TB26 3
#define PAD_LR26 3

#define PAD_TB27 0
#define PAD_LR27 0

#define PAD_TB28 0
#define PAD_LR28 0

#define PAD_TB29 1
#define PAD_LR29 1

#define PAD_TB30 0
#define PAD_LR30 0

#define PAD_TB31 0
#define PAD_LR31 0

#define PAD_TB32 1
#define PAD_LR32 1

#define PAD_TB33 0
#define PAD_LR33 0

#define PAD_TB34 0
#define PAD_LR34 0

#define PAD_TB35 2
#define PAD_LR35 2

#define PAD_TB36 0
#define PAD_LR36 0

#define PAD_TB37 0
#define PAD_LR37 0

#define PAD_TB38 1
#define PAD_LR38 1

#define PAD_TB39 0
#define PAD_LR39 0

#define PAD_TB40 0
#define PAD_LR40 0

#define PAD_TB41 1
#define PAD_LR41 1

#define PAD_TB42 0
#define PAD_LR42 0

#define PAD_TB43 0
#define PAD_LR43 0

#define PAD_TB44 3
#define PAD_LR44 3

#define PAD_TB45 0
#define PAD_LR45 0

#define POOL_STRIDE1 1
#define POOL_SIZE1 3

#define OUT_DIM 1

#define SAMPLES 1
int outs[SAMPLES][OUT_DIM];

void mcunet_vww() {

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

	int dout5 = NUM_FIL5;
	int hout5 = ((hout4 - FILTER5+ 2 * PAD_TB5)/STRIDE5)+1;
	int wout5 = ((wout4 - FILTER5+ 2 * PAD_LR5)/STRIDE5)+1;

	int dout6 = NUM_FIL6;
	int hout6 = ((hout5 - FILTER6+ 2 * PAD_TB6)/STRIDE6)+1;
	int wout6 = ((wout5 - FILTER6+ 2 * PAD_LR6)/STRIDE6)+1;

	int dout7 = NUM_FIL7;
	int hout7 = ((hout6 - FILTER7+ 2 * PAD_TB7)/STRIDE7)+1;
	int wout7 = ((wout6 - FILTER7+ 2 * PAD_LR7)/STRIDE7)+1;

	int dout8 = NUM_FIL8;
	int hout8 = ((hout7 - FILTER8+ 2 * PAD_TB8)/STRIDE8)+1;
	int wout8 = ((wout7 - FILTER8+ 2 * PAD_LR8)/STRIDE8)+1;

	int dout9 = NUM_FIL9;
	int hout9 = ((hout8 - FILTER9+ 2 * PAD_TB9)/STRIDE9)+1;
	int wout9 = ((wout8 - FILTER9+ 2 * PAD_LR9)/STRIDE9)+1;

	int dout10 = dout9;
	int hout10 = hout9;
	int wout10 = wout9;

	int dout11 = NUM_FIL10;
	int hout11 = ((hout10 - FILTER10+ 2 * PAD_TB10)/STRIDE10)+1;
	int wout11 = ((wout10 - FILTER10+ 2 * PAD_LR10)/STRIDE10)+1;

	int dout12 = NUM_FIL11;
	int hout12 = ((hout11 - FILTER11+ 2 * PAD_TB11)/STRIDE11)+1;
	int wout12 = ((wout11 - FILTER11+ 2 * PAD_LR11)/STRIDE11)+1;

	int dout13 = NUM_FIL12;
	int hout13 = ((hout12 - FILTER12+ 2 * PAD_TB12)/STRIDE12)+1;
	int wout13 = ((wout12 - FILTER12+ 2 * PAD_LR12)/STRIDE12)+1;

	int dout14 = dout13;
	int hout14 = hout13;
	int wout14 = wout13;

	int dout15 = NUM_FIL13;
	int hout15 = ((hout14 - FILTER13+ 2 * PAD_TB13)/STRIDE13)+1;
	int wout15 = ((wout14 - FILTER13+ 2 * PAD_LR13)/STRIDE13)+1;

	int dout16 = NUM_FIL14;
	int hout16 = ((hout15 - FILTER14+ 2 * PAD_TB14)/STRIDE14)+1;
	int wout16 = ((wout15 - FILTER14+ 2 * PAD_LR14)/STRIDE14)+1;

	int dout17 = NUM_FIL15;
	int hout17 = ((hout16 - FILTER15+ 2 * PAD_TB15)/STRIDE15)+1;
	int wout17 = ((wout16 - FILTER15+ 2 * PAD_LR15)/STRIDE15)+1;

	int dout18 = NUM_FIL16;
	int hout18 = ((hout17 - FILTER16+ 2 * PAD_TB16)/STRIDE16)+1;
	int wout18 = ((wout17 - FILTER16+ 2 * PAD_LR16)/STRIDE16)+1;

	int dout19 = NUM_FIL17;
	int hout19 = ((hout18 - FILTER17+ 2 * PAD_TB17)/STRIDE17)+1;
	int wout19 = ((wout18 - FILTER17+ 2 * PAD_LR17)/STRIDE17)+1;

	int dout20 = NUM_FIL18;
	int hout20 = ((hout19 - FILTER18+ 2 * PAD_TB18)/STRIDE18)+1;
	int wout20 = ((wout19 - FILTER18+ 2 * PAD_LR18)/STRIDE18)+1;

	int dout21 = dout20;
	int hout21 = hout20;
	int wout21 = wout20;

	int dout22 = NUM_FIL19;
	int hout22 = ((hout21 - FILTER19+ 2 * PAD_TB19)/STRIDE19)+1;
	int wout22 = ((wout21 - FILTER19+ 2 * PAD_LR19)/STRIDE19)+1;

	int dout23 = NUM_FIL20;
	int hout23 = ((hout22 - FILTER20+ 2 * PAD_TB20)/STRIDE20)+1;
	int wout23 = ((wout22 - FILTER20+ 2 * PAD_LR20)/STRIDE20)+1;

	int dout24 = NUM_FIL21;
	int hout24 = ((hout23 - FILTER21+ 2 * PAD_TB21)/STRIDE21)+1;
	int wout24 = ((wout23 - FILTER21+ 2 * PAD_LR21)/STRIDE21)+1;

	int dout25 = dout24;
	int hout25 = hout24;
	int wout25 = wout24;

	int dout26 = NUM_FIL22;
	int hout26 = ((hout25 - FILTER22+ 2 * PAD_TB22)/STRIDE22)+1;
	int wout26 = ((wout25 - FILTER22+ 2 * PAD_LR22)/STRIDE22)+1;

	int dout27 = NUM_FIL23;
	int hout27 = ((hout26 - FILTER23+ 2 * PAD_TB23)/STRIDE23)+1;
	int wout27 = ((wout26 - FILTER23+ 2 * PAD_LR23)/STRIDE23)+1;

	int dout28 = NUM_FIL24;
	int hout28 = ((hout27 - FILTER24+ 2 * PAD_TB24)/STRIDE24)+1;
	int wout28 = ((wout27 - FILTER24+ 2 * PAD_LR24)/STRIDE24)+1;

	int dout29 = NUM_FIL25;
	int hout29 = ((hout28 - FILTER25+ 2 * PAD_TB25)/STRIDE25)+1;
	int wout29 = ((wout28 - FILTER25+ 2 * PAD_LR25)/STRIDE25)+1;

	int dout30 = NUM_FIL26;
	int hout30 = ((hout29 - FILTER26+ 2 * PAD_TB26)/STRIDE26)+1;
	int wout30 = ((wout29 - FILTER26+ 2 * PAD_LR26)/STRIDE26)+1;

	int dout31 = NUM_FIL27;
	int hout31 = ((hout30 - FILTER27+ 2 * PAD_TB27)/STRIDE27)+1;
	int wout31 = ((wout30 - FILTER27+ 2 * PAD_LR27)/STRIDE27)+1;

	int dout32 = dout31;
	int hout32 = hout31;
	int wout32 = wout31;

	int dout33 = NUM_FIL28;
	int hout33 = ((hout32 - FILTER28+ 2 * PAD_TB28)/STRIDE28)+1;
	int wout33 = ((wout32 - FILTER28+ 2 * PAD_LR28)/STRIDE28)+1;

	int dout34 = NUM_FIL29;
	int hout34 = ((hout33 - FILTER29+ 2 * PAD_TB29)/STRIDE29)+1;
	int wout34 = ((wout33 - FILTER29+ 2 * PAD_LR29)/STRIDE29)+1;

	int dout35 = NUM_FIL30;
	int hout35 = ((hout34 - FILTER30+ 2 * PAD_TB30)/STRIDE30)+1;
	int wout35 = ((wout34 - FILTER30+ 2 * PAD_LR30)/STRIDE30)+1;

	int dout36 = NUM_FIL31;
	int hout36 = ((hout35 - FILTER31+ 2 * PAD_TB31)/STRIDE31)+1;
	int wout36 = ((wout35 - FILTER31+ 2 * PAD_LR31)/STRIDE31)+1;

	int dout37 = NUM_FIL32;
	int hout37 = ((hout36 - FILTER32+ 2 * PAD_TB32)/STRIDE32)+1;
	int wout37 = ((wout36 - FILTER32+ 2 * PAD_LR32)/STRIDE32)+1;

	int dout38 = NUM_FIL33;
	int hout38 = ((hout37 - FILTER33+ 2 * PAD_TB33)/STRIDE33)+1;
	int wout38 = ((wout37 - FILTER33+ 2 * PAD_LR33)/STRIDE33)+1;

	int dout39 = dout38;
	int hout39 = hout38;
	int wout39 = wout38;

	int dout40 = NUM_FIL34;
	int hout40 = ((hout39 - FILTER34+ 2 * PAD_TB34)/STRIDE34)+1;
	int wout40 = ((wout39 - FILTER34+ 2 * PAD_LR34)/STRIDE34)+1;

	int dout41 = NUM_FIL35;
	int hout41 = ((hout40 - FILTER35+ 2 * PAD_TB35)/STRIDE35)+1;
	int wout41 = ((wout40 - FILTER35+ 2 * PAD_LR35)/STRIDE35)+1;

	int dout42 = NUM_FIL36;
	int hout42 = ((hout41 - FILTER36+ 2 * PAD_TB36)/STRIDE36)+1;
	int wout42 = ((wout41 - FILTER36+ 2 * PAD_LR36)/STRIDE36)+1;

	int dout43 = NUM_FIL37;
	int hout43 = ((hout42 - FILTER37+ 2 * PAD_TB37)/STRIDE37)+1;
	int wout43 = ((wout42 - FILTER37+ 2 * PAD_LR37)/STRIDE37)+1;

	int dout44 = NUM_FIL38;
	int hout44 = ((hout43 - FILTER38+ 2 * PAD_TB38)/STRIDE38)+1;
	int wout44 = ((wout43 - FILTER38+ 2 * PAD_LR38)/STRIDE38)+1;

	int dout45 = NUM_FIL39;
	int hout45 = ((hout44 - FILTER39+ 2 * PAD_TB39)/STRIDE39)+1;
	int wout45 = ((wout44 - FILTER39+ 2 * PAD_LR39)/STRIDE39)+1;

	int dout46 = dout45;
	int hout46 = hout45;
	int wout46 = wout45;

	int dout47 = NUM_FIL40;
	int hout47 = ((hout46 - FILTER40+ 2 * PAD_TB40)/STRIDE40)+1;
	int wout47 = ((wout46 - FILTER40+ 2 * PAD_LR40)/STRIDE40)+1;

	int dout48 = NUM_FIL41;
	int hout48 = ((hout47 - FILTER41+ 2 * PAD_TB41)/STRIDE41)+1;
	int wout48 = ((wout47 - FILTER41+ 2 * PAD_LR41)/STRIDE41)+1;

	int dout49 = NUM_FIL42;
	int hout49 = ((hout48 - FILTER42+ 2 * PAD_TB42)/STRIDE42)+1;
	int wout49 = ((wout48 - FILTER42+ 2 * PAD_LR42)/STRIDE42)+1;

	int dout50 = dout49;
	int hout50 = hout49;
	int wout50 = wout49;

	int dout51 = NUM_FIL43;
	int hout51 = ((hout50 - FILTER43+ 2 * PAD_TB43)/STRIDE43)+1;
	int wout51 = ((wout50 - FILTER43+ 2 * PAD_LR43)/STRIDE43)+1;

	int dout52 = NUM_FIL44;
	int hout52 = ((hout51 - FILTER44+ 2 * PAD_TB44)/STRIDE44)+1;
	int wout52 = ((wout51 - FILTER44+ 2 * PAD_LR44)/STRIDE44)+1;

	int dout53 = NUM_FIL45;
	int hout53 = ((hout52 - FILTER45+ 2 * PAD_TB45)/STRIDE45)+1;
	int wout53 = ((wout52 - FILTER45+ 2 * PAD_LR45)/STRIDE45)+1;

	int dout54 = dout53;
	int hout54 = ((hout53 - POOL_SIZE1)/POOL_STRIDE1) + 1;
	int wout54 = ((wout53 - POOL_SIZE1)/POOL_STRIDE1) + 1;

	int flatten_dim = dout54 * hout54 * wout54;

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
	int pad_5[4] = {PAD_TB5, PAD_TB5, PAD_LR5, PAD_LR5};
	int outp_dim5[3] = {hout5, wout5, dout5};
	int f_dim5[4] = {NUM_FIL5, FILTER5, FILTER5, NUM_FIL4};

	int out6[hout6][wout6][dout6];
	int pad_6[4] = {PAD_TB6, PAD_TB6, PAD_LR6, PAD_LR6};
	int outp_dim6[3] = {hout6, wout6, dout6};
	int f_dim6[4] = {NUM_FIL6, FILTER6, FILTER6, NUM_FIL5};

	int out7[hout7][wout7][dout7];
	int pad_7[4] = {PAD_TB7, PAD_TB7, PAD_LR7, PAD_LR7};
	int outp_dim7[3] = {hout7, wout7, dout7};
	int f_dim7[4] = {NUM_FIL7, FILTER7, FILTER7, NUM_FIL6};

	int out8[hout8][wout8][dout8];
	int pad_8[4] = {PAD_TB8, PAD_TB8, PAD_LR8, PAD_LR8};
	int outp_dim8[3] = {hout8, wout8, dout8};
	int f_dim8[4] = {NUM_FIL8, FILTER8, FILTER8, NUM_FIL7};

	int out9[hout9][wout9][dout9];
	int pad_9[4] = {PAD_TB9, PAD_TB9, PAD_LR9, PAD_LR9};
	int outp_dim9[3] = {hout9, wout9, dout9};
	int f_dim9[4] = {NUM_FIL9, FILTER9, FILTER9, NUM_FIL8};

	int out10[hout10][wout10][dout10];
	int outp_dim10[3] = {hout10, wout10, dout10};

	int out11[hout11][wout11][dout11];
	int pad_11[4] = {PAD_TB10, PAD_TB10, PAD_LR10, PAD_LR10};
	int outp_dim11[3] = {hout11, wout11, dout11};
	int f_dim11[4] = {NUM_FIL10, FILTER10, FILTER10, NUM_FIL9};

	int out12[hout12][wout12][dout12];
	int pad_12[4] = {PAD_TB11, PAD_TB11, PAD_LR11, PAD_LR11};
	int outp_dim12[3] = {hout12, wout12, dout12};
	int f_dim12[4] = {NUM_FIL11, FILTER11, FILTER11, NUM_FIL10};

	int out13[hout13][wout13][dout13];
	int pad_13[4] = {PAD_TB12, PAD_TB12, PAD_LR12, PAD_LR12};
	int outp_dim13[3] = {hout13, wout13, dout13};
	int f_dim13[4] = {NUM_FIL12, FILTER12, FILTER12, NUM_FIL11};

	int out14[hout14][wout14][dout14];
	int outp_dim14[3] = {hout14, wout14, dout14};

	int out15[hout15][wout15][dout15];
	int pad_15[4] = {PAD_TB13, PAD_TB13, PAD_LR13, PAD_LR13};
	int outp_dim15[3] = {hout15, wout15, dout15};
	int f_dim15[4] = {NUM_FIL13, FILTER13, FILTER13, NUM_FIL12};

	int out16[hout16][wout16][dout16];
	int pad_16[4] = {PAD_TB14, PAD_TB14, PAD_LR14, PAD_LR14};
	int outp_dim16[3] = {hout16, wout16, dout16};
	int f_dim16[4] = {NUM_FIL14, FILTER14, FILTER14, NUM_FIL13};

	int out17[hout17][wout17][dout17];
	int pad_17[4] = {PAD_TB15, PAD_TB15, PAD_LR15, PAD_LR15};
	int outp_dim17[3] = {hout17, wout17, dout17};
	int f_dim17[4] = {NUM_FIL15, FILTER15, FILTER15, NUM_FIL14};

	int out18[hout18][wout18][dout18];
	int pad_18[4] = {PAD_TB16, PAD_TB16, PAD_LR16, PAD_LR16};
	int outp_dim18[3] = {hout18, wout18, dout18};
	int f_dim18[4] = {NUM_FIL16, FILTER16, FILTER16, NUM_FIL15};

	int out19[hout19][wout19][dout19];
	int pad_19[4] = {PAD_TB17, PAD_TB17, PAD_LR17, PAD_LR17};
	int outp_dim19[3] = {hout19, wout19, dout19};
	int f_dim19[4] = {NUM_FIL17, FILTER17, FILTER17, NUM_FIL16};

	int out20[hout20][wout20][dout20];
	int pad_20[4] = {PAD_TB18, PAD_TB18, PAD_LR18, PAD_LR18};
	int outp_dim20[3] = {hout20, wout20, dout20};
	int f_dim20[4] = {NUM_FIL18, FILTER18, FILTER18, NUM_FIL17};

	int out21[hout21][wout21][dout21];
	int outp_dim21[3] = {hout21, wout21, dout21};

	int out22[hout22][wout22][dout22];
	int pad_22[4] = {PAD_TB19, PAD_TB19, PAD_LR19, PAD_LR19};
	int outp_dim22[3] = {hout22, wout22, dout22};
	int f_dim22[4] = {NUM_FIL19, FILTER19, FILTER19, NUM_FIL18};

	int out23[hout23][wout23][dout23];
	int pad_23[4] = {PAD_TB20, PAD_TB20, PAD_LR20, PAD_LR20};
	int outp_dim23[3] = {hout23, wout23, dout23};
	int f_dim23[4] = {NUM_FIL20, FILTER20, FILTER20, NUM_FIL19};

	int out24[hout24][wout24][dout24];
	int pad_24[4] = {PAD_TB21, PAD_TB21, PAD_LR21, PAD_LR21};
	int outp_dim24[3] = {hout24, wout24, dout24};
	int f_dim24[4] = {NUM_FIL21, FILTER21, FILTER21, NUM_FIL20};

	int out25[hout25][wout25][dout25];
	int outp_dim25[3] = {hout25, wout25, dout25};

	int out26[hout26][wout26][dout26];
	int pad_26[4] = {PAD_TB22, PAD_TB22, PAD_LR22, PAD_LR22};
	int outp_dim26[3] = {hout26, wout26, dout26};
	int f_dim26[4] = {NUM_FIL22, FILTER22, FILTER22, NUM_FIL21};

	int out27[hout27][wout27][dout27];
	int pad_27[4] = {PAD_TB23, PAD_TB23, PAD_LR23, PAD_LR23};
	int outp_dim27[3] = {hout27, wout27, dout27};
	int f_dim27[4] = {NUM_FIL23, FILTER23, FILTER23, NUM_FIL22};

	int out28[hout28][wout28][dout28];
	int pad_28[4] = {PAD_TB24, PAD_TB24, PAD_LR24, PAD_LR24};
	int outp_dim28[3] = {hout28, wout28, dout28};
	int f_dim28[4] = {NUM_FIL24, FILTER24, FILTER24, NUM_FIL23};

	int out29[hout29][wout29][dout29];
	int pad_29[4] = {PAD_TB25, PAD_TB25, PAD_LR25, PAD_LR25};
	int outp_dim29[3] = {hout29, wout29, dout29};
	int f_dim29[4] = {NUM_FIL25, FILTER25, FILTER25, NUM_FIL24};

	int out30[hout30][wout30][dout30];
	int pad_30[4] = {PAD_TB26, PAD_TB26, PAD_LR26, PAD_LR26};
	int outp_dim30[3] = {hout30, wout30, dout30};
	int f_dim30[4] = {NUM_FIL26, FILTER26, FILTER26, NUM_FIL25};

	int out31[hout31][wout31][dout31];
	int pad_31[4] = {PAD_TB27, PAD_TB27, PAD_LR27, PAD_LR27};
	int outp_dim31[3] = {hout31, wout31, dout31};
	int f_dim31[4] = {NUM_FIL27, FILTER27, FILTER27, NUM_FIL26};

	int out32[hout32][wout32][dout32];
	int outp_dim32[3] = {hout32, wout32, dout32};

	int out33[hout33][wout33][dout33];
	int pad_33[4] = {PAD_TB28, PAD_TB28, PAD_LR28, PAD_LR28};
	int outp_dim33[3] = {hout33, wout33, dout33};
	int f_dim33[4] = {NUM_FIL28, FILTER28, FILTER28, NUM_FIL27};

	int out34[hout34][wout34][dout34];
	int pad_34[4] = {PAD_TB29, PAD_TB29, PAD_LR29, PAD_LR29};
	int outp_dim34[3] = {hout34, wout34, dout34};
	int f_dim34[4] = {NUM_FIL29, FILTER29, FILTER29, NUM_FIL28};

	int out35[hout35][wout35][dout35];
	int pad_35[4] = {PAD_TB30, PAD_TB30, PAD_LR30, PAD_LR30};
	int outp_dim35[3] = {hout35, wout35, dout35};
	int f_dim35[4] = {NUM_FIL30, FILTER30, FILTER30, NUM_FIL29};

	int out36[hout36][wout36][dout36];
	int pad_36[4] = {PAD_TB31, PAD_TB31, PAD_LR31, PAD_LR31};
	int outp_dim36[3] = {hout36, wout36, dout36};
	int f_dim36[4] = {NUM_FIL31, FILTER31, FILTER31, NUM_FIL30};

	int out37[hout37][wout37][dout37];
	int pad_37[4] = {PAD_TB32, PAD_TB32, PAD_LR32, PAD_LR32};
	int outp_dim37[3] = {hout37, wout37, dout37};
	int f_dim37[4] = {NUM_FIL32, FILTER32, FILTER32, NUM_FIL31};

	int out38[hout38][wout38][dout38];
	int pad_38[4] = {PAD_TB33, PAD_TB33, PAD_LR33, PAD_LR33};
	int outp_dim38[3] = {hout38, wout38, dout38};
	int f_dim38[4] = {NUM_FIL33, FILTER33, FILTER33, NUM_FIL32};

	int out39[hout39][wout39][dout39];
	int outp_dim39[3] = {hout39, wout39, dout39};

	int out40[hout40][wout40][dout40];
	int pad_40[4] = {PAD_TB34, PAD_TB34, PAD_LR34, PAD_LR34};
	int outp_dim40[3] = {hout40, wout40, dout40};
	int f_dim40[4] = {NUM_FIL34, FILTER34, FILTER34, NUM_FIL33};

	int out41[hout41][wout41][dout41];
	int pad_41[4] = {PAD_TB35, PAD_TB35, PAD_LR35, PAD_LR35};
	int outp_dim41[3] = {hout41, wout41, dout41};
	int f_dim41[4] = {NUM_FIL35, FILTER35, FILTER35, NUM_FIL34};

	int out42[hout42][wout42][dout42];
	int pad_42[4] = {PAD_TB36, PAD_TB36, PAD_LR36, PAD_LR36};
	int outp_dim42[3] = {hout42, wout42, dout42};
	int f_dim42[4] = {NUM_FIL36, FILTER36, FILTER36, NUM_FIL35};

	int out43[hout43][wout43][dout43];
	int pad_43[4] = {PAD_TB37, PAD_TB37, PAD_LR37, PAD_LR37};
	int outp_dim43[3] = {hout43, wout43, dout43};
	int f_dim43[4] = {NUM_FIL37, FILTER37, FILTER37, NUM_FIL36};

	int out44[hout44][wout44][dout44];
	int pad_44[4] = {PAD_TB38, PAD_TB38, PAD_LR38, PAD_LR38};
	int outp_dim44[3] = {hout44, wout44, dout44};
	int f_dim44[4] = {NUM_FIL38, FILTER38, FILTER38, NUM_FIL37};

	int out45[hout45][wout45][dout45];
	int pad_45[4] = {PAD_TB39, PAD_TB39, PAD_LR39, PAD_LR39};
	int outp_dim45[3] = {hout45, wout45, dout45};
	int f_dim45[4] = {NUM_FIL39, FILTER39, FILTER39, NUM_FIL38};

	int out46[hout46][wout46][dout46];
	int outp_dim46[3] = {hout46, wout46, dout46};

	int out47[hout47][wout47][dout47];
	int pad_47[4] = {PAD_TB40, PAD_TB40, PAD_LR40, PAD_LR40};
	int outp_dim47[3] = {hout47, wout47, dout47};
	int f_dim47[4] = {NUM_FIL40, FILTER40, FILTER40, NUM_FIL39};

	int out48[hout48][wout48][dout48];
	int pad_48[4] = {PAD_TB41, PAD_TB41, PAD_LR41, PAD_LR41};
	int outp_dim48[3] = {hout48, wout48, dout48};
	int f_dim48[4] = {NUM_FIL41, FILTER41, FILTER41, NUM_FIL40};

	int out49[hout49][wout49][dout49];
	int pad_49[4] = {PAD_TB42, PAD_TB42, PAD_LR42, PAD_LR42};
	int outp_dim49[3] = {hout49, wout49, dout49};
	int f_dim49[4] = {NUM_FIL42, FILTER42, FILTER42, NUM_FIL41};

	int out50[hout50][wout50][dout50];
	int outp_dim50[3] = {hout50, wout50, dout50};

	int out51[hout51][wout51][dout51];
	int pad_51[4] = {PAD_TB43, PAD_TB43, PAD_LR43, PAD_LR43};
	int outp_dim51[3] = {hout51, wout51, dout51};
	int f_dim51[4] = {NUM_FIL43, FILTER43, FILTER43, NUM_FIL42};

	int out52[hout52][wout52][dout52];
	int pad_52[4] = {PAD_TB44, PAD_TB44, PAD_LR44, PAD_LR44};
	int outp_dim52[3] = {hout52, wout52, dout52};
	int f_dim52[4] = {NUM_FIL44, FILTER44, FILTER44, NUM_FIL43};

	int out53[hout53][wout53][dout53];
	int pad_53[4] = {PAD_TB45, PAD_TB45, PAD_LR45, PAD_LR45};
	int outp_dim53[3] = {hout53, wout53, dout53};
	int f_dim53[4] = {NUM_FIL45, FILTER45, FILTER45, NUM_FIL44};

	int out54[hout54][wout54][dout54];
	int outp_dim54[3] = {hout54, wout54, dout54};

	int out55[flatten_dim];

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
		
		dw_conv_opt(outp_dim1, f_dim2, outp_dim2, out1, F2, B2, out2, STRIDE2, pad_2, SB2, MV2, SV2);
		pw_conv_2bits(outp_dim2, f_dim3, outp_dim3, out2, F3, B3, out3, STRIDE3, pad_3, SB3, MV3, SV3);
		
		pw_conv_4bits(outp_dim3, f_dim4, outp_dim4, out3, F4, B4, out4, STRIDE4, pad_4, SB4, MV4, SV4);
		dw_conv_opt(outp_dim4, f_dim5, outp_dim5, out4, F5, B5, out5, STRIDE5, pad_5, SB5, MV5, SV5);
		pw_conv_4bits(outp_dim5, f_dim6, outp_dim6, out5, F6, B6, out6, STRIDE6, pad_6, SB6, MV6, SV6);
		
		pw_conv_2bits(outp_dim6, f_dim7, outp_dim7, out6, F7, B7, out7, STRIDE7, pad_7, SB7, MV7, SV7);
		dw_conv_opt(outp_dim7, f_dim8, outp_dim8, out7, F8, B8, out8, STRIDE8, pad_8, SB8, MV8, SV8);
		pw_conv_2bits(outp_dim8, f_dim9, outp_dim9, out8, F9, B9, out9, STRIDE9, pad_9, SB9, MV9, SV9);
		
		shortcut(outp_dim10, out9, out6, out10);

		pw_conv_2bits(outp_dim10, f_dim11, outp_dim11, out10, F10, B10, out11, STRIDE10, pad_11, SB10, MV10, SV10);
		dw_conv_opt(outp_dim11, f_dim12, outp_dim12, out11, F11, B11, out12, STRIDE11, pad_12, SB11, MV11, SV11);
		pw_conv_2bits(outp_dim12, f_dim13, outp_dim13, out12, F12, B12, out13, STRIDE12, pad_13, SB12, MV12, SV12);
		shortcut(outp_dim14, out13, out10, out14);

		pw_conv_2bits(outp_dim14, f_dim15, outp_dim15, out14, F13, B13, out15, STRIDE13, pad_15, SB13, MV13, SV13);
		dw_conv_opt(outp_dim15, f_dim16, outp_dim16, out15, F14, B14, out16, STRIDE14, pad_16, SB14, MV14, SV14);
		pw_conv_2bits(outp_dim16, f_dim17, outp_dim17, out16, F15, B15, out17, STRIDE15, pad_17, SB15, MV15, SV15);
		
		pw_conv_4bits(outp_dim17, f_dim18, outp_dim18, out17, F16, B16, out18, STRIDE16, pad_18, SB16, MV16, SV16);
		dw_conv_opt(outp_dim18, f_dim19, outp_dim19, out18, F17, B17, out19, STRIDE17, pad_19, SB17, MV17, SV17);
		pw_conv_4bits(outp_dim19, f_dim20, outp_dim20, out19, F18, B18, out20, STRIDE18, pad_20, SB18, MV18, SV18);
		
		shortcut(outp_dim21, out20, out17, out21);

		pw_conv_4bits(outp_dim21, f_dim22, outp_dim22, out21, F19, B19, out22, STRIDE19, pad_22, SB19, MV19, SV19);
		dw_conv_opt(outp_dim22, f_dim23, outp_dim23, out22, F20, B20, out23, STRIDE20, pad_23, SB20, MV20, SV20);
		pw_conv_4bits(outp_dim23, f_dim24, outp_dim24, out23, F21, B21, out24, STRIDE21, pad_24, SB21, MV21, SV21);
		
		shortcut(outp_dim25, out24, out21, out25);

		pw_conv_4bits(outp_dim25, f_dim26, outp_dim26, out25, F22, B22, out26, STRIDE22, pad_26, SB22, MV22, SV22);
		dw_conv_opt(outp_dim26, f_dim27, outp_dim27, out26, F23, B23, out27, STRIDE23, pad_27, SB23, MV23, SV23);
		pw_conv_4bits(outp_dim27, f_dim28, outp_dim28, out27, F24, B24, out28, STRIDE24, pad_28, SB24, MV24, SV24);
		
		pw_conv_4bits(outp_dim28, f_dim29, outp_dim29, out28, F25, B25, out29, STRIDE25, pad_29, SB25, MV25, SV25);
		dw_conv_opt(outp_dim29, f_dim30, outp_dim30, out29, F26, B26, out30, STRIDE26, pad_30, SB26, MV26, SV26);
		pw_conv_4bits(outp_dim30, f_dim31, outp_dim31, out30, F27, B27, out31, STRIDE27, pad_31, SB27, MV27, SV27);
		
		shortcut(outp_dim32, out31, out28, out32);

		pw_conv_2bits(outp_dim32, f_dim33, outp_dim33, out32, F28, B28, out33, STRIDE28, pad_33, SB28, MV28, SV28);
		dw_conv_opt(outp_dim33, f_dim34, outp_dim34, out33, F29, B29, out34, STRIDE29, pad_34, SB29, MV29, SV29);
		pw_conv_2bits(outp_dim34, f_dim35, outp_dim35, out34, F30, B30, out35, STRIDE30, pad_35, SB30, MV30, SV30);
		
		pw_conv_2bits(outp_dim35, f_dim36, outp_dim36, out35, F31, B31, out36, STRIDE31, pad_36, SB31, MV31, SV31);
		dw_conv_opt(outp_dim36, f_dim37, outp_dim37, out36, F32, B32, out37, STRIDE32, pad_37, SB32, MV32, SV32);
		pw_conv_2bits(outp_dim37, f_dim38, outp_dim38, out37, F33, B33, out38, STRIDE33, pad_38, SB33, MV33, SV33);
		
		shortcut(outp_dim39, out38, out35, out39);

		pw_conv_2bits(outp_dim39, f_dim40, outp_dim40, out39, F34, B34, out40, STRIDE34, pad_40, SB34, MV34, SV34);
		dw_conv_opt(outp_dim40, f_dim41, outp_dim41, out40, F35, B35, out41, STRIDE35, pad_41, SB35, MV35, SV35);
		pw_conv_2bits(outp_dim41, f_dim42, outp_dim42, out41, F36, B36, out42, STRIDE36, pad_42, SB36, MV36, SV36);
		
		pw_conv_8bits(outp_dim42, f_dim43, outp_dim43, out42, F37, B37, out43, STRIDE37, pad_43, SB37, MV37, SV37);
		dw_conv_opt(outp_dim43, f_dim44, outp_dim44, out43, F38, B38, out44, STRIDE38, pad_44, SB38, MV38, SV38);
		pw_conv_8bits(outp_dim44, f_dim45, outp_dim45, out44, F39, B39, out45, STRIDE39, pad_45, SB39, MV39, SV39);
		
		shortcut(outp_dim46, out45, out42, out46);

		pw_conv_2bits(outp_dim46, f_dim47, outp_dim47, out46, F40, B40, out47, STRIDE40, pad_47, SB40, MV40, SV40);
		dw_conv_opt(outp_dim47, f_dim48, outp_dim48, out47, F41, B41, out48, STRIDE41, pad_48, SB41, MV41, SV41);
		pw_conv_2bits(outp_dim48, f_dim49, outp_dim49, out48, F42, B42, out49, STRIDE42, pad_49, SB42, MV42, SV42);
		
		shortcut(outp_dim50, out49, out46, out50);

		pw_conv_2bits(outp_dim50, f_dim51, outp_dim51, out50, F43, B43, out51, STRIDE43, pad_51, SB43, MV43, SV43);
		dw_conv_opt(outp_dim51, f_dim52, outp_dim52, out51, F44, B44, out52, STRIDE44, pad_52, SB44, MV44, SV44);
		pw_conv_2bits(outp_dim52, f_dim53, outp_dim53, out52, F45, B45, out53, STRIDE45, pad_53, SB45, MV45, SV45);
		
		avgpool2_compressed_signed(outp_dim53, outp_dim54, out53, out54, POOL_SIZE1, POOL_STRIDE1);

		flatten(outp_dim54, out54, out55);

		mlp_layer_8bits(out55, out, flatten_dim, OUT_DIM, W1, B46, SB46, MV46, SV46);

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

	mcunet_vww();

	return 0;
}
