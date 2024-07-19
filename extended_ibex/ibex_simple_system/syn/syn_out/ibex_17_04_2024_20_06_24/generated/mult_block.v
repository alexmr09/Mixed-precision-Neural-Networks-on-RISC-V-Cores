module mult_block (
	weight_vals,
	activations,
	normal_mul,
	partial_prods
);
	input wire [67:0] weight_vals;
	input wire [67:0] activations;
	input wire normal_mul;
	output wire [135:0] partial_prods;
	wire [33:0] temp_prods [0:3];
	assign partial_prods[102+:34] = temp_prods[0];
	assign partial_prods[68+:34] = temp_prods[1];
	assign partial_prods[34+:34] = temp_prods[2];
	assign partial_prods[0+:34] = temp_prods[3];
	mult_16_bits M0(
		.operant_a(weight_vals[51+:17]),
		.operant_b(activations[51+:17]),
		.normal_mul(normal_mul),
		.out_res(temp_prods[0])
	);
	mult_16_bits M1(
		.operant_a(weight_vals[34+:17]),
		.operant_b(activations[34+:17]),
		.normal_mul(normal_mul),
		.out_res(temp_prods[1])
	);
	mult_16_bits M2(
		.operant_a(weight_vals[17+:17]),
		.operant_b(activations[17+:17]),
		.normal_mul(normal_mul),
		.out_res(temp_prods[2])
	);
	mult_16_bits M3(
		.operant_a(weight_vals[0+:17]),
		.operant_b(activations[0+:17]),
		.normal_mul(normal_mul),
		.out_res(temp_prods[3])
	);
endmodule
