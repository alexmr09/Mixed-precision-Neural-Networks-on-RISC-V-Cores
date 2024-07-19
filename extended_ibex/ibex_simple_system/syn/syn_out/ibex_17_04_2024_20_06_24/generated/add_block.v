module add_block (
	p_a,
	p_b,
	mode_3,
	sums
);
	input wire [63:0] p_a;
	input wire [63:0] p_b;
	input wire mode_3;
	output wire [63:0] sums;
	wire [31:0] temp_sum [0:1];
	wire [31:0] mode_3_temp_sum;
	wire [31:0] temp_res [0:1];
	assign mode_3_temp_sum = temp_sum[0] + temp_sum[1];
	assign temp_res[0] = {{20 {mode_3_temp_sum[27]}}, mode_3_temp_sum[27:16]};
	assign temp_res[1] = {{20 {mode_3_temp_sum[11]}}, mode_3_temp_sum[11:0]};
	assign sums[32+:32] = (mode_3 ? temp_res[0] : temp_sum[0]);
	assign sums[0+:32] = (mode_3 ? temp_res[1] : temp_sum[1]);
	adder_32_bits A0(
		.operant_a(p_a[32+:32]),
		.operant_b(p_b[32+:32]),
		.mode_3(mode_3),
		.out_res(temp_sum[0])
	);
	adder_32_bits A1(
		.operant_a(p_a[0+:32]),
		.operant_b(p_b[0+:32]),
		.mode_3(mode_3),
		.out_res(temp_sum[1])
	);
endmodule
