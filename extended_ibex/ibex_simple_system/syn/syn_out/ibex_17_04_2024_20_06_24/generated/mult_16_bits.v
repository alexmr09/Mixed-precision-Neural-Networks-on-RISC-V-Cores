module mult_16_bits (
	operant_a,
	operant_b,
	normal_mul,
	out_res
);
	input wire [16:0] operant_a;
	input wire [16:0] operant_b;
	input wire normal_mul;
	output wire [33:0] out_res;
	wire [33:0] temp_outcome;
	wire non_zero;
	wire sign;
	assign non_zero = operant_b[9:0] != 0;
	assign sign = (((non_zero & operant_a[2]) & operant_b[15]) & ~operant_b[16]) & ~normal_mul;
	wire [16:0] oper_b;
	assign oper_b = (normal_mul ? operant_b : {{6 {operant_b[10]}}, operant_b[10:0]});
	assign temp_outcome = $signed(operant_a) * $signed(oper_b);
	assign out_res = temp_outcome + {21'b000000000000000000000, sign, 12'b000000000000};
endmodule
