module neur_out_unit (
	clk_i,
	clk_i_fast,
	rst_ni,
	get_res,
	out_mul_vals,
	out_shift_rl,
	out_results,
	quant_products,
	q_mul,
	q_out,
	compressed_out,
	valid_out
);
	input wire clk_i;
	input wire clk_i_fast;
	input wire rst_ni;
	input wire get_res;
	input wire [31:0] out_mul_vals;
	input wire [31:0] out_shift_rl;
	input wire [127:0] out_results;
	input wire [127:0] quant_products;
	output wire [67:0] q_mul;
	output wire [67:0] q_out;
	output wire [31:0] compressed_out;
	output wire valid_out;
	wire signed [31:0] temp_results [0:3];
	reg signed [31:0] temp_results_reg [0:3];
	reg signed [31:0] temp_results_out [0:3];
	wire relu;
	assign relu = out_shift_rl[0];
	genvar i;
	reg sign [0:3];
	wire [16:0] quant_multiplier_vals [0:3];
	wire [31:0] unsigned_out_results [0:3];
	reg iter;
	generate
		for (i = 0; i < 4; i = i + 1) begin : genblk1
			assign quant_multiplier_vals[3 - i] = {9'b000000000, out_mul_vals[8 * i+:8]};
			assign unsigned_out_results[i] = (out_results[((3 - i) * 32) + 31] ? -out_results[(3 - i) * 32+:32] : out_results[(3 - i) * 32+:32]);
		end
	endgenerate
	assign q_mul[51+:17] = quant_multiplier_vals[2 * iter];
	assign q_out[51+:17] = {1'b0, unsigned_out_results[2 * iter][15:0]};
	assign q_mul[34+:17] = quant_multiplier_vals[2 * iter];
	assign q_out[34+:17] = {1'b0, unsigned_out_results[2 * iter][31:16]};
	assign q_mul[17+:17] = quant_multiplier_vals[(2 * iter) + 1];
	assign q_out[17+:17] = {1'b0, unsigned_out_results[(2 * iter) + 1][15:0]};
	assign q_mul[0+:17] = quant_multiplier_vals[(2 * iter) + 1];
	assign q_out[0+:17] = {1'b0, unsigned_out_results[(2 * iter) + 1][31:16]};
	wire [4:0] out_shifts [0:3];
	assign out_shifts[0] = out_shift_rl[31:27];
	assign out_shifts[1] = out_shift_rl[24:20];
	assign out_shifts[2] = out_shift_rl[17:13];
	assign out_shifts[3] = out_shift_rl[10:6];
	reg [127:0] quant_products_f;
	wire [31:0] rounded_add [0:3];
	assign rounded_add[0] = {31'b0000000000000000000000000000000, sign[0]} | (1 << (out_shifts[0] - 1));
	assign rounded_add[1] = {31'b0000000000000000000000000000000, sign[1]} | (1 << (out_shifts[1] - 1));
	assign rounded_add[2] = {31'b0000000000000000000000000000000, sign[2]} | (1 << (out_shifts[2] - 1));
	assign rounded_add[3] = {31'b0000000000000000000000000000000, sign[3]} | (1 << (out_shifts[3] - 1));
	wire [31:0] quant_mult_results [0:3];
	reg [31:0] quant_mult_results_reg [0:3];
	assign quant_mult_results[0] = {32 {sign[0]}} ^ quant_products_f[96+:32];
	assign quant_mult_results[1] = {32 {sign[1]}} ^ quant_products_f[64+:32];
	assign quant_mult_results[2] = {32 {sign[2]}} ^ quant_products_f[32+:32];
	assign quant_mult_results[3] = {32 {sign[3]}} ^ quant_products_f[0+:32];
	wire signed [31:0] rounded_results [0:3];
	assign rounded_results[0] = quant_mult_results_reg[0] + rounded_add[0];
	assign rounded_results[1] = quant_mult_results_reg[1] + rounded_add[1];
	assign rounded_results[2] = quant_mult_results_reg[2] + rounded_add[2];
	assign rounded_results[3] = quant_mult_results_reg[3] + rounded_add[3];
	assign temp_results[0] = rounded_results[0] >>> out_shifts[0];
	assign temp_results[1] = rounded_results[1] >>> out_shifts[1];
	assign temp_results[2] = rounded_results[2] >>> out_shifts[2];
	assign temp_results[3] = rounded_results[3] >>> out_shifts[3];
	reg [2:0] counter;
	wire val_out;
	assign val_out = counter == 3'b101;
	assign valid_out = val_out;
	assign compressed_out = (val_out ? {temp_results_out[0][7:0], temp_results_out[1][7:0], temp_results_out[2][7:0], temp_results_out[3][7:0]} : 32'b00000000000000000000000000000000);
	always @(posedge clk_i_fast or negedge rst_ni)
		if (!rst_ni) begin
			iter <= 0;
			quant_products_f <= 128'h00000000000000000000000000000000;
		end
		else if (get_res) begin
			iter <= iter + 1;
			quant_products_f[(3 - (2 * iter)) * 32+:32] <= quant_products[96+:32] + {quant_products[79-:16], 16'b0000000000000000};
			quant_products_f[(3 - ((2 * iter) + 1)) * 32+:32] <= quant_products[32+:32] + {quant_products[15-:16], 16'b0000000000000000};
		end
	always @(posedge clk_i or negedge rst_ni)
		if (!rst_ni) begin
			begin : sv2v_autoblock_1
				reg signed [31:0] j;
				for (j = 0; j < 4; j = j + 1)
					begin
						temp_results_reg[j] <= 0;
						quant_mult_results_reg[j] <= 0;
						sign[j] <= 0;
					end
			end
			counter <= 0;
		end
		else begin
			if (get_res)
				counter <= counter + 1;
			begin : sv2v_autoblock_2
				reg signed [31:0] j;
				for (j = 0; j < 4; j = j + 1)
					begin
						sign[j] <= out_results[((3 - j) * 32) + 31];
						temp_results_reg[j] <= temp_results[j];
						quant_mult_results_reg[j] <= quant_mult_results[j];
					end
			end
			if (counter == 3'b101)
				counter <= 0;
		end
	always @(*)
		if (relu) begin : sv2v_autoblock_3
			reg signed [31:0] j;
			for (j = 0; j < 4; j = j + 1)
				if (temp_results_reg[j] < 0)
					temp_results_out[j] = 0;
				else if (temp_results_reg[j] > 255)
					temp_results_out[j] = 255;
				else
					temp_results_out[j] = temp_results_reg[j];
		end
		else begin : sv2v_autoblock_4
			reg signed [31:0] j;
			for (j = 0; j < 4; j = j + 1)
				if (temp_results_reg[j] < -128)
					temp_results_out[j] = -128;
				else if (temp_results_reg[j] > 127)
					temp_results_out[j] = 127;
				else
					temp_results_out[j] = temp_results_reg[j];
		end
endmodule
