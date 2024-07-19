module neural_unit (
	clk_i,
	clk_i_fast,
	rstn_i,
	bias_in,
	valid_in,
	get_res,
	bias_shift_mode,
	weights,
	input_val,
	out_mul_vals,
	out_shift_rl,
	par_prods_in,
	weights_out,
	act_out,
	output_val,
	valid_out
);
	input wire clk_i;
	input wire clk_i_fast;
	input wire rstn_i;
	input wire bias_in;
	input wire valid_in;
	input wire get_res;
	input wire [31:0] bias_shift_mode;
	input wire [31:0] weights;
	input wire [31:0] input_val;
	input wire [31:0] out_mul_vals;
	input wire [31:0] out_shift_rl;
	input wire [127:0] par_prods_in;
	output wire [67:0] weights_out;
	output wire [67:0] act_out;
	output wire [31:0] output_val;
	output wire valid_out;
	wire occupied;
	reg occupied_reg;
	reg [127:0] out_reg;
	wire [31:0] res_out;
	reg [0:1] valid_in_reg;
	reg [1:0] iteration;
	reg [2:0] mode_reg;
	reg [31:0] weight_reg;
	wire [31:0] weight_reg_transfered;
	reg [31:0] input_reg;
	wire [31:0] input_reg_transfered;
	wire [135:0] weight_vals;
	reg [135:0] weight_vals_reg;
	wire [135:0] activations;
	reg [135:0] activations_reg;
	wire [127:0] mab_results;
	wire mab_valid;
	wire [67:0] q_mul;
	wire [67:0] q_out;
	wire res_out_valid;
	assign valid_out = (bias_in | valid_in) | (get_res & res_out_valid);
	assign output_val = (get_res ? res_out : out_reg[96+:32]);
	assign weight_reg_transfered = weight_reg;
	assign input_reg_transfered = input_reg;
	neur_decoder neur_dec(
		.iteration(iteration),
		.mode(mode_reg),
		.weights_dec(weight_reg_transfered),
		.input_vals(input_reg_transfered),
		.weight_vals(weight_vals),
		.activations(activations)
	);
	wire [67:0] weights_out_m;
	wire [67:0] act_out_m;
	wire [127:0] par_prods_in_m;
	reg [127:0] par_prods_in_reg;
	genvar i;
	generate
		for (i = 0; i < 4; i = i + 1) begin : genblk1
			assign weights_out[(3 - i) * 17+:17] = (get_res ? q_out[(3 - i) * 17+:17] : weights_out_m[(3 - i) * 17+:17]);
			assign act_out[(3 - i) * 17+:17] = (get_res ? q_mul[(3 - i) * 17+:17] : act_out_m[(3 - i) * 17+:17]);
			assign par_prods_in_m[(3 - i) * 32+:32] = par_prods_in_reg[(3 - i) * 32+:32];
		end
	endgenerate
	mul_add_block MAB(
		.rst_ni(rstn_i),
		.clk_i(clk_i),
		.clk_i_fast(clk_i_fast),
		.enable(valid_in_reg[1]),
		.weight_vals(weight_vals_reg),
		.activations(activations_reg),
		.mode(mode_reg[1:0]),
		.weights_out(weights_out_m),
		.act_out(act_out_m),
		.par_prods_in(par_prods_in_m),
		.results(mab_results),
		.occupied(occupied),
		.valid_out(mab_valid)
	);
	neur_out_unit neur_output(
		.clk_i_fast(clk_i_fast),
		.clk_i(clk_i),
		.get_res(get_res & ~occupied),
		.rst_ni(rstn_i),
		.out_mul_vals(out_mul_vals),
		.out_shift_rl(out_shift_rl),
		.out_results(out_reg),
		.quant_products(par_prods_in_m),
		.q_mul(q_mul),
		.q_out(q_out),
		.compressed_out(res_out),
		.valid_out(res_out_valid)
	);
	always @(posedge clk_i_fast or negedge rstn_i)
		if (~rstn_i)
			par_prods_in_reg <= 128'h00000000000000000000000000000000;
		else begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < 4; i = i + 1)
				par_prods_in_reg[(3 - i) * 32+:32] <= par_prods_in[(3 - i) * 32+:32];
		end
	always @(posedge clk_i or negedge rstn_i)
		if (~rstn_i) begin
			mode_reg <= 0;
			occupied_reg <= 0;
			begin : sv2v_autoblock_2
				reg signed [31:0] i;
				for (i = 0; i < 4; i = i + 1)
					out_reg[(3 - i) * 32+:32] <= 0;
			end
			iteration <= 0;
			valid_in_reg <= 2'h0;
			weight_reg <= 0;
			input_reg <= 0;
			weight_vals_reg <= 136'h0000000000000000000000000000000000;
			activations_reg <= 136'h0000000000000000000000000000000000;
		end
		else begin
			valid_in_reg[0] <= valid_in;
			valid_in_reg[1] <= valid_in_reg[0];
			occupied_reg <= occupied;
			begin : sv2v_autoblock_3
				reg signed [31:0] i;
				for (i = 0; i < 8; i = i + 1)
					begin
						weight_vals_reg[(7 - i) * 17+:17] <= weight_vals[(7 - i) * 17+:17];
						activations_reg[(7 - i) * 17+:17] <= activations[(7 - i) * 17+:17];
					end
			end
			if (bias_in) begin
				out_reg[96+:32] <= {{24 {weights[31]}}, weights[31:24]} << bias_shift_mode[31:27];
				out_reg[64+:32] <= {{24 {weights[23]}}, weights[23:16]} << bias_shift_mode[24:20];
				out_reg[32+:32] <= {{24 {weights[15]}}, weights[15:8]} << bias_shift_mode[17:13];
				out_reg[0+:32] <= {{24 {weights[7]}}, weights[7:0]} << bias_shift_mode[10:6];
				mode_reg <= bias_shift_mode[2:0];
				iteration <= 3;
			end
			if (valid_in) begin
				iteration <= iteration + 1;
				weight_reg <= weights;
				input_reg <= input_val;
			end
			if (mab_valid) begin : sv2v_autoblock_4
				reg signed [31:0] i;
				for (i = 0; i < 4; i = i + 1)
					out_reg[(3 - i) * 32+:32] <= out_reg[(3 - i) * 32+:32] + mab_results[(3 - i) * 32+:32];
			end
		end
endmodule
