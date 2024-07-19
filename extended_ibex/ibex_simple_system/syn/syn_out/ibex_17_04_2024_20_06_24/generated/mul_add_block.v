module mul_add_block (
	rst_ni,
	clk_i,
	clk_i_fast,
	enable,
	weight_vals,
	activations,
	mode,
	par_prods_in,
	weights_out,
	act_out,
	results,
	occupied,
	valid_out
);
	input wire rst_ni;
	input wire clk_i;
	input wire clk_i_fast;
	input wire enable;
	input wire [135:0] weight_vals;
	input wire [135:0] activations;
	input wire [1:0] mode;
	input wire [127:0] par_prods_in;
	output wire [67:0] weights_out;
	output wire [67:0] act_out;
	output wire [127:0] results;
	output wire occupied;
	output wire valid_out;
	wire [16:0] weight_transfered [0:3];
	wire [16:0] act_transfered [0:3];
	wire [31:0] par_prods [0:3];
	wire [63:0] p_a_transfered;
	wire [63:0] p_b_transfered;
	wire [63:0] add_results;
	reg [31:0] add_res_reg [0:3];
	reg [31:0] add_res_wire [0:3];
	reg [31:0] add_res_out [0:3];
	reg iter;
	reg valid_out_reg [0:1];
	reg prev_enable [0:2];
	wire mode_3;
	assign mode_3 = mode == 3;
	assign occupied = (((prev_enable[0] | prev_enable[1]) | prev_enable[2]) | valid_out_reg[0]) | valid_out_reg[1];
	assign p_a_transfered[32+:32] = par_prods[0];
	assign p_b_transfered[32+:32] = par_prods[1];
	assign p_a_transfered[0+:32] = par_prods[2];
	assign p_b_transfered[0+:32] = par_prods[3];
	assign valid_out = valid_out_reg[1];
	genvar i;
	generate
		for (i = 0; i < 4; i = i + 1) begin : genblk1
			assign results[(3 - i) * 32+:32] = add_res_out[i];
			assign weight_transfered[i] = weight_vals[(7 - ((4 * iter) + i)) * 17+:17];
			assign act_transfered[i] = activations[(7 - ((4 * iter) + i)) * 17+:17];
			assign weights_out[(3 - i) * 17+:17] = weight_transfered[i];
			assign act_out[(3 - i) * 17+:17] = act_transfered[i];
			assign par_prods[i] = par_prods_in[(3 - i) * 32+:32];
		end
	endgenerate
	add_block AB(
		.p_a(p_a_transfered),
		.p_b(p_b_transfered),
		.mode_3(mode_3),
		.sums(add_results)
	);
	always @(posedge clk_i or negedge rst_ni)
		if (~rst_ni) begin
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < 4; i = i + 1)
					add_res_out[i] <= 0;
			end
			valid_out_reg[0] <= 0;
			valid_out_reg[1] <= 0;
		end
		else begin
			begin : sv2v_autoblock_2
				reg signed [31:0] i;
				for (i = 0; i < 4; i = i + 1)
					add_res_out[i] <= add_res_wire[i];
			end
			valid_out_reg[0] <= prev_enable[2];
			valid_out_reg[1] <= valid_out_reg[0];
		end
	wire [1:1] sv2v_tmp_18290;
	assign sv2v_tmp_18290 = enable;
	always @(*) prev_enable[0] = sv2v_tmp_18290;
	always @(posedge clk_i_fast or negedge rst_ni)
		if (~rst_ni) begin
			begin : sv2v_autoblock_3
				reg signed [31:0] i;
				for (i = 0; i < 4; i = i + 1)
					begin
						add_res_reg[i] <= 0;
						add_res_wire[i] <= 0;
					end
			end
			prev_enable[1] <= 0;
			prev_enable[2] <= 0;
			iter <= 0;
		end
		else begin
			begin : sv2v_autoblock_4
				reg signed [31:0] i;
				for (i = 0; i < 4; i = i + 1)
					add_res_wire[i] <= add_res_reg[i];
			end
			add_res_reg[2 * iter] <= add_results[32+:32];
			add_res_reg[(2 * iter) + 1] <= add_results[0+:32];
			prev_enable[1] <= prev_enable[0];
			prev_enable[2] <= prev_enable[1];
			if (!(prev_enable[0] | prev_enable[1]))
				iter <= 0;
			else
				iter <= iter + 1;
		end
endmodule
