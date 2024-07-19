module neur_control_unit (
	clk_i,
	rstn_i,
	bias_in,
	valid_in,
	get_res,
	mode,
	weights,
	input_val,
	position,
	out_options,
	mult_prod,
	mult_valid,
	mult_oper_a0,
	mult_oper_b0,
	mult_oper_a1,
	mult_oper_b1,
	mult_oper_a2,
	mult_oper_b2,
	mult_oper_a3,
	mult_oper_b3,
	neur_mult_en,
	neur_mode,
	output_val,
	valid_out
);
	input wire clk_i;
	input wire rstn_i;
	input wire bias_in;
	input wire valid_in;
	input wire get_res;
	input wire [31:0] mode;
	input wire [31:0] weights;
	input wire [31:0] input_val;
	input wire [31:0] position;
	input wire [31:0] out_options;
	input wire [31:0] mult_prod;
	input wire mult_valid;
	output wire [15:0] mult_oper_a0;
	output wire [15:0] mult_oper_b0;
	output wire [15:0] mult_oper_a1;
	output wire [15:0] mult_oper_b1;
	output wire [15:0] mult_oper_a2;
	output wire [15:0] mult_oper_b2;
	output wire [15:0] mult_oper_a3;
	output wire [15:0] mult_oper_b3;
	output wire neur_mult_en;
	output wire [1:0] neur_mode;
	output wire [31:0] output_val;
	output wire valid_out;
	reg start;
	reg [2:0] mode_reg;
	reg [1:0] mul_counter;
	reg [1:0] mul_pos;
	reg [31:0] weight_reg;
	reg [31:0] input_reg;
	wire [15:0] neur_oper_a0;
	wire [15:0] neur_oper_a1;
	wire [15:0] neur_oper_a2;
	wire [15:0] neur_oper_a3;
	wire [15:0] neur_oper_b0;
	wire [15:0] neur_oper_b1;
	wire [15:0] neur_oper_b2;
	wire [15:0] neur_oper_b3;
	reg [127:0] out_reg;
	wire [31:0] res_out;
	assign valid_out = (bias_in | valid_in) | get_res;
	assign output_val = (get_res ? res_out : out_reg[96+:32]);
	assign mult_oper_a0 = (start ? neur_oper_a0 : 16'b0000000000000000);
	assign mult_oper_b0 = (start ? neur_oper_b0 : 16'b0000000000000000);
	assign mult_oper_a1 = (start ? neur_oper_a1 : 16'b0000000000000000);
	assign mult_oper_b1 = (start ? neur_oper_b1 : 16'b0000000000000000);
	assign mult_oper_a2 = (start ? neur_oper_a2 : 16'b0000000000000000);
	assign mult_oper_b2 = (start ? neur_oper_b2 : 16'b0000000000000000);
	assign mult_oper_a3 = (start ? neur_oper_a3 : 16'b0000000000000000);
	assign mult_oper_b3 = (start ? neur_oper_b3 : 16'b0000000000000000);
	assign neur_mult_en = start;
	assign neur_mode = mode_reg[1:0];
	neur_decoder neur_dec(
		.rst_ni(rstn_i),
		.clk_i(clk_i),
		.enable(start),
		.inputs(input_reg),
		.weights(weight_reg),
		.mode(mode_reg),
		.operant_a0(neur_oper_a0),
		.operant_b0(neur_oper_b0),
		.operant_a1(neur_oper_a1),
		.operant_b1(neur_oper_b1),
		.operant_a2(neur_oper_a2),
		.operant_b2(neur_oper_b2),
		.operant_a3(neur_oper_a3),
		.operant_b3(neur_oper_b3)
	);
	neur_out_unit neur_output(
		.position(position[1:0]),
		.out_options(out_options[1:0]),
		.out_results(out_reg),
		.compressed_out(res_out)
	);
	always @(posedge clk_i or negedge rstn_i)
		if (~rstn_i) begin
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < 4; i = i + 1)
					begin
						out_reg[(3 - i) * 32+:32] <= 0;
						input_reg[(3 - i) * 8+:8] <= 0;
						weight_reg[(7 - (2 * i)) * 4+:4] <= 0;
						weight_reg[(7 - ((2 * i) + 1)) * 4+:4] <= 0;
					end
			end
			mode_reg <= 0;
			mul_counter <= 0;
			mul_pos <= 0;
			start <= 0;
		end
		else begin
			if (bias_in) begin
				begin : sv2v_autoblock_2
					reg signed [31:0] i;
					for (i = 0; i < 4; i = i + 1)
						out_reg[(0 + i) * 32+:32] <= {{24 {weights[(8 * (i + 1)) - 1]}}, weights[i * 8+:8]};
				end
				mode_reg <= mode[2:0];
				mul_pos <= 0;
			end
			if (valid_in) begin
				begin : sv2v_autoblock_3
					reg signed [31:0] i;
					for (i = 0; i < 8; i = i + 1)
						weight_reg[(0 + i) * 4+:4] <= weights[i * 4+:4];
				end
				begin : sv2v_autoblock_4
					reg signed [31:0] i;
					for (i = 0; i < 4; i = i + 1)
						input_reg[(3 - i) * 8+:8] <= input_val[i * 8+:8];
				end
				start <= 1;
				mul_counter <= {1'b0, mode_reg[0]} << 1;
			end
			if (start & (mul_counter == 2'b11))
				start <= 0;
			if (mult_valid) begin
				mul_counter <= mul_counter + 1;
				mul_pos <= mul_pos + 1;
				out_reg[(3 - mul_pos) * 32+:32] <= out_reg[(3 - mul_pos) * 32+:32] + mult_prod;
			end
		end
endmodule
