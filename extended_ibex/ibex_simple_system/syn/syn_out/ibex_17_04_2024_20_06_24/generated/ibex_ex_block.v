module ibex_ex_block (
	clk_i,
	clk_i_fast,
	rst_ni,
	alu_operator_i,
	alu_operand_a_i,
	alu_operand_b_i,
	alu_instr_first_cycle_i,
	bt_a_operand_i,
	bt_b_operand_i,
	multdiv_operator_i,
	mult_en_i,
	div_en_i,
	mult_sel_i,
	div_sel_i,
	multdiv_signed_mode_i,
	multdiv_operand_a_i,
	multdiv_operand_b_i,
	multdiv_ready_id_i,
	data_ind_timing_i,
	imd_val_we_o,
	imd_val_d_o,
	imd_val_q_i,
	neur_valid_in_i,
	neur_bias_in_i,
	get_res_i,
	neur_bias_shift_mode_i,
	weights_i,
	input_val_i,
	neur_out_mul_vals_i,
	neur_out_shift_rl_i,
	alu_adder_result_ex_o,
	result_ex_o,
	branch_target_o,
	branch_decision_o,
	ex_valid_o
);
	parameter integer RV32M = 32'sd3;
	parameter integer RV32B = 32'sd0;
	parameter [0:0] BranchTargetALU = 0;
	input wire clk_i;
	input wire clk_i_fast;
	input wire rst_ni;
	input wire [6:0] alu_operator_i;
	input wire [31:0] alu_operand_a_i;
	input wire [31:0] alu_operand_b_i;
	input wire alu_instr_first_cycle_i;
	input wire [31:0] bt_a_operand_i;
	input wire [31:0] bt_b_operand_i;
	input wire [1:0] multdiv_operator_i;
	input wire mult_en_i;
	input wire div_en_i;
	input wire mult_sel_i;
	input wire div_sel_i;
	input wire [1:0] multdiv_signed_mode_i;
	input wire [31:0] multdiv_operand_a_i;
	input wire [31:0] multdiv_operand_b_i;
	input wire multdiv_ready_id_i;
	input wire data_ind_timing_i;
	output wire [1:0] imd_val_we_o;
	output wire [67:0] imd_val_d_o;
	input wire [67:0] imd_val_q_i;
	input wire neur_valid_in_i;
	input wire neur_bias_in_i;
	input wire get_res_i;
	input wire [31:0] neur_bias_shift_mode_i;
	input wire [31:0] weights_i;
	input wire [31:0] input_val_i;
	input wire [31:0] neur_out_mul_vals_i;
	input wire [31:0] neur_out_shift_rl_i;
	output wire [31:0] alu_adder_result_ex_o;
	output wire [31:0] result_ex_o;
	output wire [31:0] branch_target_o;
	output wire branch_decision_o;
	output wire ex_valid_o;
	wire [31:0] alu_result;
	wire [31:0] multdiv_result;
	wire [32:0] multdiv_alu_operand_b;
	wire [32:0] multdiv_alu_operand_a;
	wire [33:0] alu_adder_result_ext;
	wire alu_cmp_result;
	wire alu_is_equal_result;
	wire multdiv_valid;
	wire multdiv_sel;
	wire [63:0] alu_imd_val_q;
	wire [63:0] alu_imd_val_d;
	wire [1:0] alu_imd_val_we;
	wire [67:0] multdiv_imd_val_d;
	wire [1:0] multdiv_imd_val_we;
	wire [31:0] temp_result;
	wire [31:0] neur_result;
	wire temp_valid;
	wire neur_valid;
	wire neur_sel;
	assign neur_sel = (neur_valid_in_i | neur_bias_in_i) | get_res_i;
	generate
		if (RV32M != 32'sd0) begin : gen_multdiv_m
			assign multdiv_sel = mult_sel_i | div_sel_i;
		end
		else begin : gen_multdiv_no_m
			assign multdiv_sel = 1'b0;
		end
	endgenerate
	assign imd_val_d_o[34+:34] = (multdiv_sel ? multdiv_imd_val_d[34+:34] : {2'b00, alu_imd_val_d[32+:32]});
	assign imd_val_d_o[0+:34] = (multdiv_sel ? multdiv_imd_val_d[0+:34] : {2'b00, alu_imd_val_d[0+:32]});
	assign imd_val_we_o = (multdiv_sel ? multdiv_imd_val_we : alu_imd_val_we);
	assign alu_imd_val_q = {imd_val_q_i[65-:32], imd_val_q_i[31-:32]};
	assign temp_result = (multdiv_sel ? multdiv_result : alu_result);
	assign result_ex_o = (neur_sel ? neur_result : temp_result);
	assign branch_decision_o = alu_cmp_result;
	generate
		if (BranchTargetALU) begin : g_branch_target_alu
			wire [32:0] bt_alu_result;
			wire unused_bt_carry;
			assign bt_alu_result = bt_a_operand_i + bt_b_operand_i;
			assign unused_bt_carry = bt_alu_result[32];
			assign branch_target_o = bt_alu_result[31:0];
		end
		else begin : g_no_branch_target_alu
			wire [31:0] unused_bt_a_operand;
			wire [31:0] unused_bt_b_operand;
			assign unused_bt_a_operand = bt_a_operand_i;
			assign unused_bt_b_operand = bt_b_operand_i;
			assign branch_target_o = alu_adder_result_ex_o;
		end
	endgenerate
	wire [16:0] w_vals [0:3];
	wire [16:0] a_vals [0:3];
	wire [135:0] p_vals;
	wire [67:0] nu_w_vals;
	wire [67:0] nu_a_vals;
	wire [127:0] nu_p_vals;
	wire [67:0] ib_w_vals;
	wire [67:0] ib_a_vals;
	wire [135:0] ib_p_vals;
	genvar i;
	generate
		for (i = 0; i < 4; i = i + 1) begin : genblk3
			assign w_vals[i] = (multdiv_sel ? ib_w_vals[(3 - i) * 17+:17] : nu_w_vals[(3 - i) * 17+:17]);
			assign a_vals[i] = (multdiv_sel ? ib_a_vals[(3 - i) * 17+:17] : nu_a_vals[(3 - i) * 17+:17]);
		end
	endgenerate
	reg [67:0] mul_w_vals;
	reg [67:0] mul_a_vals;
	always @(posedge clk_i_fast or negedge rst_ni)
		if (~rst_ni) begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < 4; i = i + 1)
				begin
					mul_w_vals[(3 - i) * 17+:17] <= 0;
					mul_a_vals[(3 - i) * 17+:17] <= 0;
				end
		end
		else begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = 0; i < 4; i = i + 1)
				begin
					mul_w_vals[(3 - i) * 17+:17] <= w_vals[i];
					mul_a_vals[(3 - i) * 17+:17] <= a_vals[i];
				end
		end
	assign nu_p_vals[96+:32] = p_vals[133-:32];
	assign nu_p_vals[64+:32] = p_vals[99-:32];
	assign nu_p_vals[32+:32] = p_vals[65-:32];
	assign nu_p_vals[0+:32] = p_vals[31-:32];
	assign ib_p_vals[102+:34] = p_vals[102+:34];
	assign ib_p_vals[68+:34] = p_vals[68+:34];
	assign ib_p_vals[34+:34] = p_vals[34+:34];
	neural_unit nnu(
		.clk_i(clk_i),
		.clk_i_fast(clk_i_fast),
		.rstn_i(rst_ni),
		.valid_in(neur_valid_in_i),
		.bias_in(neur_bias_in_i),
		.get_res(get_res_i),
		.bias_shift_mode(neur_bias_shift_mode_i),
		.weights(weights_i),
		.input_val(input_val_i),
		.out_mul_vals(neur_out_mul_vals_i),
		.out_shift_rl(neur_out_shift_rl_i),
		.weights_out(nu_w_vals),
		.act_out(nu_a_vals),
		.par_prods_in(nu_p_vals),
		.output_val(neur_result),
		.valid_out(neur_valid)
	);
	mult_block mul_sys(
		.weight_vals(mul_w_vals),
		.activations(mul_a_vals),
		.normal_mul(mult_en_i),
		.partial_prods(p_vals)
	);
	ibex_alu #(.RV32B(RV32B)) alu_i(
		.operator_i(alu_operator_i),
		.operand_a_i(alu_operand_a_i),
		.operand_b_i(alu_operand_b_i),
		.instr_first_cycle_i(alu_instr_first_cycle_i),
		.imd_val_q_i(alu_imd_val_q),
		.imd_val_we_o(alu_imd_val_we),
		.imd_val_d_o(alu_imd_val_d),
		.multdiv_operand_a_i(multdiv_alu_operand_a),
		.multdiv_operand_b_i(multdiv_alu_operand_b),
		.multdiv_sel_i(multdiv_sel),
		.adder_result_o(alu_adder_result_ex_o),
		.adder_result_ext_o(alu_adder_result_ext),
		.result_o(alu_result),
		.comparison_result_o(alu_cmp_result),
		.is_equal_result_o(alu_is_equal_result)
	);
	generate
		if (RV32M == 32'sd1) begin : gen_multdiv_slow
			ibex_multdiv_slow multdiv_i(
				.clk_i(clk_i),
				.rst_ni(rst_ni),
				.mult_en_i(mult_en_i),
				.div_en_i(div_en_i),
				.mult_sel_i(mult_sel_i),
				.div_sel_i(div_sel_i),
				.operator_i(multdiv_operator_i),
				.signed_mode_i(multdiv_signed_mode_i),
				.op_a_i(multdiv_operand_a_i),
				.op_b_i(multdiv_operand_b_i),
				.alu_adder_ext_i(alu_adder_result_ext),
				.alu_adder_i(alu_adder_result_ex_o),
				.equal_to_zero_i(alu_is_equal_result),
				.data_ind_timing_i(data_ind_timing_i),
				.valid_o(multdiv_valid),
				.alu_operand_a_o(multdiv_alu_operand_a),
				.alu_operand_b_o(multdiv_alu_operand_b),
				.imd_val_q_i(imd_val_q_i),
				.imd_val_d_o(multdiv_imd_val_d),
				.imd_val_we_o(multdiv_imd_val_we),
				.multdiv_ready_id_i(multdiv_ready_id_i),
				.multdiv_result_o(multdiv_result)
			);
		end
		else if ((RV32M == 32'sd2) || (RV32M == 32'sd3)) begin : gen_multdiv_fast
			ibex_multdiv_fast #(.RV32M(RV32M)) multdiv_i(
				.clk_i(clk_i),
				.rst_ni(rst_ni),
				.mult_en_i(mult_en_i),
				.div_en_i(div_en_i),
				.mult_sel_i(mult_sel_i),
				.div_sel_i(div_sel_i),
				.operator_i(multdiv_operator_i),
				.signed_mode_i(multdiv_signed_mode_i),
				.op_a_i(multdiv_operand_a_i),
				.op_b_i(multdiv_operand_b_i),
				.alu_operand_a_o(multdiv_alu_operand_a),
				.alu_operand_b_o(multdiv_alu_operand_b),
				.alu_adder_ext_i(alu_adder_result_ext),
				.alu_adder_i(alu_adder_result_ex_o),
				.equal_to_zero_i(alu_is_equal_result),
				.data_ind_timing_i(data_ind_timing_i),
				.imd_val_q_i(imd_val_q_i),
				.imd_val_d_o(multdiv_imd_val_d),
				.imd_val_we_o(multdiv_imd_val_we),
				.multdiv_ready_id_i(multdiv_ready_id_i),
				.ib_w_oper(ib_w_vals),
				.ib_a_oper(ib_a_vals),
				.ib_p_oper(ib_p_vals[34+:102]),
				.valid_o(multdiv_valid),
				.multdiv_result_o(multdiv_result)
			);
		end
	endgenerate
	assign temp_valid = (multdiv_sel ? multdiv_valid : ~(|alu_imd_val_we));
	assign ex_valid_o = (neur_sel ? neur_valid : temp_valid);
endmodule
