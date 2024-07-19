module ibex_ex_block (
	clk_i,
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
	neur_mode_i,
	weights_i,
	input_val_i,
	neur_out_options_i,
	neur_position_i,
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
	input wire [31:0] neur_mode_i;
	input wire [31:0] weights_i;
	input wire [31:0] input_val_i;
	input wire [31:0] neur_out_options_i;
	input wire [31:0] neur_position_i;
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
	wire neur_mult_en;
	assign imd_val_d_o[34+:34] = (multdiv_sel | neur_mult_en ? multdiv_imd_val_d[34+:34] : {2'b00, alu_imd_val_d[32+:32]});
	assign imd_val_d_o[0+:34] = (multdiv_sel | neur_mult_en ? multdiv_imd_val_d[0+:34] : {2'b00, alu_imd_val_d[0+:32]});
	assign imd_val_we_o = (multdiv_sel | neur_mult_en ? multdiv_imd_val_we : alu_imd_val_we);
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
	wire [15:0] neur_oper_a0;
	wire [15:0] neur_oper_b0;
	wire [15:0] neur_oper_a1;
	wire [15:0] neur_oper_b1;
	wire [15:0] neur_oper_a2;
	wire [15:0] neur_oper_b2;
	wire [15:0] neur_oper_a3;
	wire [15:0] neur_oper_b3;
	wire [31:0] mult_prod;
	wire mult_valid;
	wire [1:0] neur_mode;
	neur_control_unit nnu(
		.clk_i(clk_i),
		.rstn_i(rst_ni),
		.valid_in(neur_valid_in_i),
		.bias_in(neur_bias_in_i),
		.get_res(get_res_i),
		.mode(neur_mode_i),
		.weights(weights_i),
		.input_val(input_val_i),
		.position(neur_position_i),
		.out_options(neur_out_options_i),
		.mult_prod(mult_prod),
		.mult_valid(mult_valid),
		.mult_oper_a0(neur_oper_a0),
		.mult_oper_b0(neur_oper_b0),
		.mult_oper_a1(neur_oper_a1),
		.mult_oper_b1(neur_oper_b1),
		.mult_oper_a2(neur_oper_a2),
		.mult_oper_b2(neur_oper_b2),
		.mult_oper_a3(neur_oper_a3),
		.mult_oper_b3(neur_oper_b3),
		.neur_mult_en(neur_mult_en),
		.neur_mode(neur_mode),
		.output_val(neur_result),
		.valid_out(neur_valid)
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
				.valid_o(multdiv_valid),
				.multdiv_result_o(multdiv_result),
				.neur_mode(neur_mode),
				.neur_oper_a0(neur_oper_a0),
				.neur_oper_b0(neur_oper_b0),
				.neur_oper_a1(neur_oper_a1),
				.neur_oper_b1(neur_oper_b1),
				.neur_oper_a2(neur_oper_a2),
				.neur_oper_b2(neur_oper_b2),
				.neur_oper_a3(neur_oper_a3),
				.neur_oper_b3(neur_oper_b3),
				.neur_mul_en(neur_mult_en),
				.neur_mul_res(mult_prod),
				.neur_mul_valid_o(mult_valid)
			);
		end
	endgenerate
	assign temp_valid = (multdiv_sel ? multdiv_valid : ~(|alu_imd_val_we));
	assign ex_valid_o = (neur_sel ? neur_valid : temp_valid);
endmodule
