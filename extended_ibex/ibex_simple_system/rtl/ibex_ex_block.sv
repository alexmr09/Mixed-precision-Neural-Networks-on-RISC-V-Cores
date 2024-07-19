// Copyright lowRISC contributors.
// Copyright 2018 ETH Zurich and University of Bologna, see also CREDITS.md.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/**
 * Execution stage
 *
 * Execution block: Hosts ALU and MUL/DIV unit
 */
module ibex_ex_block #(
  parameter ibex_pkg::rv32m_e RV32M          = ibex_pkg::RV32MSingleCycle,
  parameter ibex_pkg::rv32b_e RV32B          = ibex_pkg::RV32BNone,
  parameter bit               BranchTargetALU = 0
) (
/* verilator lint_off UNUSEDSIGNAL */
  input  logic                  clk_i,
  input  logic                  clk_i_fast,
  input  logic                  rst_ni,

  // ALU
  input  ibex_pkg::alu_op_e     alu_operator_i,
  input  logic [31:0]           alu_operand_a_i,
  input  logic [31:0]           alu_operand_b_i,
  input  logic                  alu_instr_first_cycle_i,

  // Branch Target ALU
  // All of these signals are unusued when BranchTargetALU == 0
  input  logic [31:0]           bt_a_operand_i,
  input  logic [31:0]           bt_b_operand_i,

  // Multiplier/Divider
  input  ibex_pkg::md_op_e      multdiv_operator_i,
  input  logic                  mult_en_i,             // dynamic enable signal, for FSM control
  input  logic                  div_en_i,              // dynamic enable signal, for FSM control
  input  logic                  mult_sel_i,            // static decoder output, for data muxes
  input  logic                  div_sel_i,             // static decoder output, for data muxes
  input  logic  [1:0]           multdiv_signed_mode_i,
  input  logic [31:0]           multdiv_operand_a_i,
  input  logic [31:0]           multdiv_operand_b_i,
  input  logic                  multdiv_ready_id_i,
  input  logic                  data_ind_timing_i,

  // intermediate val reg
  output logic [1:0]            imd_val_we_o,
  output logic [33:0]           imd_val_d_o[2],
  input  logic [33:0]           imd_val_q_i[2],

  // Neural Unit ex_stage input values 
  input logic                   neur_valid_in_i,
  input logic                   neur_bias_in_i,
  input logic                   get_res_i,
  
  input logic [31:0]            neur_bias_shift_mode_i,
  
  input logic [31:0]            weights_i,
  input logic [31:0]            input_val_i,
  
  input logic [31:0]            neur_out_mul_vals_i,
  input logic [31:0]            neur_out_shift_rl_i,
  
  // Outputs
  output logic [31:0]           alu_adder_result_ex_o, // to LSU
  output logic [31:0]           result_ex_o,
  output logic [31:0]           branch_target_o,       // to IF
  output logic                  branch_decision_o,     // to ID

  output logic                  ex_valid_o             // EX has valid output
);

  import ibex_pkg::*;

  logic [31:0] alu_result, multdiv_result;

  logic [32:0] multdiv_alu_operand_b, multdiv_alu_operand_a;
  logic [33:0] alu_adder_result_ext;
  logic        alu_cmp_result, alu_is_equal_result;
  logic        multdiv_valid;
  logic        multdiv_sel;
  logic [31:0] alu_imd_val_q[2];
  logic [31:0] alu_imd_val_d[2];
  logic [ 1:0] alu_imd_val_we;
  logic [33:0] multdiv_imd_val_d[2];
  logic [ 1:0] multdiv_imd_val_we;

  logic [31:0] temp_result, neur_result;
  logic temp_valid, neur_valid;
  logic neur_sel;
  
  assign neur_sel = neur_valid_in_i | neur_bias_in_i | get_res_i;
  
  /*
    The multdiv_i output is never selected if RV32M=RV32MNone
    At synthesis time, all the combinational and sequential logic
    from the multdiv_i module are eliminated
  */
  if (RV32M != RV32MNone) begin : gen_multdiv_m
    assign multdiv_sel = mult_sel_i | div_sel_i;
  end else begin : gen_multdiv_no_m
    assign multdiv_sel = 1'b0;
  end

 
  // Intermediate Value Register Mux
  assign imd_val_d_o[0] = multdiv_sel ? multdiv_imd_val_d[0] : {2'b0, alu_imd_val_d[0]};
  assign imd_val_d_o[1] = multdiv_sel ? multdiv_imd_val_d[1] : {2'b0, alu_imd_val_d[1]};
  assign imd_val_we_o   = multdiv_sel ? multdiv_imd_val_we : alu_imd_val_we;

  assign alu_imd_val_q = '{imd_val_q_i[0][31:0], imd_val_q_i[1][31:0]};
  
  // We assign in the temp_result variable the output of the ALU or Multiplier Unit
  assign temp_result  = multdiv_sel ? multdiv_result : alu_result;
  
  // The output of the Execution block depends on whether or not neur instructions are executed
  assign result_ex_o = neur_sel ? neur_result : temp_result;
  
  // branch handling
  assign branch_decision_o  = alu_cmp_result;

  if (BranchTargetALU) begin : g_branch_target_alu
    logic [32:0] bt_alu_result;
    logic        unused_bt_carry;

    assign bt_alu_result   = bt_a_operand_i + bt_b_operand_i;

    assign unused_bt_carry = bt_alu_result[32];
    assign branch_target_o = bt_alu_result[31:0];
  end else begin : g_no_branch_target_alu
    // Unused bt_operand signals cause lint errors, this avoids them
    logic [31:0] unused_bt_a_operand, unused_bt_b_operand;

    assign unused_bt_a_operand = bt_a_operand_i;
    assign unused_bt_b_operand = bt_b_operand_i;

    assign branch_target_o = alu_adder_result_ex_o;
  end


  ////////////////////////
  // CUSTOM NEURAL UNIT //
  ////////////////////////
  
  logic [16:0] w_vals[4], a_vals[4];
  logic [33:0] p_vals[4];
  
  logic [16:0] nu_w_vals[4], nu_a_vals[4];
  logic [31:0] nu_p_vals[4];
  
  logic [16:0] ib_w_vals[4], ib_a_vals[4];
  logic [33:0] ib_p_vals[4];
  
  for (genvar i=0; i<4; i=i+1) begin
  	assign w_vals[i] = multdiv_sel ? ib_w_vals[i] : nu_w_vals[i];
  	assign a_vals[i] = multdiv_sel ? ib_a_vals[i] : nu_a_vals[i];
  end
  
  logic [16:0] mul_w_vals[4], mul_a_vals[4];
  
  always_ff @(posedge clk_i_fast or negedge rst_ni) begin
    if(~rst_ni) begin
        for(int i = 0; i < 4; i = i+1) begin
            mul_w_vals[i] <= 0;
            mul_a_vals[i] <= 0;
        end  
    end
    else begin
        for(int i = 0; i < 4; i = i+1) begin
            mul_w_vals[i] <= w_vals[i];
            mul_a_vals[i] <= a_vals[i];
        end  
    end
  end
  
  assign nu_p_vals[0] = p_vals[0][31:0];
  assign nu_p_vals[1] = p_vals[1][31:0];
  assign nu_p_vals[2] = p_vals[2][31:0];
  assign nu_p_vals[3] = p_vals[3][31:0];
  
  assign ib_p_vals[0] = p_vals[0];
  assign ib_p_vals[1] = p_vals[1];
  assign ib_p_vals[2] = p_vals[2];
  
    
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
  	.weight_vals    (mul_w_vals  ),
  	.activations    (mul_a_vals  ),
  	.normal_mul	 (mult_en_i  ),
  	.partial_prods  (p_vals      )
  );

  /////////
  // ALU //
  /////////

  ibex_alu #(
    .RV32B(RV32B)
  ) alu_i (
    .operator_i         (alu_operator_i),
    .operand_a_i        (alu_operand_a_i),
    .operand_b_i        (alu_operand_b_i),
    .instr_first_cycle_i(alu_instr_first_cycle_i),
    .imd_val_q_i        (alu_imd_val_q),
    .imd_val_we_o       (alu_imd_val_we),
    .imd_val_d_o        (alu_imd_val_d),
    .multdiv_operand_a_i(multdiv_alu_operand_a),
    .multdiv_operand_b_i(multdiv_alu_operand_b),
    .multdiv_sel_i      (multdiv_sel),
    .adder_result_o     (alu_adder_result_ex_o),
    .adder_result_ext_o (alu_adder_result_ext),
    .result_o           (alu_result),
    .comparison_result_o(alu_cmp_result),
    .is_equal_result_o  (alu_is_equal_result)
  );

  ////////////////
  // Multiplier //
  ////////////////

  if (RV32M == RV32MSlow) begin : gen_multdiv_slow
    ibex_multdiv_slow multdiv_i (
      .clk_i             (clk_i),
      .rst_ni            (rst_ni),
      .mult_en_i         (mult_en_i),
      .div_en_i          (div_en_i),
      .mult_sel_i        (mult_sel_i),
      .div_sel_i         (div_sel_i),
      .operator_i        (multdiv_operator_i),
      .signed_mode_i     (multdiv_signed_mode_i),
      .op_a_i            (multdiv_operand_a_i),
      .op_b_i            (multdiv_operand_b_i),
      .alu_adder_ext_i   (alu_adder_result_ext),
      .alu_adder_i       (alu_adder_result_ex_o),
      .equal_to_zero_i   (alu_is_equal_result),
      .data_ind_timing_i (data_ind_timing_i),
      .valid_o           (multdiv_valid),
      .alu_operand_a_o   (multdiv_alu_operand_a),
      .alu_operand_b_o   (multdiv_alu_operand_b),
      .imd_val_q_i       (imd_val_q_i),
      .imd_val_d_o       (multdiv_imd_val_d),
      .imd_val_we_o      (multdiv_imd_val_we),
      .multdiv_ready_id_i(multdiv_ready_id_i),
      .multdiv_result_o  (multdiv_result)
    );
  end else if (RV32M == RV32MFast || RV32M == RV32MSingleCycle) begin : gen_multdiv_fast
    ibex_multdiv_fast #(
      .RV32M(RV32M)
    ) multdiv_i (
      .clk_i             (clk_i),
      .rst_ni            (rst_ni),
      .mult_en_i         (mult_en_i),
      .div_en_i          (div_en_i),
      .mult_sel_i        (mult_sel_i),
      .div_sel_i         (div_sel_i),
      .operator_i        (multdiv_operator_i),
      .signed_mode_i     (multdiv_signed_mode_i),
      .op_a_i            (multdiv_operand_a_i),
      .op_b_i            (multdiv_operand_b_i),
      .alu_operand_a_o   (multdiv_alu_operand_a),
      .alu_operand_b_o   (multdiv_alu_operand_b),
      .alu_adder_ext_i   (alu_adder_result_ext),
      .alu_adder_i       (alu_adder_result_ex_o),
      .equal_to_zero_i   (alu_is_equal_result),
      .data_ind_timing_i (data_ind_timing_i),
      .imd_val_q_i       (imd_val_q_i),
      .imd_val_d_o       (multdiv_imd_val_d),
      .imd_val_we_o      (multdiv_imd_val_we),
      .multdiv_ready_id_i(multdiv_ready_id_i),
      
      .ib_w_oper	  (ib_w_vals),
      .ib_a_oper	  (ib_a_vals),
      .ib_p_oper	  (ib_p_vals[0:2]),
      
      .valid_o           (multdiv_valid),
      .multdiv_result_o  (multdiv_result)
    );
  end

  // Multiplier/divider may require multiple cycles. The ALU output is valid in the same cycle
  // unless the intermediate result register is being written (which indicates this isn't the
  // final cycle of ALU operation).
  
  assign temp_valid = multdiv_sel ? multdiv_valid : ~(|alu_imd_val_we);
  
  assign ex_valid_o = neur_sel ? neur_valid : temp_valid;
  
endmodule
