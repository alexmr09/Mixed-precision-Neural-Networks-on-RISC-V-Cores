module mult_16_bits(
/* verilator lint_off UNUSEDSIGNAL */
    input logic [16:0] operant_a,
    input logic [16:0] operant_b,
    input logic normal_mul,
    output logic [33:0] out_res
);

    logic [33:0] temp_outcome;
    logic non_zero, sign;
    
    assign non_zero = operant_b[9:0] != 0;
    
    assign sign = non_zero & operant_a[2] & operant_b[15] & ~operant_b[16] & ~normal_mul;
    
    logic [16:0] oper_b; 
    assign oper_b = normal_mul? operant_b : {{6{operant_b[10]}}, operant_b[10:0]};
    
    assign temp_outcome = $signed(operant_a) * $signed(oper_b);
    
    assign out_res = temp_outcome + {21'b0, sign, 12'b0};
    
endmodule
