module adder_32_bits(
    input logic [31:0] operant_a,
    input logic [31:0] operant_b,
    input logic mode_3,
    output logic [31:0] out_res
);
    logic [31:0] add_oper_a, add_oper_b;
    
    assign add_oper_a = mode_3 ? {4'b0, {2{operant_a[21]}}, operant_a[21:12], 4'b0, {2{operant_a[9]}}, operant_a[9:0]} : operant_a;
    assign add_oper_b = mode_3 ? {4'b0, {2{operant_b[21]}}, operant_b[21:12], 4'b0, {2{operant_b[9]}}, operant_b[9:0]} : operant_b;
     
    assign out_res = add_oper_a + add_oper_b;

endmodule
