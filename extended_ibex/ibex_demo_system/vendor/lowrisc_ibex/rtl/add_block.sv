module add_block(
/* verilator lint_off UNUSEDSIGNAL */
    input logic [31:0] p_a[2],
    input logic [31:0] p_b[2],
    input logic mode_3,
    
    output logic [31:0] sums[2]
);

    logic [31:0] temp_sum[2];
    logic [31:0] mode_3_temp_sum;
    logic [31:0] temp_res[2];
    
    assign mode_3_temp_sum = temp_sum[0] + temp_sum[1];
    assign temp_res[0] = {{20{mode_3_temp_sum[27]}}, mode_3_temp_sum[27:16]};
    assign temp_res[1] = {{20{mode_3_temp_sum[11]}}, mode_3_temp_sum[11:0]};
    
    assign sums[0] = mode_3 ? temp_res[0] : temp_sum[0];
    assign sums[1] = mode_3 ? temp_res[1] : temp_sum[1];
    
    adder_32_bits A0(
        .operant_a  (p_a[0]         ),
        .operant_b  (p_b[0]         ),
        .mode_3     (mode_3         ),
        .out_res    (temp_sum[0]    )
    );
    
    adder_32_bits A1(
        .operant_a  (p_a[1]         ),
        .operant_b  (p_b[1]         ),
        .mode_3     (mode_3         ),
        .out_res    (temp_sum[1]    )
    );

endmodule

