module mult_block(
    input logic [16:0] weight_vals[4],
    input logic [16:0] activations[4],
    input logic normal_mul,
    output logic [33:0] partial_prods[4]
);

    logic [33:0] temp_prods[4];
    
    assign partial_prods[0] = temp_prods[0];
    assign partial_prods[1] = temp_prods[1];
    assign partial_prods[2] = temp_prods[2];
    assign partial_prods[3] = temp_prods[3];
    
    mult_16_bits M0(
        .operant_a  (weight_vals[0]     ),
        .operant_b  (activations[0]     ),
        .normal_mul (normal_mul	  ),
        .out_res    (temp_prods[0]      )
    );
    
    mult_16_bits M1(
        .operant_a  (weight_vals[1]     ),
        .operant_b  (activations[1]     ),
        .normal_mul (normal_mul	  ),
        .out_res    (temp_prods[1]      )
    );
    
    mult_16_bits M2(
        .operant_a  (weight_vals[2]     ),
        .operant_b  (activations[2]     ),
        .normal_mul (normal_mul	  ),
        .out_res    (temp_prods[2]      )
    );
    
    mult_16_bits M3(
        .operant_a  (weight_vals[3]     ),
        .operant_b  (activations[3]     ),
        .normal_mul (normal_mul	  ),
        .out_res    (temp_prods[3]      )
    );

endmodule

