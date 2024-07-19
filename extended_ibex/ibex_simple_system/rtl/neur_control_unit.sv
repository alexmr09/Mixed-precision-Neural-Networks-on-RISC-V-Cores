module neur_control_unit(
/* verilator lint_off UNUSEDSIGNAL */
    input logic clk_i,
    input logic rstn_i,
    
    input logic bias_in,
    input logic valid_in,
    input logic get_res,
    
    input logic [31:0] mode,
    input logic [31:0] weights,
    input logic [31:0] input_val,
    
    input logic [31:0] position,
    input logic [31:0] out_options,
    
    input logic [31:0] mult_prod,
    input logic mult_valid,
    
    output logic [15:0] mult_oper_a0,
    output logic [15:0] mult_oper_b0,
    
    output logic [15:0] mult_oper_a1,
    output logic [15:0] mult_oper_b1,
    
    output logic [15:0] mult_oper_a2,
    output logic [15:0] mult_oper_b2,
    
    output logic [15:0] mult_oper_a3,
    output logic [15:0] mult_oper_b3,
    
    output logic neur_mult_en,
    output logic [1:0] neur_mode,
    
    output logic [31:0] output_val,
    output logic valid_out 
);
    
    logic start;
    logic [2:0] mode_reg;
    logic [1:0] mul_counter, mul_pos;
    logic [3:0] weight_reg[8];
    logic [7:0] input_reg[4];
    logic [15:0] neur_oper_a0, neur_oper_a1, neur_oper_a2, neur_oper_a3;
    logic [15:0] neur_oper_b0, neur_oper_b1, neur_oper_b2, neur_oper_b3;
    logic [31:0] out_reg[4];
    logic [31:0] res_out;    

    assign valid_out = bias_in | valid_in | get_res;
    assign output_val = get_res ? res_out : out_reg[0];
    
    assign mult_oper_a0 = start ? neur_oper_a0 : 16'b0; 
    assign mult_oper_b0 = start ? neur_oper_b0 : 16'b0;
    
    assign mult_oper_a1 = start ? neur_oper_a1 : 16'b0; 
    assign mult_oper_b1 = start ? neur_oper_b1 : 16'b0;
    
    assign mult_oper_a2 = start ? neur_oper_a2 : 16'b0; 
    assign mult_oper_b2 = start ? neur_oper_b2 : 16'b0;
    
    assign mult_oper_a3 = start ? neur_oper_a3 : 16'b0; 
    assign mult_oper_b3 = start ? neur_oper_b3 : 16'b0;
    
    assign neur_mult_en = start;
    assign neur_mode = mode_reg[1:0];
    
    neur_decoder neur_dec(
        .rst_ni         (rstn_i         ),
        .clk_i          (clk_i          ),
        .enable         (start          ),
        .inputs         (input_reg      ),
        .weights        (weight_reg     ),
        .mode           (mode_reg       ),
        .operant_a0     (neur_oper_a0   ),
        .operant_b0     (neur_oper_b0   ),
        .operant_a1     (neur_oper_a1   ),
        .operant_b1     (neur_oper_b1   ),
        .operant_a2     (neur_oper_a2   ),
        .operant_b2     (neur_oper_b2   ),
        .operant_a3     (neur_oper_a3   ),
        .operant_b3     (neur_oper_b3   )
    );
    
    neur_out_unit neur_output(
        .position           (position[1:0]      ),
        .out_options        (out_options[1:0]   ),
        .out_results        (out_reg            ),
        .compressed_out     (res_out            )
    );
    
    always_ff @(posedge clk_i or negedge rstn_i) begin
        if(~rstn_i) begin
            for (int i=0; i<4; i++) begin
                out_reg[i] <= 0;
                input_reg[i] <= 0;
                weight_reg[2*i] <= 0;
                weight_reg[2*i+1] <= 0;
            end
            mode_reg <= 0;
            mul_counter <= 0;
            mul_pos <= 0;
            start <= 0;
        end
        else begin
            if(bias_in) begin
                for (int i=0; i<4; i++) begin
                    out_reg[3-i] <= {{24{weights[8*(i+1)-1]}}, weights[i * 8 +: 8]};
                end
                mode_reg <= mode[2:0];
                mul_pos <= 0;
            end
        
            if(valid_in) begin
                for (int i = 0; i < 8; i++) begin
                    weight_reg[7 - i] <= weights[i * 4 +: 4];
                end
            
                for (int i = 0; i < 4; i++) begin
                    input_reg[i] <= input_val[i * 8 +: 8];
                end
                start <= 1;
                mul_counter <= {1'b0, mode_reg[0]} << 1;
            end
        
            if(start & mul_counter == 2'b11) begin
                start <= 0;
            end
        
            if(mult_valid) begin
                mul_counter <= mul_counter + 1;
                mul_pos <= mul_pos + 1;
                out_reg[mul_pos] <= out_reg[mul_pos] + mult_prod;
            end
        end        			
    end
    
endmodule
