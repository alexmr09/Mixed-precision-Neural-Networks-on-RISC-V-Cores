module neural_unit(
/* verilator lint_off UNUSEDSIGNAL */
    input logic clk_i,
    input logic clk_i_fast,
    input logic rstn_i,
    
    input logic bias_in,
    input logic valid_in,
    input logic get_res,
    
    input logic [31:0] bias_shift_mode,
    
    input logic [31:0] weights,
    input logic [31:0] input_val,
    
    input logic [31:0] out_mul_vals,
    input logic [31:0] out_shift_rl,
    
    input logic  [31:0] par_prods_in[4],
    output logic [16:0] weights_out[4],
    output logic [16:0] act_out[4],
    
    output logic [31:0] output_val,
    output logic valid_out 
);
    
    logic occupied, occupied_reg;
    logic [31:0] out_reg[4];
    logic [31:0] res_out;
    logic valid_in_reg[2];
    logic [1:0] iteration;
    logic [2:0] mode_reg;
    
    logic [31:0] weight_reg, weight_reg_transfered;
    logic [31:0] input_reg, input_reg_transfered;
    
    logic [16:0] weight_vals[8], weight_vals_reg[8];
    logic [16:0] activations[8], activations_reg[8];
    
    logic [31:0] mab_results[4];
    logic mab_valid;
    
    logic [16:0] q_mul[4];
    logic [16:0] q_out[4];
    logic res_out_valid;

    assign valid_out = bias_in | valid_in | (get_res & res_out_valid);
    assign output_val = get_res ? res_out : out_reg[0];

    assign weight_reg_transfered = weight_reg;
    assign input_reg_transfered = input_reg;
    
    neur_decoder neur_dec(
        .iteration      (iteration                  ),
        .mode           (mode_reg                   ),
        .weights_dec    (weight_reg_transfered      ),
        .input_vals     (input_reg_transfered       ),
        .weight_vals    (weight_vals                ),
        .activations    (activations                )
    );
    
    logic [16:0] weights_out_m[4], act_out_m[4];
    logic [31:0] par_prods_in_m[4], par_prods_in_reg[4];
    
    for(genvar i = 0; i < 4; i = i+1) begin
        assign weights_out[i] = (get_res & ~occupied) ? q_out[i] : weights_out_m[i];
        assign act_out[i] = (get_res & ~occupied) ? q_mul[i] : act_out_m[i];
        assign par_prods_in_m[i] = par_prods_in_reg[i];
    end
      
    mul_add_block MAB(
        .rst_ni         (rstn_i         ),
        .clk_i          (clk_i          ),
        .clk_i_fast     (clk_i_fast     ),
        .enable         (valid_in_reg[1]),
        .weight_vals    (weight_vals_reg),
        .activations    (activations_reg),
        
        .mode           (mode_reg[1:0]  ),
        
        .weights_out    (weights_out_m  ),
        .act_out        (act_out_m      ),
        .par_prods_in   (par_prods_in_m ),
        
        .results        (mab_results    ),
        .occupied       (occupied       ),
        .valid_out      (mab_valid      )
    );
    
    neur_out_unit neur_output(
        .clk_i_fast         (clk_i_fast         ),
        .clk_i              (clk_i              ),
        .get_res            (get_res & ~occupied),
        .rst_ni             (rstn_i             ),
        
        .out_mul_vals       (out_mul_vals       ),
        .out_shift_rl       (out_shift_rl       ),
        
        .out_results        (out_reg            ),
        .quant_products     (par_prods_in_m     ),
        
        .q_mul              (q_mul              ),
        .q_out              (q_out              ),
          
        .compressed_out     (res_out            ),
        .valid_out          (res_out_valid      )
    );
    
    always_ff @(posedge clk_i_fast or negedge rstn_i) begin
        if(~rstn_i) begin
            par_prods_in_reg <= {0, 0, 0, 0};
        end
        else begin
            for (int i = 0; i < 4; i++) begin
                par_prods_in_reg[i] <= par_prods_in[i];
            end
        end
    end
    
    always_ff @(posedge clk_i or negedge rstn_i) begin
        if(~rstn_i) begin
            mode_reg <= 0;
            occupied_reg <= 0;
            for (int i=0; i<4; i++) begin
                out_reg[i] <= 0;
            end
            iteration <= 0;
            valid_in_reg <= {0,0};
            weight_reg <= 0;
            input_reg <= 0;
            weight_vals_reg <= {0, 0, 0, 0, 0, 0, 0, 0};
            activations_reg <= {0, 0, 0, 0, 0, 0, 0, 0};
        end
        else begin
            valid_in_reg[0] <= valid_in;
            valid_in_reg[1] <= valid_in_reg[0];
            occupied_reg <= occupied;
            
            for (int i = 0; i < 8; i++) begin
                weight_vals_reg[i] <= weight_vals[i];
                activations_reg[i] <= activations[i];
            end
            
            if(bias_in) begin
                out_reg[0] <= {{24{weights[31]}}, weights[31:24]} << bias_shift_mode[31:27];
                out_reg[1] <= {{24{weights[23]}}, weights[23:16]} << bias_shift_mode[24:20];
                out_reg[2] <= {{24{weights[15]}}, weights[15:8]} << bias_shift_mode[17:13];
                out_reg[3] <= {{24{weights[7]}}, weights[7:0]} << bias_shift_mode[10:6];
                
                mode_reg <= bias_shift_mode[2:0];
                iteration <= 3;
            end
            
            if(valid_in) begin
                iteration <= iteration + 1;
                weight_reg <= weights;
                input_reg <= input_val;
            end
            
            if(mab_valid) begin
                for (int i=0; i<4; i++) begin
                    out_reg[i] <= out_reg[i] + mab_results[i];
                end
            end
        end
    end
    
endmodule
