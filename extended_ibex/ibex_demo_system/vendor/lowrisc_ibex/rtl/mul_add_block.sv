module mul_add_block(
    input logic rst_ni,
    input logic clk_i,
    input logic clk_i_fast,
    input logic enable,
    input logic [16:0] weight_vals[8],
    input logic [16:0] activations[8],
    
    input logic [1:0] mode,
    
    input logic  [31:0] par_prods_in[4],
    output logic [16:0] weights_out[4],
    output logic [16:0] act_out[4],
    
    output logic [31:0] results[4],
    output logic occupied,
    output logic valid_out
);

    logic [16:0] weight_transfered[4], act_transfered[4];
    logic [31:0] par_prods[4];
    
//    logic [31:0] par_prods_reg[4];
    logic [31:0] p_a_transfered[2], p_b_transfered[2];
    logic [31:0] add_results[2];
    logic [31:0] add_res_reg[4], add_res_wire[4], add_res_out[4];
    
    logic iter; 
    logic valid_out_reg[2], prev_enable[3];
    
    logic  mode_3;
    assign mode_3 = (mode == 3);
    
    assign occupied = (prev_enable[0] | prev_enable[1] | prev_enable[2] | valid_out_reg[0] | valid_out_reg[1]);
    
    assign p_a_transfered[0] = par_prods[0];
    assign p_b_transfered[0] = par_prods[1];
    
    assign p_a_transfered[1] = par_prods[2];
    assign p_b_transfered[1] = par_prods[3];
    
    assign valid_out = valid_out_reg[1];

    for(genvar i = 0; i < 4; i = i+1) begin
        assign results[i] = add_res_out[i];
        assign weight_transfered[i] = weight_vals[4*iter + i];
        assign act_transfered[i] = activations[4*iter + i];
        
        assign weights_out[i] = weight_transfered[i];
        assign act_out[i] = act_transfered[i];
        assign par_prods[i] = par_prods_in[i];
    end
    
    add_block AB(
        .p_a        (p_a_transfered ),
        .p_b        (p_b_transfered ),
        .mode_3     (mode_3         ),
        .sums       (add_results    )
    );

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if(~rst_ni) begin
            for(int i = 0; i < 4; i++) begin
                add_res_out[i] <= 0;
            end
            
            valid_out_reg[0] <= 0;
            valid_out_reg[1] <= 0;
        end
        else begin
        
            for(int i = 0; i < 4; i = i+1) begin
                add_res_out[i] <= add_res_wire[i]; 
            end
                        
            valid_out_reg[0] <= prev_enable[2];
            valid_out_reg[1] <= valid_out_reg[0];
        end
    end
    
    assign prev_enable[0] = enable;
    
    always_ff @(posedge clk_i_fast or negedge rst_ni) begin
        if(~rst_ni) begin
            for (int i = 0; i < 4; i++) begin
 //               par_prods_reg[i] <= 0; 
                add_res_reg[i] <= 0;
                add_res_wire[i] <= 0;
            end
            prev_enable[1] <= 0;
            prev_enable[2] <= 0; 
            iter <= 0;
        end
        else begin
            for (int i = 0; i < 4; i++) begin
 //               par_prods_reg[i] <= par_prods[i]; 
                add_res_wire[i] <= add_res_reg[i];  
            end
                
            add_res_reg[2*iter] <= add_results[0]; 
            add_res_reg[2*iter + 1] <= add_results[1];
            
            prev_enable[1] <= prev_enable[0];
            prev_enable[2] <= prev_enable[1];
            
            // Check if 'enable' goes from 0 to 1
            if(!(prev_enable[0] | prev_enable[1])) begin
                iter <= 0;
            end
            else begin
               iter <= iter+1;
            end         
        end
    end
    
endmodule

