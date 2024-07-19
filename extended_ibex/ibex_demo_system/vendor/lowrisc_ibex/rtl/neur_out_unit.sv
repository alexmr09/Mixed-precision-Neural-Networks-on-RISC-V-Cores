module neur_out_unit(
/* verilator lint_off UNUSEDSIGNAL */
    input logic clk_i,
    input logic clk_i_fast,
    input logic rst_ni,
    input logic get_res,
    
    input logic [31:0] out_mul_vals,
    input logic [31:0] out_shift_rl,
    
    input logic [31:0] out_results[4],
    input logic [31:0] quant_products[4],
    
    output logic [16:0] q_mul[4],
    output logic [16:0] q_out[4],
    output logic [31:0] compressed_out,
    output logic valid_out
);

    logic signed [31:0] temp_results[4];
    logic signed [31:0] temp_results_reg[4], temp_results_out[4];
   
    logic relu;
    assign relu = out_shift_rl[0];
    genvar i;
    
    logic sign[4];
    logic [16:0] quant_multiplier_vals[4]; 
    logic [31:0] unsigned_out_results[4];
    logic iter;
    
    for(i = 0; i < 4; i++) begin
        //assign sign[i] = out_results[i][31];
        assign quant_multiplier_vals[3-i] = {9'b0, out_mul_vals[8*i +: 8]};
        assign unsigned_out_results[i] = out_results[i][31] ? -out_results[i] : out_results[i];
    end
    
    assign q_mul[0] = quant_multiplier_vals[2*iter];
    assign q_out[0] = {1'b0, unsigned_out_results[2*iter][15:0]};
    
    assign q_mul[1] = quant_multiplier_vals[2*iter];
    assign q_out[1] = {1'b0, unsigned_out_results[2*iter][31:16]};
    
    assign q_mul[2] = quant_multiplier_vals[2*iter + 1];
    assign q_out[2] = {1'b0, unsigned_out_results[2*iter + 1][15:0]};
    
    assign q_mul[3] = quant_multiplier_vals[2*iter + 1];
    assign q_out[3] = {1'b0, unsigned_out_results[2*iter + 1][31:16]};

    logic [4:0] out_shifts[4];
    
    assign out_shifts[0] = out_shift_rl[31:27];
    assign out_shifts[1] = out_shift_rl[24:20];
    assign out_shifts[2] = out_shift_rl[17:13];
    assign out_shifts[3] = out_shift_rl[10:6];
    
    logic [31:0] quant_products_f[4];
    
    logic [31:0] rounded_add[4];
    assign rounded_add[0] = {31'b0, sign[0]} | (1 << (out_shifts[0] - 1));
    assign rounded_add[1] = {31'b0, sign[1]} | (1 << (out_shifts[1] - 1));
    assign rounded_add[2] = {31'b0, sign[2]} | (1 << (out_shifts[2] - 1));
    assign rounded_add[3] = {31'b0, sign[3]} | (1 << (out_shifts[3] - 1));
    
    logic [31:0] quant_mult_results[4], quant_mult_results_reg[4];    
    assign quant_mult_results[0] = ({32{sign[0]}} ^ quant_products_f[0]);
    assign quant_mult_results[1] = ({32{sign[1]}} ^ quant_products_f[1]);
    assign quant_mult_results[2] = ({32{sign[2]}} ^ quant_products_f[2]);
    assign quant_mult_results[3] = ({32{sign[3]}} ^ quant_products_f[3]);

    logic signed [31:0] rounded_results[4];
    assign rounded_results[0] = quant_mult_results_reg[0] + rounded_add[0];
    assign rounded_results[1] = quant_mult_results_reg[1] + rounded_add[1];
    assign rounded_results[2] = quant_mult_results_reg[2] + rounded_add[2];
    assign rounded_results[3] = quant_mult_results_reg[3] + rounded_add[3];
    
    assign temp_results[0] = rounded_results[0] >>> out_shifts[0];
    assign temp_results[1] = rounded_results[1] >>> out_shifts[1];
    assign temp_results[2] = rounded_results[2] >>> out_shifts[2];
    assign temp_results[3] = rounded_results[3] >>> out_shifts[3];
    
    logic [2:0] counter;
    logic val_out;
    
    assign val_out = (counter == 3'b101);
    assign valid_out = val_out;
    assign compressed_out = val_out ? {temp_results_out[0][7:0], temp_results_out[1][7:0], temp_results_out[2][7:0], temp_results_out[3][7:0]} : 32'b0; 

    always_ff @(posedge clk_i_fast or negedge rst_ni) begin 
        if(!rst_ni) begin
            iter <= 0;
            quant_products_f <= {0,0,0,0};
        end
        else begin
            if(get_res) begin
                iter <= iter + 1;
                quant_products_f[2*iter] <= quant_products[0] + {quant_products[1][15:0], 16'b0};
                quant_products_f[2*iter+1] <= quant_products[2] + {quant_products[3][15:0], 16'b0};
            end 
        end
    end
    
    always_ff @(posedge clk_i or negedge rst_ni) begin 
        if(!rst_ni) begin
            for(int j = 0; j < 4; j++) begin
                temp_results_reg[j] <= 0;
                quant_mult_results_reg[j] <= 0;
                sign[j] <= 0;
            end
            counter <= 0;
        end
        else begin
            if(get_res) begin
                counter <= counter + 1;
            end
            for(int j = 0; j < 4; j++) begin
                sign[j] <= out_results[j][31];
                temp_results_reg[j] <= temp_results[j];
                quant_mult_results_reg[j] <= quant_mult_results[j];
            end
            if(counter == 3'b101) begin
                counter <= 0;
            end
        end
    end
    
    always_comb begin         
        if(relu) begin
            for (int j = 0; j < 4; j++) begin
                if(temp_results_reg[j] < 0) begin
                    temp_results_out[j] = 0;
                end
                else if(temp_results_reg[j] > 255) begin
                    temp_results_out[j] = 255;
                end
                else begin
                    temp_results_out[j] = temp_results_reg[j];
                end
            end 
        end
        else begin
            for (int j = 0; j < 4; j++) begin
                if(temp_results_reg[j] < -128) begin
                    temp_results_out[j] = -128;
                end
                else if(temp_results_reg[j] > 127) begin
                    temp_results_out[j] = 127;
                end
                else begin
                    temp_results_out[j] = temp_results_reg[j];
                end
            end 
        end
    end
endmodule
