module neur_decoder (
	iteration,
	mode,
	weights_dec,
	input_vals,
	weight_vals,
	activations
);
	input wire [1:0] iteration;
	input wire [2:0] mode;
	input wire [31:0] weights_dec;
	input wire [31:0] input_vals;
	output wire [135:0] weight_vals;
	output wire [135:0] activations;
	reg [16:0] w_vals [0:7];
	reg [16:0] a_vals [0:7];
	genvar i;
	generate
		for (i = 0; i < 4; i = i + 1) begin : genblk1
			assign weight_vals[(7 - (2 * i)) * 17+:17] = w_vals[2 * i];
			assign weight_vals[(7 - ((2 * i) + 1)) * 17+:17] = w_vals[(2 * i) + 1];
			assign activations[(7 - (2 * i)) * 17+:17] = a_vals[2 * i];
			assign activations[(7 - ((2 * i) + 1)) * 17+:17] = a_vals[(2 * i) + 1];
		end
	endgenerate
	wire [16:0] input_values [0:3];
	wire [2:0] weights_temp_mode_3 [0:15];
	wire signed_input;
	assign signed_input = mode[2];
	generate
		for (i = 0; i < 4; i = i + 1) begin : genblk2
			assign input_values[3 - i] = {{9 {signed_input & input_vals[(8 * (i + 1)) - 1]}}, input_vals[i * 8+:8]};
		end
		for (i = 0; i < 16; i = i + 1) begin : genblk3
			assign weights_temp_mode_3[15 - i] = {weights_dec[(2 * (i + 1)) - 1], weights_dec[i * 2+:2]};
		end
	endgenerate
	always @(*)
		case (mode[1:0])
			2'b00: begin
				w_vals[0] = {{9 {weights_dec[31]}}, weights_dec[31:24]};
				w_vals[1] = 0;
				w_vals[2] = {{9 {weights_dec[23]}}, weights_dec[23:16]};
				w_vals[3] = 0;
				w_vals[4] = {{9 {weights_dec[15]}}, weights_dec[15:8]};
				w_vals[5] = 0;
				w_vals[6] = {{9 {weights_dec[7]}}, weights_dec[7:0]};
				w_vals[7] = 0;
				a_vals[0] = input_values[iteration];
				a_vals[1] = 0;
				a_vals[2] = input_values[iteration];
				a_vals[3] = 0;
				a_vals[4] = input_values[iteration];
				a_vals[5] = 0;
				a_vals[6] = input_values[iteration];
				a_vals[7] = 0;
			end
			2'b01: begin
				w_vals[0] = {{9 {weights_dec[31]}}, weights_dec[31:24]};
				w_vals[1] = 0;
				w_vals[2] = {{9 {weights_dec[23]}}, weights_dec[23:16]};
				w_vals[3] = 0;
				w_vals[4] = {{9 {weights_dec[15]}}, weights_dec[15:8]};
				w_vals[5] = 0;
				w_vals[6] = {{9 {weights_dec[7]}}, weights_dec[7:0]};
				w_vals[7] = 0;
				a_vals[0] = input_values[0];
				a_vals[1] = 0;
				a_vals[2] = input_values[1];
				a_vals[3] = 0;
				a_vals[4] = input_values[2];
				a_vals[5] = 0;
				a_vals[6] = input_values[3];
				a_vals[7] = 0;
			end
			2'b10: begin
				w_vals[0] = {{13 {weights_dec[31]}}, weights_dec[31:28]};
				w_vals[1] = {{13 {weights_dec[15]}}, weights_dec[15:12]};
				w_vals[2] = {{13 {weights_dec[27]}}, weights_dec[27:24]};
				w_vals[3] = {{13 {weights_dec[11]}}, weights_dec[11:8]};
				w_vals[4] = {{13 {weights_dec[23]}}, weights_dec[23:20]};
				w_vals[5] = {{13 {weights_dec[7]}}, weights_dec[7:4]};
				w_vals[6] = {{13 {weights_dec[19]}}, weights_dec[19:16]};
				w_vals[7] = {{13 {weights_dec[3]}}, weights_dec[3:0]};
				a_vals[0] = (iteration[0] ? input_values[2] : input_values[0]);
				a_vals[1] = (iteration[0] ? input_values[3] : input_values[1]);
				a_vals[2] = (iteration[0] ? input_values[2] : input_values[0]);
				a_vals[3] = (iteration[0] ? input_values[3] : input_values[1]);
				a_vals[4] = (iteration[0] ? input_values[2] : input_values[0]);
				a_vals[5] = (iteration[0] ? input_values[3] : input_values[1]);
				a_vals[6] = (iteration[0] ? input_values[2] : input_values[0]);
				a_vals[7] = (iteration[0] ? input_values[3] : input_values[1]);
			end
			2'b11: begin
				w_vals[0] = {{14 {weights_temp_mode_3[1][2]}}, weights_temp_mode_3[1]} + {{2 {weights_temp_mode_3[0][2]}}, weights_temp_mode_3[0], 12'b000000000000};
				w_vals[1] = {{14 {weights_temp_mode_3[5][2]}}, weights_temp_mode_3[5]} + {{2 {weights_temp_mode_3[4][2]}}, weights_temp_mode_3[4], 12'b000000000000};
				w_vals[2] = {{14 {weights_temp_mode_3[9][2]}}, weights_temp_mode_3[9]} + {{2 {weights_temp_mode_3[8][2]}}, weights_temp_mode_3[8], 12'b000000000000};
				w_vals[3] = {{14 {weights_temp_mode_3[13][2]}}, weights_temp_mode_3[13]} + {{2 {weights_temp_mode_3[12][2]}}, weights_temp_mode_3[12], 12'b000000000000};
				w_vals[4] = {{14 {weights_temp_mode_3[3][2]}}, weights_temp_mode_3[3]} + {{2 {weights_temp_mode_3[2][2]}}, weights_temp_mode_3[2], 12'b000000000000};
				w_vals[5] = {{14 {weights_temp_mode_3[7][2]}}, weights_temp_mode_3[7]} + {{2 {weights_temp_mode_3[6][2]}}, weights_temp_mode_3[6], 12'b000000000000};
				w_vals[6] = {{14 {weights_temp_mode_3[11][2]}}, weights_temp_mode_3[11]} + {{2 {weights_temp_mode_3[10][2]}}, weights_temp_mode_3[10], 12'b000000000000};
				w_vals[7] = {{14 {weights_temp_mode_3[15][2]}}, weights_temp_mode_3[15]} + {{2 {weights_temp_mode_3[14][2]}}, weights_temp_mode_3[14], 12'b000000000000};
				a_vals[0] = {7'b0100000, input_values[0][9:0]};
				a_vals[1] = {7'b0100000, input_values[1][9:0]};
				a_vals[2] = {7'b0100000, input_values[2][9:0]};
				a_vals[3] = {7'b0100000, input_values[3][9:0]};
				a_vals[4] = {7'b0100000, input_values[0][9:0]};
				a_vals[5] = {7'b0100000, input_values[1][9:0]};
				a_vals[6] = {7'b0100000, input_values[2][9:0]};
				a_vals[7] = {7'b0100000, input_values[3][9:0]};
			end
		endcase
endmodule
