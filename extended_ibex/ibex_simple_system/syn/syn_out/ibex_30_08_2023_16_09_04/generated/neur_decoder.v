module neur_decoder (
	rst_ni,
	clk_i,
	enable,
	inputs,
	weights,
	mode,
	operant_a0,
	operant_b0,
	operant_a1,
	operant_b1,
	operant_a2,
	operant_b2,
	operant_a3,
	operant_b3
);
	input wire rst_ni;
	input wire clk_i;
	input wire enable;
	input wire [31:0] inputs;
	input wire [31:0] weights;
	input wire [2:0] mode;
	output wire [15:0] operant_a0;
	output wire [15:0] operant_b0;
	output wire [15:0] operant_a1;
	output wire [15:0] operant_b1;
	output wire [15:0] operant_a2;
	output wire [15:0] operant_b2;
	output wire [15:0] operant_a3;
	output wire [15:0] operant_b3;
	wire input_unsigned;
	assign input_unsigned = ~mode[2];
	reg [15:0] decoded_a0;
	reg [15:0] decoded_a1;
	reg [15:0] decoded_a2;
	reg [15:0] decoded_a3;
	reg [15:0] decoded_b0;
	reg [15:0] decoded_b1;
	reg [15:0] decoded_b2;
	reg [15:0] decoded_b3;
	assign operant_a0 = (enable ? decoded_a0 : 16'b0000000000000000);
	assign operant_b0 = (enable ? decoded_b0 : 16'b0000000000000000);
	assign operant_a1 = (enable ? decoded_a1 : 16'b0000000000000000);
	assign operant_b1 = (enable ? decoded_b1 : 16'b0000000000000000);
	assign operant_a2 = (enable ? decoded_a2 : 16'b0000000000000000);
	assign operant_b2 = (enable ? decoded_b2 : 16'b0000000000000000);
	assign operant_a3 = (enable ? decoded_a3 : 16'b0000000000000000);
	assign operant_b3 = (enable ? decoded_b3 : 16'b0000000000000000);
	reg [1:0] state_q;
	reg [1:0] state_d;
	always @(*) begin
		state_d = state_q;
		case (state_q)
			2'd0: begin
				state_d = 2'd1;
				case (mode[1:0])
					2'b00: begin
						decoded_a0 = {{8 {weights[31]}}, weights[28+:4], weights[24+:4]};
						decoded_b0 = {inputs[16+:8], inputs[24+:8]};
						decoded_a1 = {{8 {weights[31]}}, weights[28+:4], weights[24+:4]};
						decoded_b1 = {inputs[0+:8], inputs[8+:8]};
						decoded_a2 = {16 {weights[31]}};
						decoded_b2 = {inputs[16+:8], inputs[24+:8]};
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b01: begin
						decoded_a0 = {{8 {weights[31]}}, weights[28+:4], weights[24+:4]};
						decoded_b0 = {inputs[0+:8], inputs[8+:8]};
						decoded_a1 = {{8 {weights[23]}}, weights[20+:4], weights[16+:4]};
						decoded_b1 = {inputs[16+:8], inputs[24+:8]};
						decoded_a2 = 16'b0000000000000000;
						decoded_b2 = 16'b0000000000000000;
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b10: begin
						decoded_a0 = {{12 {weights[31]}}, weights[28+:4]};
						decoded_b0 = {inputs[0+:8], inputs[8+:8]};
						decoded_a1 = {{12 {weights[27]}}, weights[24+:4]};
						decoded_b1 = {inputs[16+:8], inputs[24+:8]};
						decoded_a2 = 16'b0000000000000000;
						decoded_b2 = 16'b0000000000000000;
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b11: begin
						decoded_a0 = {{12 {weights[31]}}, weights[28+:4]};
						decoded_b0 = {{8 {input_unsigned & inputs[7]}}, inputs[0+:8]};
						decoded_a1 = {{12 {weights[27]}}, weights[24+:4]};
						decoded_b1 = {{8 {input_unsigned & inputs[15]}}, inputs[8+:8]};
						decoded_a2 = {{12 {weights[23]}}, weights[20+:4]};
						decoded_b2 = {{8 {input_unsigned & inputs[23]}}, inputs[16+:8]};
						decoded_a3 = {{12 {weights[19]}}, weights[16+:4]};
						decoded_b3 = {{8 {input_unsigned & inputs[31]}}, inputs[24+:8]};
					end
				endcase
			end
			2'd1: begin
				state_d = 2'd2;
				case (mode[1:0])
					2'b00: begin
						decoded_a0 = {{8 {weights[23]}}, weights[20+:4], weights[16+:4]};
						decoded_b0 = {inputs[16+:8], inputs[24+:8]};
						decoded_a1 = {{8 {weights[23]}}, weights[20+:4], weights[16+:4]};
						decoded_b1 = {inputs[0+:8], inputs[8+:8]};
						decoded_a2 = {16 {weights[23]}};
						decoded_b2 = {inputs[16+:8], inputs[24+:8]};
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b01: begin
						decoded_a0 = {{8 {weights[15]}}, weights[12+:4], weights[8+:4]};
						decoded_b0 = {inputs[0+:8], inputs[8+:8]};
						decoded_a1 = {{8 {weights[7]}}, weights[4+:4], weights[0+:4]};
						decoded_b1 = {inputs[16+:8], inputs[24+:8]};
						decoded_a2 = 16'b0000000000000000;
						decoded_b2 = 16'b0000000000000000;
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b10: begin
						decoded_a0 = {{12 {weights[23]}}, weights[20+:4]};
						decoded_b0 = {inputs[0+:8], inputs[8+:8]};
						decoded_a1 = {{12 {weights[19]}}, weights[16+:4]};
						decoded_b1 = {inputs[16+:8], inputs[24+:8]};
						decoded_a2 = 16'b0000000000000000;
						decoded_b2 = 16'b0000000000000000;
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b11: begin
						decoded_a0 = {{12 {weights[15]}}, weights[12+:4]};
						decoded_b0 = {{8 {input_unsigned & inputs[7]}}, inputs[0+:8]};
						decoded_a1 = {{12 {weights[11]}}, weights[8+:4]};
						decoded_b1 = {{8 {input_unsigned & inputs[15]}}, inputs[8+:8]};
						decoded_a2 = {{12 {weights[7]}}, weights[4+:4]};
						decoded_b2 = {{8 {input_unsigned & inputs[23]}}, inputs[16+:8]};
						decoded_a3 = {{12 {weights[3]}}, weights[0+:4]};
						decoded_b3 = {{8 {input_unsigned & inputs[31]}}, inputs[24+:8]};
					end
				endcase
			end
			2'd2: begin
				state_d = 2'd3;
				case (mode[1:0])
					2'b00: begin
						decoded_a0 = {{8 {weights[15]}}, weights[12+:4], weights[8+:4]};
						decoded_b0 = {inputs[16+:8], inputs[24+:8]};
						decoded_a1 = {{8 {weights[15]}}, weights[12+:4], weights[8+:4]};
						decoded_b1 = {inputs[0+:8], inputs[8+:8]};
						decoded_a2 = {16 {weights[15]}};
						decoded_b2 = {inputs[16+:8], inputs[24+:8]};
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b01: begin
						decoded_a0 = {{8 {weights[31]}}, weights[28+:4], weights[24+:4]};
						decoded_b0 = {inputs[0+:8], inputs[8+:8]};
						decoded_a1 = {{8 {weights[23]}}, weights[20+:4], weights[16+:4]};
						decoded_b1 = {inputs[16+:8], inputs[24+:8]};
						decoded_a2 = 16'b0000000000000000;
						decoded_b2 = 16'b0000000000000000;
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b10: begin
						decoded_a0 = {{12 {weights[15]}}, weights[12+:4]};
						decoded_b0 = {inputs[0+:8], inputs[8+:8]};
						decoded_a1 = {{12 {weights[11]}}, weights[8+:4]};
						decoded_b1 = {inputs[16+:8], inputs[24+:8]};
						decoded_a2 = 16'b0000000000000000;
						decoded_b2 = 16'b0000000000000000;
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b11: begin
						decoded_a0 = {{12 {weights[31]}}, weights[28+:4]};
						decoded_b0 = {{8 {input_unsigned & inputs[7]}}, inputs[0+:8]};
						decoded_a1 = {{12 {weights[27]}}, weights[24+:4]};
						decoded_b1 = {{8 {input_unsigned & inputs[15]}}, inputs[8+:8]};
						decoded_a2 = {{12 {weights[23]}}, weights[20+:4]};
						decoded_b2 = {{8 {input_unsigned & inputs[23]}}, inputs[16+:8]};
						decoded_a3 = {{12 {weights[19]}}, weights[16+:4]};
						decoded_b3 = {{8 {input_unsigned & inputs[31]}}, inputs[24+:8]};
					end
				endcase
			end
			2'd3: begin
				state_d = 2'd0;
				case (mode[1:0])
					2'b00: begin
						decoded_a0 = {{8 {weights[7]}}, weights[4+:4], weights[0+:4]};
						decoded_b0 = {inputs[16+:8], inputs[24+:8]};
						decoded_a1 = {{8 {weights[7]}}, weights[4+:4], weights[0+:4]};
						decoded_b1 = {inputs[0+:8], inputs[8+:8]};
						decoded_a2 = {16 {weights[7]}};
						decoded_b2 = {inputs[16+:8], inputs[24+:8]};
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b01: begin
						decoded_a0 = {{8 {weights[15]}}, weights[12+:4], weights[8+:4]};
						decoded_b0 = {inputs[0+:8], inputs[8+:8]};
						decoded_a1 = {{8 {weights[7]}}, weights[4+:4], weights[0+:4]};
						decoded_b1 = {inputs[16+:8], inputs[24+:8]};
						decoded_a2 = 16'b0000000000000000;
						decoded_b2 = 16'b0000000000000000;
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b10: begin
						decoded_a0 = {{12 {weights[7]}}, weights[4+:4]};
						decoded_b0 = {inputs[0+:8], inputs[8+:8]};
						decoded_a1 = {{12 {weights[3]}}, weights[0+:4]};
						decoded_b1 = {inputs[16+:8], inputs[24+:8]};
						decoded_a2 = 16'b0000000000000000;
						decoded_b2 = 16'b0000000000000000;
						decoded_a3 = 16'b0000000000000000;
						decoded_b3 = 16'b0000000000000000;
					end
					2'b11: begin
						decoded_a0 = {{12 {weights[15]}}, weights[12+:4]};
						decoded_b0 = {{8 {input_unsigned & inputs[7]}}, inputs[0+:8]};
						decoded_a1 = {{12 {weights[11]}}, weights[8+:4]};
						decoded_b1 = {{8 {input_unsigned & inputs[15]}}, inputs[8+:8]};
						decoded_a2 = {{12 {weights[7]}}, weights[4+:4]};
						decoded_b2 = {{8 {input_unsigned & inputs[23]}}, inputs[16+:8]};
						decoded_a3 = {{12 {weights[3]}}, weights[0+:4]};
						decoded_b3 = {{8 {input_unsigned & inputs[31]}}, inputs[24+:8]};
					end
					default: state_d = 2'd0;
				endcase
			end
		endcase
	end
	always @(posedge clk_i or negedge rst_ni)
		if (!rst_ni)
			state_q <= 2'd0;
		else if (enable)
			state_q <= state_d;
endmodule
