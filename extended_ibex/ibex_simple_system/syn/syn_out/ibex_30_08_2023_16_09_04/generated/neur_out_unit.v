module neur_out_unit (
	position,
	out_options,
	out_results,
	compressed_out
);
	input wire [1:0] position;
	input wire [1:0] out_options;
	input wire [127:0] out_results;
	output wire [31:0] compressed_out;
	wire [31:0] temp_results [0:1];
	wire sign [0:1];
	reg [31:0] outs;
	assign compressed_out = outs;
	wire relu;
	wire compression_en;
	assign relu = out_options[1];
	assign compression_en = out_options[0];
	genvar i;
	generate
		for (i = 0; i < 2; i = i + 1) begin : genblk1
			assign sign[i] = out_results[((3 - (position + i)) * 32) + 31] & relu;
			assign temp_results[i] = (sign[i] ? 32'b00000000000000000000000000000000 : out_results[(3 - (position + i)) * 32+:32]);
		end
	endgenerate
	always @(*)
		if (compression_en)
			outs = {temp_results[0][15:0], temp_results[1][15:0]};
		else
			outs = temp_results[0];
endmodule
