module ibex_register_file_latch (
	clk_i,
	rst_ni,
	test_en_i,
	dummy_instr_id_i,
	dummy_instr_wb_i,
	raddr_a_i,
	rdata_a_o,
	raddr_b_i,
	rdata_b_o,
	waddr_a_i,
	wdata_a_i,
	we_a_i,
	err_o
);
	parameter [0:0] RV32E = 0;
	parameter [31:0] DataWidth = 32;
	parameter [0:0] DummyInstructions = 0;
	parameter [0:0] WrenCheck = 0;
	parameter [DataWidth - 1:0] WordZeroVal = 1'sb0;
	input wire clk_i;
	input wire rst_ni;
	input wire test_en_i;
	input wire dummy_instr_id_i;
	input wire dummy_instr_wb_i;
	input wire [4:0] raddr_a_i;
	output wire [DataWidth - 1:0] rdata_a_o;
	input wire [4:0] raddr_b_i;
	output wire [DataWidth - 1:0] rdata_b_o;
	input wire [4:0] waddr_a_i;
	input wire [DataWidth - 1:0] wdata_a_i;
	input wire we_a_i;
	output wire err_o;
	localparam [31:0] ADDR_WIDTH = (RV32E ? 4 : 5);
	localparam [31:0] NUM_WORDS = 2 ** ADDR_WIDTH;
	reg [DataWidth - 1:0] mem [0:NUM_WORDS - 1];
	reg [NUM_WORDS - 1:0] waddr_onehot_a;
	wire [NUM_WORDS - 1:1] mem_clocks;
	reg [DataWidth - 1:0] wdata_a_q;
	wire [ADDR_WIDTH - 1:0] raddr_a_int;
	wire [ADDR_WIDTH - 1:0] raddr_b_int;
	wire [ADDR_WIDTH - 1:0] waddr_a_int;
	assign raddr_a_int = raddr_a_i[ADDR_WIDTH - 1:0];
	assign raddr_b_int = raddr_b_i[ADDR_WIDTH - 1:0];
	assign waddr_a_int = waddr_a_i[ADDR_WIDTH - 1:0];
	wire clk_int;
	assign rdata_a_o = mem[raddr_a_int];
	assign rdata_b_o = mem[raddr_b_int];
	prim_clock_gating cg_we_global(
		.clk_i(clk_i),
		.en_i(we_a_i),
		.test_en_i(test_en_i),
		.clk_o(clk_int)
	);
	always @(posedge clk_int or negedge rst_ni) begin : sample_wdata
		if (!rst_ni)
			wdata_a_q <= WordZeroVal;
		else if (we_a_i)
			wdata_a_q <= wdata_a_i;
	end
	function automatic signed [4:0] sv2v_cast_5_signed;
		input reg signed [4:0] inp;
		sv2v_cast_5_signed = inp;
	endfunction
	always @(*) begin : wad
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < NUM_WORDS; i = i + 1)
				begin : wad_word_iter
					if (we_a_i && (waddr_a_int == sv2v_cast_5_signed(i)))
						waddr_onehot_a[i] = 1'b1;
					else
						waddr_onehot_a[i] = 1'b0;
				end
		end
	end
	generate
		if (WrenCheck) begin : gen_wren_check
			wire [NUM_WORDS - 1:0] waddr_onehot_a_buf;
			prim_generic_buf #(.Width(NUM_WORDS)) u_prim_generic_buf(
				.in_i(waddr_onehot_a),
				.out_o(waddr_onehot_a_buf)
			);
			prim_onehot_check #(
				.AddrWidth(ADDR_WIDTH),
				.AddrCheck(1),
				.EnableCheck(1)
			) u_prim_onehot_check(
				.clk_i(clk_i),
				.rst_ni(rst_ni),
				.oh_i(waddr_onehot_a_buf),
				.addr_i(waddr_a_i),
				.en_i(we_a_i),
				.err_o(err_o)
			);
		end
		else begin : gen_no_wren_check
			wire unused_strobe;
			assign unused_strobe = waddr_onehot_a[0];
			assign err_o = 1'b0;
		end
	endgenerate
	genvar x;
	generate
		for (x = 1; x < NUM_WORDS; x = x + 1) begin : gen_cg_word_iter
			prim_clock_gating cg_i(
				.clk_i(clk_int),
				.en_i(waddr_onehot_a[x]),
				.test_en_i(test_en_i),
				.clk_o(mem_clocks[x])
			);
		end
	endgenerate
	genvar i;
	generate
		for (i = 1; i < NUM_WORDS; i = i + 1) begin : g_rf_latches
			always @(*)
				if (mem_clocks[i])
					mem[i] = wdata_a_q;
		end
		if (DummyInstructions) begin : g_dummy_r0
			wire we_r0_dummy;
			wire r0_clock;
			reg [DataWidth - 1:0] mem_r0;
			assign we_r0_dummy = we_a_i & dummy_instr_wb_i;
			prim_clock_gating cg_i(
				.clk_i(clk_int),
				.en_i(we_r0_dummy),
				.test_en_i(test_en_i),
				.clk_o(r0_clock)
			);
			always @(*) begin : latch_wdata
				if (r0_clock)
					mem_r0 = wdata_a_q;
			end
			wire [DataWidth:1] sv2v_tmp_79BDA;
			assign sv2v_tmp_79BDA = (dummy_instr_id_i ? mem_r0 : WordZeroVal);
			always @(*) mem[0] = sv2v_tmp_79BDA;
		end
		else begin : g_normal_r0
			wire unused_dummy_instr;
			assign unused_dummy_instr = dummy_instr_id_i ^ dummy_instr_wb_i;
			wire [DataWidth:1] sv2v_tmp_5D6C3;
			assign sv2v_tmp_5D6C3 = WordZeroVal;
			always @(*) mem[0] = sv2v_tmp_5D6C3;
		end
	endgenerate
endmodule
