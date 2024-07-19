module prim_double_clock_gating (
  input  clk_i,
  input  en_i,
  input  test_en_i,
  output clk_o1,    // Original clock output (gated)
  output clk_o2 // Simulated double frequency clock output
);

  reg en_latch;
  reg clk_double = 0; // Used to toggle the double frequency clock

  // Latch enable signal on the falling edge of clk_i
  always @(negedge clk_i) begin
    en_latch <= en_i | test_en_i;
  end

  // Toggle clk_double every time clk_i goes high, effectively simulating double the toggling rate of clk_i
  always @(posedge clk_i) begin
    if (en_latch) begin // Only toggle when enabled
      clk_double <= ~clk_double;
    end
  end

  assign clk_o1 = en_latch & clk_i;  // Gated clock output
  
  // For simulation purposes, clk_double_o is toggled at every positive edge of clk_i when enabled
  // This simulates a "double frequency" effect within the constraints of a simulation environment
  assign clk_o2 = clk_double & en_latch;

endmodule

