# read_vcd_activities gcd
read_liberty /home/alex/Desktop/ibex_tools/synthesis/nangate45/lib/NangateOpenCellLibrary_typical.lib
read_verilog ibex_top_netlost.sta.v
link_design ibex_top

read_sdc ibex_top.nangate.out.sdc

# Generate vcd file
#  iverilog -o gcd_tb gcd_tb.v
#  vvp gcd_tb

read_power_activities -scope TOP/ibex_simple_system/u_top/u_ibex_top -vcd ../../../../sim.vcd
report_power
