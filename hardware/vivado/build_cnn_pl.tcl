set script_dir [file dirname [file normalize [info script]]]
set out_root [file normalize "$script_dir/build_from_base"]
if {[info exists ::env(FITGUARD_OUTPUT_ROOT)]} {
    set out_root [file normalize $::env(FITGUARD_OUTPUT_ROOT)]
}
if {![info exists ::env(FITGUARD_BASE_XPR)]} {
    error "Set FITGUARD_BASE_XPR to the stable fitguard_unified.xpr before using this integration script"
}
set base_xpr [file normalize $::env(FITGUARD_BASE_XPR)]
if {![file exists $base_xpr]} {
    error "Base Vivado project does not exist: $base_xpr"
}
set project_dir $out_root/vivado
set deliver_dir $out_root/deliverables
set report_dir $out_root/reports
file mkdir $deliver_dir
file mkdir $report_dir

open_project $base_xpr
save_project_as fitguard_unified_cnn_pl $project_dir -exclude_run_results -force

set_property ip_repo_paths [list \
    [file normalize "$script_dir/ip_repo/fitguard_cnn_conv"] \
    [file normalize "$script_dir/ip_repo/fitguard_vision"]] [current_project]
update_ip_catalog
open_bd_design [get_files */design_1.bd]

set_property CONFIG.NUM_MI {3} [get_bd_cells axi_smc]
set_property CONFIG.NUM_SI {4} [get_bd_cells axi_mem_intercon]

set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_cnn_0]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_m_axi_mm2s_data_width {32} \
    CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_mm2s_stream_data_width {32} \
    CONFIG.c_s2mm_stream_data_width {32}] $dma

set cnn [create_bd_cell -type ip -vlnv fitguard.org:hls:fitguard_cnn_conv:1.0 fitguard_cnn_conv_0]

connect_bd_intf_net [get_bd_intf_pins axi_dma_cnn_0/M_AXIS_MM2S] [get_bd_intf_pins fitguard_cnn_conv_0/input_r]
connect_bd_intf_net [get_bd_intf_pins fitguard_cnn_conv_0/output_r] [get_bd_intf_pins axi_dma_cnn_0/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M02_AXI] [get_bd_intf_pins axi_dma_cnn_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins axi_dma_cnn_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_cnn_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S03_AXI]

set clk [get_bd_pins processing_system7_0/FCLK_CLK0]
set rstn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn]
set icrstn [get_bd_pins rst_ps7_0_50M/interconnect_aresetn]
connect_bd_net $clk \
    [get_bd_pins axi_dma_cnn_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_cnn_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_cnn_0/m_axi_s2mm_aclk] \
    [get_bd_pins fitguard_cnn_conv_0/ap_clk] \
    [get_bd_pins axi_mem_intercon/S02_ACLK] \
    [get_bd_pins axi_mem_intercon/S03_ACLK]
connect_bd_net $rstn \
    [get_bd_pins axi_dma_cnn_0/axi_resetn] \
    [get_bd_pins fitguard_cnn_conv_0/ap_rst_n] \
    [get_bd_pins axi_mem_intercon/S02_ARESETN] \
    [get_bd_pins axi_mem_intercon/S03_ARESETN]

assign_bd_address [get_bd_addr_segs axi_dma_cnn_0/S_AXI_LITE/Reg]
set cnn_seg [get_bd_addr_segs -quiet processing_system7_0/Data/SEG_axi_dma_cnn_0_Reg]
if {[llength $cnn_seg] == 1} {
    set_property offset 0x40410000 $cnn_seg
    set_property range 64K $cnn_seg
}
assign_bd_address [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM]

validate_bd_design
save_bd_design
generate_target all [get_files */design_1.bd]

set wrapper [make_wrapper -files [get_files */design_1.bd] -top]
if {[llength $wrapper] > 0} { add_files -norecurse $wrapper }
update_compile_order -fileset sources_1

reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "Synthesis did not complete: [get_property STATUS [get_runs synth_1]]"
}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {![string match "write_bitstream Complete*" [get_property STATUS [get_runs impl_1]]]} {
    error "Implementation did not complete: [get_property STATUS [get_runs impl_1]]"
}

open_run impl_1
report_utilization -file $report_dir/post_route_utilization.rpt
report_timing_summary -file $report_dir/post_route_timing_summary.rpt
report_power -file $report_dir/post_route_power.rpt
write_hw_platform -fixed -include_bit -force $deliver_dir/fitguard_unified_cnn_pl.xsa

set bitfile [get_property DIRECTORY [get_runs impl_1]]/design_1_wrapper.bit
file copy -force $bitfile $deliver_dir/fitguard_unified_cnn_pl.bit
set hwh_candidates [glob -nocomplain $project_dir/fitguard_unified_cnn_pl.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh]
if {[llength $hwh_candidates] != 1} { error "Unable to locate generated HWH" }
file copy -force [lindex $hwh_candidates 0] $deliver_dir/fitguard_unified_cnn_pl.hwh

set fp [open $deliver_dir/build_manifest.txt w]
puts $fp "project=fitguard_unified_cnn_pl"
puts $fp "source_project=$base_xpr"
puts $fp "cnn_ip=fitguard.org:hls:fitguard_cnn_conv:1.0"
puts $fp "cnn_dma=axi_dma_cnn_0"
puts $fp "cnn_dma_address=0x40410000"
puts $fp "status=[get_property STATUS [get_runs impl_1]]"
close $fp
close_project
exit
