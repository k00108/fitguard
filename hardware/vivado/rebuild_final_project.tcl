set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize "$script_dir/project"]
set project_xpr [file normalize "$project_dir/fitguard_unified_cnn_pl.xpr"]
set output_dir [file normalize "$script_dir/rebuilt_output"]
set cnn_ip_repo [file normalize "$script_dir/ip_repo/fitguard_cnn_conv"]
set vision_ip_repo [file normalize "$script_dir/ip_repo/fitguard_vision"]

if {![file exists $project_xpr]} {
    error "Missing packaged Vivado project: $project_xpr"
}
file mkdir $output_dir

open_project $project_xpr
set_property ip_repo_paths [list $cnn_ip_repo $vision_ip_repo] [current_project]
update_ip_catalog

set bd_file [get_files -quiet */design_1.bd]
if {[llength $bd_file] != 1} {
    error "Expected exactly one design_1.bd, found [llength $bd_file]"
}
open_bd_design $bd_file
validate_bd_design
save_bd_design
generate_target all $bd_file

update_compile_order -fileset sources_1
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    error "Synthesis failed: [get_property STATUS [get_runs synth_1]]"
}

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "Implementation failed: [get_property STATUS [get_runs impl_1]]"
}

set bit_file [file normalize "$project_dir/fitguard_unified_cnn_pl.runs/impl_1/design_1_wrapper.bit"]
set hwh_file [file normalize "$project_dir/fitguard_unified_cnn_pl.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh"]
if {![file exists $bit_file]} { error "Bitstream not found: $bit_file" }
if {![file exists $hwh_file]} { error "HWH not found: $hwh_file" }

file copy -force $bit_file "$output_dir/fitguard_unified_cnn_pl.bit"
file copy -force $hwh_file "$output_dir/fitguard_unified_cnn_pl.hwh"

open_run impl_1
report_utilization -file "$output_dir/post_route_utilization.rpt"
report_timing_summary -file "$output_dir/post_route_timing_summary.rpt"
report_power -file "$output_dir/post_route_power.rpt"
puts "FitGuard rebuild completed: $output_dir"
close_project
