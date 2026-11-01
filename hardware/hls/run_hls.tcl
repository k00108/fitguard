set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize "$script_dir/hls_project"]
open_project -reset $project_dir
set_top fitguard_cnn_conv
file copy -force "$script_dir/fitguard_cnn_conv.cpp" "$project_dir/fitguard_cnn_conv.cpp"
cd $project_dir
add_files "fitguard_cnn_conv.cpp" -cflags "-I$script_dir/generated"
add_files -tb "$script_dir/fitguard_cnn_conv_tb.cpp" -cflags "-I$script_dir/generated"
open_solution -reset solution1 -flow_target vivado
set_part xc7z020clg400-1
create_clock -period 20 -name default
config_export -format ip_catalog -output "$script_dir/fitguard_cnn_conv_ip.zip" -rtl verilog -vendor fitguard.org -library hls -version 1.0
csim_design
csynth_design
export_design -format ip_catalog -rtl verilog
exit
