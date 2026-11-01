
wire kernel_monitor_reset;
wire kernel_monitor_clock;
wire kernel_monitor_report;
assign kernel_monitor_reset = ~ap_rst_n;
assign kernel_monitor_clock = ap_clk;
assign kernel_monitor_report = 1'b0;
wire [1:0] axis_block_sigs;
wire [8:0] inst_idle_sigs;
wire [5:0] inst_block_sigs;
wire kernel_block;

assign axis_block_sigs[0] = ~yuyv_threshold_U0.in_stream_TDATA_blk_n;
assign axis_block_sigs[1] = ~pack_output_U0.out_stream_TDATA_blk_n;

assign inst_idle_sigs[0] = yuyv_threshold_U0.ap_idle;
assign inst_block_sigs[0] = (yuyv_threshold_U0.ap_done & ~yuyv_threshold_U0.ap_continue) | ~yuyv_threshold_U0.thresholded_blk_n;
assign inst_idle_sigs[1] = morph3x3_true_1_U0.ap_idle;
assign inst_block_sigs[1] = (morph3x3_true_1_U0.ap_done & ~morph3x3_true_1_U0.ap_continue) | ~morph3x3_true_1_U0.grp_morph3x3_true_1_Pipeline_row_loop_column_loop_fu_46.thresholded_blk_n | ~morph3x3_true_1_U0.grp_morph3x3_true_1_Pipeline_row_loop_column_loop_fu_46.eroded_open_blk_n;
assign inst_idle_sigs[2] = morph3x3_false_2_U0.ap_idle;
assign inst_block_sigs[2] = (morph3x3_false_2_U0.ap_done & ~morph3x3_false_2_U0.ap_continue) | ~morph3x3_false_2_U0.grp_morph3x3_false_2_Pipeline_row_loop_column_loop_fu_46.eroded_open_blk_n | ~morph3x3_false_2_U0.grp_morph3x3_false_2_Pipeline_row_loop_column_loop_fu_46.opened_blk_n;
assign inst_idle_sigs[3] = morph3x3_false_U0.ap_idle;
assign inst_block_sigs[3] = (morph3x3_false_U0.ap_done & ~morph3x3_false_U0.ap_continue) | ~morph3x3_false_U0.grp_morph3x3_false_Pipeline_row_loop_column_loop_fu_46.opened_blk_n | ~morph3x3_false_U0.grp_morph3x3_false_Pipeline_row_loop_column_loop_fu_46.dilated_close_blk_n;
assign inst_idle_sigs[4] = morph3x3_true_U0.ap_idle;
assign inst_block_sigs[4] = (morph3x3_true_U0.ap_done & ~morph3x3_true_U0.ap_continue) | ~morph3x3_true_U0.grp_morph3x3_true_Pipeline_row_loop_column_loop_fu_46.dilated_close_blk_n | ~morph3x3_true_U0.grp_morph3x3_true_Pipeline_row_loop_column_loop_fu_46.closed_blk_n;
assign inst_idle_sigs[5] = pack_output_U0.ap_idle;
assign inst_block_sigs[5] = (pack_output_U0.ap_done & ~pack_output_U0.ap_continue) | ~pack_output_U0.closed_blk_n;

assign inst_idle_sigs[6] = 1'b0;
assign inst_idle_sigs[7] = yuyv_threshold_U0.ap_idle;
assign inst_idle_sigs[8] = pack_output_U0.ap_idle;

fitguard_vision_hls_deadlock_idx0_monitor fitguard_vision_hls_deadlock_idx0_monitor_U (
    .clock(kernel_monitor_clock),
    .reset(kernel_monitor_reset),
    .axis_block_sigs(axis_block_sigs),
    .inst_idle_sigs(inst_idle_sigs),
    .inst_block_sigs(inst_block_sigs),
    .block(kernel_block)
);


always @ (kernel_block or kernel_monitor_reset) begin
    if (kernel_block == 1'b1 && kernel_monitor_reset == 1'b0) begin
        find_kernel_block = 1'b1;
    end
    else begin
        find_kernel_block = 1'b0;
    end
end
