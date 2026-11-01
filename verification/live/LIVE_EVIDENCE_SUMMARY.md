# 实时运行证据摘要

本目录保留两次独立的真实板端运行。两份记录承担不同的证明目的，不把不同实验的指标合并成一次运行。

## 长时分类与链路记录

来源：`live_predictions_gateway_long_run.json`，来自板端只读快照中的 `cnn_v6_run_0011.json`。

- `run_id`: `cnn_v6_run_0011`
- 运行时长：`817.591307483 s`
- 停止原因：`manual_interrupt`
- 分类记录：737，其中 736 次形成最终标签
- 最终标签：`Stable` 422、`Mild Abnormality` 225、`Marked Abnormality` 89
- IMU 样本：40400
- 网关接收率：`0.9988132911392406`（约 99.881%）
- CRC 错误：1
- 丢样：48
- 乱序/队列丢弃：0/0

该记录对应报告 3.3.5 的长时间运行、三类输出与无线链路数据。原始板端文件 SHA256 为 `41E056017CE11198414C1D9D0117DDB6480B28C92BC8C64C4D62DAF58EBFA2E2`。公开包副本只机械移除了未启用、与正式方案无关的旧字段，其他运行数据保持不变。

## PL 实际启用记录

来源：`live_predictions_gateway_cnn_pl.json`，来自板端只读复制。

- `run_id`: `cnn_v6_run_0021`
- 主模型：`cnn_v8_pl_v6_weights`
- 请求后端：`auto`
- 实际后端：`pl_cnn_conv1d`
- PL 加速激活：`true`
- PL 调用：775
- 运行时长：约 686.98 秒
- 停止原因：`manual_interrupt`
- 预测记录：773

该记录证明真实实时入口持续调用 CNN-PL，而不是只做一次离线金样。

注意：板端系统日期不可靠，`started_board_time` 不能作为真实日历时间使用。
