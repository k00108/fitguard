# FitGuard 双模态步态评估终端

FitGuard 是面向 PYNQ-Z2 的步态稳定性评估原型。系统将腰部 MPU6050 惯性数据与摄像头人物运动信息进行时间窗级融合，输出 `Stable`（稳定）、`Mild Abnormality`（轻度异常）与 `Marked Abnormality`（重度异常）三类状态；当人物跟踪或无线 IMU 链路不可靠时，系统输出 `DATA_INSUFFICIENT` / `LINK_FAULT`，而不是用错误数据强行分类。

本仓库为 FitGuard 比赛工程包，包含可复现构建、板端运行和验证所需的工程文件；使用仓库内容不会自动对 PYNQ 板端现有文件执行写入、覆盖或删除操作。

## 项目亮点

- 双模态融合：50 Hz 腰部 IMU 与运动目标跟踪共同参与判断。
- PL 加速：六层 Conv1D、ReLU、池化和 GAP 在 PL 中执行；Dense 融合、状态机与异常闭环在 PS 中执行。
- 可回退：`auto` 后端优先使用 `pl_cnn_conv1d`，硬件不可用时回退 NumPy，并记录原因。
- 工程闭环：包含 HLS 源码、Vivado 工程源、bit/hwh/xsa、板端运行代码、模型、金样、自检、实现报告与实板运行证据。
- 稳定性保护：人物跟踪滞回、传感器失效检测、状态防抖、重度异常快速响应和手动停止后的资源释放。

## 系统边界

| 部分 | 主要职责 |
|---|---|
| PL | 既有视觉掩膜流水线；CNN 双分支共六层 Conv1D、ReLU、池化、GAP；AXI DMA 数据搬运 |
| PS | 摄像头/串口采集、目标跟踪、输入规范化、Dense 融合、softmax、转身保护、状态防抖、故障状态、界面与记录 |
| ESP32/MPU6050 | 采集腰部六轴数据并通过无线链路和 USB 网关送入 PYNQ |

## 最短验收路径

1. 将 `software/pynq/` 下的两个目录放到 PYNQ 的 `/home/xilinx/jupyter_notebooks/`。
2. 不接摄像头和 IMU 时，先打开 `FitGuard_Unified_Test/FitGuard_CNN_PL_Selftest.ipynb`。要求：18/18 金样通过，`max_abs_error <= 0.001`。
3. 接好摄像头、腰部传感器和 ESP32 网关后，打开 `FitGuard_Wireless_Gateway_Deploy/FitGuard_Wireless_Live_Demo_CNN_PL.ipynb`。
4. 运行主单元格，确认后端显示 `pl_cnn_conv1d`、`pl_acceleration_active=true`，并观察三类状态及链路异常恢复。
5. 演示持续到手动停止单元格；停止后检查独立运行记录。
6. 需要同输入 PS/PL 对照时，在 `FitGuard_Unified_Test/` 运行 `python3 fitguard_cnn_ps_pl_benchmark.py --model-dir trained_models_motion_v2_cnn_v6_candidate --golden fitguard_cnn_pl_golden.npz --bitstream fitguard_unified_cnn_pl.bit --report fitguard_cnn_ps_pl_benchmark.json --repeats 10`。该脚本独立运行，不改动现有工程。

详细操作见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

比赛主报告为 `docs/fitguard_technical_report.docx`，`docs/fitguard_technical_report.pdf` 作为便于在线查看的预览版；双语海报见 `docs/fitguard_bilingual_poster.pptx`。

## 可复现构建

- HLS：`hardware/hls/run_hls.tcl`
- Vivado 最终工程重建：`hardware/vivado/rebuild_final_project.tcl`
- 已生成产物：`hardware/bitstream/`
- 板端部署代码：`software/pynq/`
- 校验：在包根目录运行 `python scripts/verify_release.py --root .`

Vivado/Vitis HLS 建议版本为 2025.2，器件为 `xc7z020clg400-1`。构建工具和 PYNQ 镜像本身不包含在本仓库中。

## 已验证结果

- M01–M06：56 次试验、1232 个窗口；Q18.13 与浮点 CNN 最终类别一致率 100%。
- HLS/板端金样：18/18 通过，最大 GAP 特征绝对误差 `0.0001220703125`。
- 板端稳态 CNN-PL DMA：均值 `33.22 ms`，P95 `33.43 ms`。
- 实现后时序：WNS `+3.586 ns`。
- 资源：LUT 29.01%、FF 10.88%、BRAM 78.57%、DSP 21.82%。
- 长时实时记录：连续运行约 817.6 秒，737 次分类记录、40400 个 IMU 样本，网关接收率约 99.881%，CRC 错误 1 次、丢样 48 个；三类最终状态均有输出。
- PL 实时记录：`pl_cnn_conv1d` 实际启用，`pl_acceleration_active=true`，PL 调用 775 次。长时链路统计与 PL 启用证据来自两次独立实测，分别保留原始记录。
- PL 实时记录中的完整模型推理均值约 `78.95 ms`、P95 约 `85.39 ms`；相邻分类更新时间 P50 约 `336 ms`、P95 约 `469 ms`。后者不是传感器到显示的端到端延迟。
- 同输入、同计算边界的 PS/PL 基准脚本已随包提供；在同板测试完成前不报告加速比，避免跨机器、跨边界比较。

完整证据见 `verification/` 与 [docs/PERFORMANCE_AND_RESOURCE_REPORT.md](docs/PERFORMANCE_AND_RESOURCE_REPORT.md)。

## 数据与隐私

工程包不包含原始人物视频、完整人体实测数据集或直接个人身份信息。报告与海报保留系统演示截图，公开使用前应确认相关授权，并确保截图不包含可识别个人身份的信息。公开包保留模型权重、匿名金样和验证汇总，足以复现推理、自检与硬件数值验证；重新训练需按文档采集或接入同格式的授权数据。详见 [docs/DATA_AND_PRIVACY.md](docs/DATA_AND_PRIVACY.md)。

## 目录

- `hardware/`：HLS、Vivado 源工程与 bit/hwh/xsa。
- `software/`：PYNQ 实时程序、采集程序与 ESP32 参考固件。
- `models/`：CNN 主模型与保留的随机森林基线。
- `verification/`：数值、HLS、实现、实板与实时链路证据。
- `docs/`：最新版比赛主报告、双语海报，以及架构、性能、复现、数据与提交要求对照。
- `media/`：去除容器元数据后的匿名演示视频。
- `scripts/`：构建/契约/发布校验脚本。
- `skill/`：可复用的 PYNQ 工程验收与发布审计技能包。

## 许可

原创源代码以 MIT License 发布，见 [LICENSE](LICENSE)。AMD/Xilinx 工具、PYNQ 系统、第三方库和生成 IP 仍受各自许可约束；原始实测数据不随本包授权。
