# FitGuard CNN-PL 工程结果

## 已完成

- 原稳定工程、随机森林、CNN V6/V7 和原 `fitguard_unified.bit` 均未覆盖。
- 建立独立 Vivado 工程 `fitguard_unified_cnn_pl`。
- 将 IMU 与视觉两个分支共六层 Conv1D、ReLU、池化和 GAP 放入 PL。
- 新增独立 AXI DMA `axi_dma_cnn_0`，控制地址 `0x40410000`；原视觉 DMA 保持不变。
- PS 保留规范化、范围特征、Dense 融合、softmax、转身保护、防抖、传感器失效和异常事件闭环。
- 新增 `auto / pl / numpy` 三种后端；`auto` 的 PL 失败会自动回退原 NumPy 路径并记录原因。
- 实时入口启动时显式预热 CNN-PL DMA，把一次性冷启动开销移到正式分类前，并立即给出后端激活或回退状态。
- 实时程序退出时显式关闭 CNN 后端并释放 DMA 连续内存，支持 Notebook 安全重复运行。
- 新增无限时实时演示 Notebook 和无需摄像头/IMU 的板端金样自检 Notebook。

## 数值验证

- M01–M06：56 次试验、1232 个窗口；Q18.13 定点模型与 V6 最终类别一致率 100%。
- HLS C 仿真：18/18 金样通过，最大 GAP 特征绝对误差 `0.000122070312`。
- PS 后半段接口回归：18/18 类别一致，最大概率误差 `1.0692e-08`。
- 软件回退契约：18/18 原始金样类别一致、概率误差 0；`auto` 正确记录原因并切换为 `numpy_ps_fallback`，严格 `pl` 模式在无硬件时明确失败。
- bit/hwh 静态契约：CNN DMA 与六层卷积 IP 均存在，DMA 地址为 `0x40410000`，双向 AXI Stream 均为 32 bit 且连接关系完整。
- 模拟 PYNQ/DMA 契约：18/18 金样精确返回，验证接收先于发送、缓存同步、DMA 等待、计数与释放流程。
- 实际 PYNQ/DMA 金样：最终纯 CNN-PL 版本重新执行，18/18 通过，最大绝对误差 `0.0001220703125`；首次 DMA 冷启动约 `12.09 s`，随后稳态均值 `33.22 ms`、P50 `33.18 ms`、P95 `33.43 ms`、最大 `33.62 ms`。本报告统一采用最终板端 JSON 的 18 次结果。
- 板端实时入口契约：导入、自动 PL 后端、无限时手动停止、传感器失效阈值、状态平滑及依赖完整性全部通过。
- 安装器契约：9 个新增文件整体预检和安装通过；重复安装会在复制前拒绝，现有文件哈希保持不变，且不会创建或覆盖稳定版 `fitguard_unified.bit`。

## 加速器与实现结果

- 定点格式：`ap_fixed<18,5,AP_RND,AP_SAT>`。
- 四路独立乘加累加；HLS 最坏延迟 1,189,731 周期，即 23.795 ms（50 MHz）。
- HLS 估计 Fmax 69.38 MHz。
- Vivado 布局布线后 WNS `+3.586 ns`，所有时序约束满足。
- 实际全系统资源：LUT 15,431/53,200（29.01%）；FF 11,581/106,400（10.88%）；BRAM 110/140（78.57%）；DSP 48/220（21.82%）。

## 实板最终验收

- 2026-10-17 实时入口记录 `cnn_v6_run_0004.json` 已确认 `backend.active=pl_cnn_conv1d`、`pl_acceleration_active=true`，`axi_dma_cnn_0` 实际执行 1 次，本次 DMA 为 33.47 ms，未发生回退。
- 同次运行摄像头设备和无线 IMU 链路均成功打开，网关 `WHO_AM_I=0x68`，发送约 50 Hz，接收率 1.0，全部链路错误计数为 0。
- 当时画面没有可靠人物，因此没有生成分类窗口；六层 Conv1D 的输出正确性仍由最终 bitstream 的 18/18 DMA 金样、最大误差 `0.0001220703125` 证明。

结论：独立 CNN-PL 版本的九个计划阶段均已完成。实时演示默认使用 `auto`，正常显示 `pl_cnn_conv1d`；若未来出现 `numpy_ps_fallback`，系统仍能工作，但该次运行不能计作 FPGA 加速演示。

## 同口径 PS/PL 对照说明

新增 `software/pynq/FitGuard_Unified_Test/fitguard_cnn_ps_pl_benchmark.py`，在同一块 PYNQ-Z2 上读取同一批 18 个归一化金样，分别执行相同边界的六层 Conv1D、ReLU、池化与 GAP：PS 侧使用 NumPy，PL 侧使用最终 bit/hwh 与 AXI DMA。脚本输出每侧均值、P50、P95、最大误差及加速比，但不修改现有模型、位流或实时入口。

当前证据包没有完成这一同板同口径测试，因此不把不同机器上的 NumPy 时间与板端 PL 时间相除，也不虚报加速比。评审前在板端执行该脚本即可生成 `fitguard_cnn_ps_pl_benchmark.json`；只有该文件通过误差阈值后，才引用其中的 speedup。
