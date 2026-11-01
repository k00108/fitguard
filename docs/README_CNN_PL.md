# FitGuard CNN V8 PL

这是与稳定版并存的独立加速版本，不覆盖 `fitguard_unified.bit`、CNN V7 或随机森林版本。

## 硬件边界

- PL：IMU 三层 Conv1D、视觉三层 Conv1D、ReLU、池化与全局平均池化。
- PS：输入规范化、两支 Dense、范围特征、融合 Dense、softmax、转身保护、防抖、异常事件闭环与界面。
- 数据接口：第二个 AXI DMA（`axi_dma_cnn_0`），输入 1472 个 float32，输出 160 个 float32。
- 定点格式：权重/激活 `ap_fixed<18,5,AP_RND,AP_SAT>`。

## 安全回退

`backend='auto'` 会优先使用 PL。加载失败、DMA 失败或返回异常时，当前预测自动切换到原 V7 NumPy 路径，并通过 `backend_status()` 暴露原因。`backend='numpy'` 可强制基线；`backend='pl'` 用于严格验收。

实时程序在手动停止或异常退出时会显式释放 CNN DMA 的连续内存缓冲区，降低 Notebook 第二次运行时因残留资源而卡住的风险。

离线回退契约已用 18 组原始金样验证：自动回退与 V7 NumPy 的类别完全一致、概率误差为 0；强制 `pl` 在硬件不可用时会明确拒绝运行，不会伪装成 PL 加速。

## 板端验收

1. 在没有摄像头和 IMU 时先执行 `fitguard_cnn_pl_selftest.py`，对 18 组金样验证 PL 数值和 DMA 延迟。
2. 实时入口会在分类前预热一次 CNN DMA；看到 `CNN PL backend ready: pl_cnn_conv1d (accelerated=True)` 表示本次演示正在使用 PL。`auto` 模式失败时会记录原因并回退 NumPy。
3. 独立 Web 页面默认关闭；Notebook 内的大图预览和全屏按钮保留。
4. 通过后打开 `FitGuard_Wireless_Live_Demo_CNN_PL.ipynb`；运行会持续到手动停止。
5. 演示面板和独立运行记录中的 `backend` 必须显示 `pl_cnn_conv1d`，才可以宣称 FPGA CNN 加速正在工作。

工程包同时包含 `fallback_contract_report.json` 与 `hardware_contract_report.json`，分别记录软件回退和 bit/hwh 静态接口核验结果。

安装时先运行 `python3 install_to_pynq.py --check-only`。安装器会一次性预检全部 9 个新增文件；只要目标中有一个同名文件，就会在复制开始前整体停止，因此不会覆盖稳定版，也不会造成半安装。
