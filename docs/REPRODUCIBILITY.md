# 复现与验收指南

## 1. 环境

- PYNQ-Z2，PYNQ Linux 镜像；
- USB 摄像头（项目实测为 Logitech C270）；
- MPU6050 腰部节点、ESP32 无线节点和 USB 网关；
- Vivado/Vitis HLS 2025.2（重新构建硬件时）；
- Python 3、NumPy、OpenCV、ipywidgets、PYNQ Python API。

## 2. 不重新综合，直接上板

1. 保留板上稳定版本；将 `software/pynq/FitGuard_Unified_Test` 与 `software/pynq/FitGuard_Wireless_Gateway_Deploy` 复制到 Jupyter 工作目录。
2. 先运行 `FitGuard_CNN_PL_Selftest.ipynb`。通过标准为 `passed=true`、18 个样本全部完成、最大误差不超过 0.001。
3. 插入摄像头和 USB 网关，确认 `/dev/video0` 与 `/dev/ttyUSB0` 存在。
4. 打开 `FitGuard_Wireless_Live_Demo_CNN_PL.ipynb`，运行主单元格。
5. 启动信息必须显示 `pl_cnn_conv1d`；如果显示 `numpy_ps_fallback`，先依据回退原因检查 bit/hwh、DMA 和内存分配。
6. 等待 64 帧时间窗填充后，界面从 `WARMING UP` 进入状态显示。演示结束时手动停止单元格，程序应安全释放摄像头、串口和 DMA 缓冲。

## 3. HLS 重建

在 Vitis HLS 命令环境中：

```text
vitis-run --mode hls --tcl hardware/hls/run_hls.tcl
```

脚本运行 C 仿真、综合并导出 IP。将生成报告与 `verification/hls/` 对比。

## 4. Vivado 重建

在 Vivado Tcl Shell 中：

```text
vivado -mode batch -source hardware/vivado/rebuild_final_project.tcl
```

脚本打开包内最终工程，重新设置包内 IP 仓库，验证 Block Design，执行综合与实现，并将新 bit/hwh 输出到 `hardware/vivado/rebuilt_output/`。构建前不要覆盖 `hardware/bitstream/` 中的黄金产物。

## 5. 发布包校验

```text
python scripts/verify_release.py --root .
```

校验器检查关键文件、禁止项、JSON 证据和 SHA256 清单。发布前还应在另一台电脑解压后重跑一次。

## 6. 常见失败

- `Cannot open the V4L2 camera`：摄像头未识别、被其他进程占用或设备编号不是 `/dev/video0`。
- `LINK_FAULT`：USB 网关无数据、ESP32 节点未发送、串口被占用或数据过旧。
- `DATA_INSUFFICIENT`：人物丢失、目标框质量低或视觉有效比例不足；恢复入框后应自动解除。
- 第二次运行卡住：先手动停止上次单元格并等待资源释放；必要时重启 Notebook 内核，已保存的数据文件不会因此丢失。
- 后端回退：检查 bit/hwh 同名、`axi_dma_cnn_0` 存在、连续内存可分配，并先运行自检。

## 7. 复现边界

本包能复现硬件构建、板端推理、数值验证和实时链路。由于隐私原因不附带原始人物视频和完整训练数据；要完全复现训练，需要使用采集 Notebook 产生获得授权的同格式数据。
