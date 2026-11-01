# 常见问题与排查

## 后端显示 NumPy 回退

检查 bit/hwh 是否同名、DMA 是否存在、地址是否匹配、PYNQ 连续内存是否释放。先跑金样自检，再跑实时入口。

## 自检通过但实时没有分类

检查时间窗是否填满、人物跟踪有效率、摄像头设备、IMU 新鲜度和目标框质量。`WARMING UP` 与 `DATA_INSUFFICIENT` 不等于模型失败。

## 第二次 Notebook 卡住

确认第一次已手动停止并进入资源释放；摄像头、串口和 DMA 不能同时被两个内核占用。必要时只重启内核，不删除已保存数据。

## 构建只能在原电脑运行

搜索盘符、用户名和绝对路径。把工程根目录改为相对脚本路径，把外部 IP 放进 `ip_repo/`，并在 Tcl 中重新设置 `ip_repo_paths`。

## 包过大或包含隐私

排除 `.runs/.cache/.Xil`、`__pycache__`、Notebook 输出、完整原始数据、事件截图和临时下载。保留源、报告、金样和必要生成产物。

## 资源报告与报告文字不一致

以最终 `post_route_utilization.rpt` 和 `post_route_timing_summary.rpt` 为准，禁止沿用早期 HLS 估算代替整机实现结果。
