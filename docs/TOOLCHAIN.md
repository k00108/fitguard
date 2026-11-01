# 工具链与依赖

## 硬件构建

- Vivado 2025.2；
- Vitis HLS 2025.2；
- 目标器件 `xc7z020clg400-1`（PYNQ-Z2）；
- PL 时钟目标 50 MHz；
- HLS 定点类型 `ap_fixed<18,5,AP_RND,AP_SAT>`。

## 板端运行

- PYNQ Linux 与 `pynq` Python 包；
- Python 3；
- NumPy、OpenCV、ipywidgets；
- 模型再训练/随机森林脚本额外需要 joblib 与 scikit-learn。

板端镜像的精确版本尚未从现有只读快照中可靠提取，因此没有虚构版本号。最终提交前应在连接板子后补录 PYNQ 镜像版本、Python 版本和主要包版本，并将其写入报告。`software/requirements-board.txt` 只列依赖名称，不建议在已稳定的板子上盲目升级。

## 外设

- USB 摄像头；
- ESP32 被测者节点与 ESP32 USB 网关；
- MPU6050 六轴惯性传感器；
