# FitGuard 最终演示操作

## 当前可演示版本

打开：

`http://<PYNQ_IP>:9090/notebooks/FitGuard_Unified_Test/FitGuard_Live_Demo.ipynb`

运行唯一代码单元格：

1. 前2秒让 MPU6050 保持静止，完成陀螺仪零偏校准；
2. 页面左侧显示 PL 输出的二值形态学掩膜；
3. 页面右侧显示摄像头画面、OpenCV轮廓、包围框和中心位置；
4. 顶部显示三分类结果和置信度；
5. 基于最近2.56秒的 `128×6` IMU与 `64×7` 视觉窗口约每0.32秒刷新平滑结果；
6. 运行结束后预测记录保存在 `live_predictions.json`。

## 演示时的说明

- PL完成灰度、阈值、3×3开运算和3×3闭运算；
- PS端OpenCV只做轮廓与位置特征；
- MPU6050通过AXI IIC以约50 Hz采样；
- 三模型分别是IMU CNN、Vision CNN和双分支Fusion CNN；
- 当前模型使用W01至W05共45个真实试次训练，并完成5折留一受试者评价；扩增数据只进入训练折。
- 融合模型真实窗口Accuracy为71.72%、试次Accuracy为80.00%；这是工程原型结果，不代表临床准确率。

## 故障处理

- `Could not send I2C data`：SDA/SCL/3.3V/GND接触不良，先断电重新固定短线；
- 摄像头打不开：关闭其他占用 `/dev/video0` 的Notebook内核；
- 已有Kernel运行：先停止旧演示，再运行新单元格；不要同时运行采集和演示；
- 页面不更新：等待2.56秒首个完整窗口建立。

## 正式提交隔离

旧模拟演示仍隔离在 `FitGuard_Synthetic_Only/`。正式提交使用 `FitGuard_Unified_Test/` 中的W开头真实数据、`trained_models_waist/`、正式Notebook和真实受试者评价报告。
