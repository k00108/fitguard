# FitGuard 三模型训练

正式数据由板端 `FitGuard_Dual_Collector.ipynb` 写入 `dataset/`。复制整个目录到训练电脑后运行：

```powershell
python -m pip install tensorflow scikit-learn numpy
python fitguard_train_models.py --dataset dataset --output trained_models --epochs 80
```

脚本会训练并比较三套 1D CNN：

- `fitguard_imu`：输入 `128×6`；
- `fitguard_vision`：输入 `64×7`；
- `fitguard_fusion`：双分支特征融合。

划分优先按受试者隔离；受试者不足三人时临时按完整试次隔离，重叠窗口绝不会跨训练集和测试集。输出包括 `.keras`、动态量化 `.tflite`、归一化参数、混淆矩阵和 `evaluation.json`。只有真实正式数据才会用于训练，`SMOKE` 测试样本会自动排除。
