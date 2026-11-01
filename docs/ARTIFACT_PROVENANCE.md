# 产物来源与一致性

本工程包通过 PYNQ Jupyter 的只读 GET 接口获取板端快照。整理过程没有向板端调用写入、删除、重命名或执行接口。

## 黄金产物哈希

| 产物 | SHA256 |
|---|---|
| `fitguard_unified_cnn_pl.bit` | `C31B8FC0D861054B7657A1D63747E90AD8AFB3299BE315F1BBC19C95EBB96A47` |
| `fitguard_unified_cnn_pl.hwh` | `AEBB8011EBC7878B9D18695B0D19CA016FEBC0EFE3A90CF0617B5539DE402231` |
| `fitguard_cnn_pl_golden.npz` | `9B6B9EDB5AC569C943C7722993B19DDCDF7D9BD2916E4F2C5FB304DD8768751B` |

bit 与 hwh 已在三个位置逐字节一致：

1. 电脑端 Vivado `deliverables/` 黄金产物；
2. PYNQ 板端只读快照；
3. 本比赛工程包 `hardware/bitstream/`。

因此板端自检与实时记录对应同一硬件版本，而不是不同构建产物的混合。

## 板端证据

- 自检：18 个金样，最大误差 `0.0001220703125`，阈值 `0.001`，通过。
- 长时实时：`cnn_v6_run_0011` 连续运行 `817.591307483 s`，记录 737 次分类、40400 个 IMU 样本，网关接收率约 99.881%，三类最终状态均有输出。
- PL 实时：`cnn_v6_run_0021` 的实际后端为 `pl_cnn_conv1d`，`pl_acceleration_active=true`，PL 调用 775 次。

两份记录分别证明长时间双模态分类/链路表现与 PL 加速实际启用，不将不同实验的指标混写为同一次运行。长时记录从板端只读快照复制，原始文件 SHA256 为 `41E056017CE11198414C1D9D0117DDB6480B28C92BC8C64C4D62DAF58EBFA2E2`；公开包仅机械移除了未启用且与正式方案无关的旧字段，运行数据未改动。

## 发布清单

`SHA256SUMS.txt` 由 `scripts/verify_release.py --write-manifest` 生成，覆盖工程包内所有文件（清单自身除外）。每次修改公开包后都必须重新生成。
