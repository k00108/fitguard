# 比赛要求逐项对应

| 比赛要求 | 本包位置 | 当前结论 |
|---|---|---|
| GitHub/Gitee 开源与开源许可 | `LICENSE`、`.gitignore`、完整目录 | 本地已准备，尚未按用户要求上传；公开前需确认署名 |
| 完整工程文件与源码 | `hardware/`、`software/`、`models/` | 已覆盖 HLS、Vivado、PS 运行、模型和参考 ESP32 固件 |
| 仿真或验证结果 | `verification/hls/`、`verification/contracts/`、`verification/board_selftest/` | HLS、静态接口、回退契约、模拟 DMA 和实板金样均有证据 |
| 资源、时钟和关键性能 | `verification/implementation/`、`docs/PERFORMANCE_AND_RESOURCE_REPORT.md` | 已包含布局布线报告和指标摘要 |
| 上板工程、运行脚本和对比数据 | `hardware/bitstream/`、`software/pynq/`、`scripts/`、`verification/` | 已提供 bit/hwh/xsa、Notebook、金样与实时记录 |
| 可复现构建脚本 | `hardware/hls/run_hls.tcl`、`hardware/vivado/rebuild_final_project.tcl` | 已提供；需 Vivado/Vitis HLS 2025.2 与相应许可 |
| 与基线的对比 | `models/random_forest_baseline/`、CNN 消融与验证结果 | 随机森林作为可回退/对照基线保留 |
| 设计判断写入报告 | `docs/`、`FINAL_ENGINEERING_REPORT.md` | 已解释双模态、PL/PS 边界、量化和故障处理 |
| LLM 协作经验技能包 | `skill/` | 已包含通用 PYNQ 发布审计流程、校验脚本与排错清单 |
| 合理使用片上资源 | 实现报告 | BRAM 78.57% 为最紧资源，未以全 CNN 上 PL 换取有限收益 |

## 仍需团队补充的证据

1. 报告中的最终实物连接图与清晰界面截图。
2. 同一测试动作下 IMU-only、Vision-only、Dual-modal 的对照表或消融图。
3. 连接全部硬件后的最终一次完整验收记录。
4. 若公开训练过程，应提供获得授权的匿名数据或采集脚本生成的同格式数据；本包不公开原始人物数据。
