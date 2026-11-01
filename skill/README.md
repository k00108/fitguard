# PYNQ 工程发布审计技能包

## 适用范围

用于 PYNQ/FPGA 项目在比赛或开源前的只读盘点、PL 运行证据核验、资源/时序/性能汇总、隐私清理、复现材料整理与发布包校验。它不负责修改实时算法，也不替代 Vivado/Vitis 的实际构建。

## 使用方法

1. 准备板端只读快照和电脑端 HLS/Vivado 工程。
2. 按 `SKILL.md` 的八步工作流整理目录。
3. 阅读 `references/workflow.md`、`evidence-schema.md` 和 `troubleshooting.md`。
4. 运行：`python scripts/verify_artifacts.py <release-root>`。
5. 校验通过后再生成 ZIP 和 SHA256，并由队员进行全硬件最终验收。

## 已验证效果

该流程用于 FitGuard CNN-PL 工程，成功核对板端和电脑端 bit/hwh、18 组金样、自检误差、实时 PL 调用、实现资源和时序，并在不修改板端的前提下形成公开包。

## 失效条件与经验

- 只看到 `.bit` 文件不能证明加速器被实时程序调用；必须读后端状态和调用计数。
- 只保留生成工程、没有 HLS/Vivado 源与重建说明，不满足从零复现。
- 把原始人体视频直接打包会制造不必要的隐私风险。
- 冷启动时间、PL 核延迟与整机端到端延迟必须分开报告。
- 自动回退提高可用性，但不能把回退运行写成 FPGA 加速结果。
