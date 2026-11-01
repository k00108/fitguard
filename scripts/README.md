# 脚本说明

- `verify_release.py`：公开包最终校验与 SHA256 清单生成，提交前优先运行。
- `sanitize_notebooks.py`：清除 Notebook 的执行编号、输出和历史报错，仅作用于指定本地目录。
- `generate_artifacts.py`：从包内 CNN 模型重新生成 HLS 权重、金样和清单。
- `verify_hardware_contract.py`：核对 bit/hwh 的静态接口与哈希。
- `install_to_pynq.py`：早期安全安装器，使用前先执行 `--check-only`；当前包也可按文档手动复制。
- `run_board_selftest.py`、`run_board_runtime_contract.py`、`run_live_pl_acceptance.py`：在 PYNQ 固定目录中使用的板端验收脚本。
- `legacy_contracts/`：CNN-PL 开发阶段的历史契约验证脚本，保留用于审计；其中部分脚本记录原始工程布局，不作为公开包的一键入口。

比赛提交和公开仓库的统一入口是：

```text
python scripts/verify_release.py --root .
```
