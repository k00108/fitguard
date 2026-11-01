from pathlib import Path
import subprocess
import sys

base = Path("/home/xilinx/jupyter_notebooks/FitGuard_Unified_Test")
cmd = [
    sys.executable,
    str(base / "fitguard_cnn_pl_selftest.py"),
    "--bitstream",
    str(base / "fitguard_unified_cnn_pl.bit"),
    "--golden",
    str(base / "fitguard_cnn_pl_golden.npz"),
    "--report",
    str(base / "fitguard_cnn_pl_board_selftest.json"),
]
completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(completed.stdout)
if completed.returncode:
    raise RuntimeError(f"CNN-PL self-test failed with code {completed.returncode}")
