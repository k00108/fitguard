"""Install only the new CNN-PL files after this package is uploaded to PYNQ."""

import argparse
from pathlib import Path
import shutil


PACKAGE = Path(__file__).resolve().parent / "deploy"


UNIFIED_FILES = (
    "fitguard_unified_cnn_pl.bit",
    "fitguard_unified_cnn_pl.hwh",
    "fitguard_cnn_pl_backend.py",
    "fitguard_motion_m01_m06_cnn_runtime_pl.py",
    "fitguard_cnn_pl_selftest.py",
    "fitguard_cnn_pl_golden.npz",
    "FitGuard_CNN_PL_Selftest.ipynb",
)

GATEWAY_FILES = (
    "fitguard_live_classifier_gateway_cnn_pl.py",
    "FitGuard_Wireless_Live_Demo_CNN_PL.ipynb",
)


def install(root: Path, check_only: bool = False) -> None:
    unified = root / "FitGuard_Unified_Test"
    gateway = root / "FitGuard_Wireless_Gateway_Deploy"
    if not unified.is_dir() or not gateway.is_dir():
        raise SystemExit(
            f"Expected FitGuard_Unified_Test and FitGuard_Wireless_Gateway_Deploy under {root}"
        )

    plan = [
        *((PACKAGE / "FitGuard_Unified_Test" / name, unified / name) for name in UNIFIED_FILES),
        *((PACKAGE / "FitGuard_Wireless_Gateway_Deploy" / name, gateway / name) for name in GATEWAY_FILES),
    ]
    missing = [str(source) for source, _ in plan if not source.is_file()]
    conflicts = [str(destination) for _, destination in plan if destination.exists()]
    if missing:
        raise FileNotFoundError("Package is incomplete:\n" + "\n".join(missing))
    if conflicts:
        raise FileExistsError(
            "Refusing the whole install before copying anything; these CNN-PL files already exist:\n"
            + "\n".join(conflicts)
        )
    print(f"preflight passed: {len(plan)} new files, no stable file will be replaced")
    if check_only:
        return

    for source, destination in plan:
        temporary = destination.with_name(destination.name + ".installing")
        shutil.copy2(source, temporary)
        temporary.rename(destination)
        print(f"installed: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/home/xilinx/jupyter_notebooks"),
        help="Jupyter notebooks root (default: /home/xilinx/jupyter_notebooks)",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    install(args.root.resolve(), check_only=args.check_only)
    if args.check_only:
        print("Check only: no files copied.")
    else:
        print("CNN-PL add-on installed. Stable .bit, CNN V7 and random-forest files were not changed.")
        print("Next: run FitGuard_Unified_Test/FitGuard_CNN_PL_Selftest.ipynb")


if __name__ == "__main__":
    main()
