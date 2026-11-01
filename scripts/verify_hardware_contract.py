#!/usr/bin/env python3
"""Static integrity and interface checks for the generated CNN-PL overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parameter(module: ET.Element, name: str) -> str | None:
    item = module.find(f"./PARAMETERS/PARAMETER[@NAME='{name}']")
    return None if item is None else item.attrib.get("VALUE")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bit", required=True, type=Path)
    parser.add_argument("--hwh", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    checks: dict[str, bool] = {
        "bit_exists_nonempty": args.bit.is_file() and args.bit.stat().st_size > 0,
        "hwh_exists_nonempty": args.hwh.is_file() and args.hwh.stat().st_size > 0,
    }
    details: dict[str, object] = {}

    if checks["hwh_exists_nonempty"]:
        root = ET.parse(args.hwh).getroot()
        modules = {node.attrib.get("INSTANCE"): node for node in root.iter("MODULE")}
        dma = modules.get("axi_dma_cnn_0")
        conv = modules.get("fitguard_cnn_conv_0")
        checks["cnn_dma_present"] = dma is not None
        checks["cnn_conv_present"] = conv is not None
        if dma is not None:
            base = parameter(dma, "C_BASEADDR")
            mm2s = parameter(dma, "C_M_AXIS_MM2S_TDATA_WIDTH")
            s2mm = parameter(dma, "C_S_AXIS_S2MM_TDATA_WIDTH")
            details.update(dma_base_address=base, mm2s_width=mm2s, s2mm_width=s2mm)
            checks["cnn_dma_base_0x40410000"] = base == "0x40410000"
            checks["cnn_dma_mm2s_32bit"] = mm2s == "32"
            checks["cnn_dma_s2mm_32bit"] = s2mm == "32"
        else:
            checks.update(
                cnn_dma_base_0x40410000=False,
                cnn_dma_mm2s_32bit=False,
                cnn_dma_s2mm_32bit=False,
            )
        xml_text = args.hwh.read_text(encoding="utf-8")
        checks["dma_to_conv_input_connected"] = (
            'INSTANCE="fitguard_cnn_conv_0" PORT="input_r_TDATA"' in xml_text
        )
        checks["conv_to_dma_output_connected"] = (
            'INSTANCE="fitguard_cnn_conv_0" PORT="output_r_TDATA"' in xml_text
        )

    details.update(
        bit_path=str(args.bit),
        hwh_path=str(args.hwh),
        bit_bytes=args.bit.stat().st_size if args.bit.is_file() else 0,
        hwh_bytes=args.hwh.stat().st_size if args.hwh.is_file() else 0,
        bit_sha256=sha256(args.bit) if args.bit.is_file() else None,
        hwh_sha256=sha256(args.hwh) if args.hwh.is_file() else None,
    )
    result = {"checks": checks, "details": details, "passed": all(checks.values())}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
