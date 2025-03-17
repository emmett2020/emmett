"""
TODO: This script is still developing. Please use ncu directly now.
Run CUDA kernel and get it's performance.
"""

import subprocess
import csv
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import loguru


class CUDAPerfGuard:
    """
    TODO
    """

    def __init__(self, config: Dict):
        self.config = config
        self.ncu_path = Path(config["ncu_path"])
        self.app_path = Path(config["app_path"])
        self.thresholds = config["thresholds"]
        self.report_file = Path(config["report_file"])
        self._validate_paths()

    def _validate_paths(self):
        if not self.ncu_path.exists():
            raise FileNotFoundError(
                f"The path of ncu not exists: {self.ncu_path}")
        if not self.app_path.exists():
            raise FileNotFoundError(
                f"The path of app not exists: {self.app_path}")

    def _build_ncu_command(self) -> List[str]:
        metrics = ",".join(
            set(metric for kernel in self.thresholds.values()
                for metric in kernel.keys()))

        return [
            str(self.ncu_path), "--target-processes", "all", "-k",
            self.config["kernel_regex"], "--metrics", metrics, "--csv",
            "--page", "raw", "--export",
            str(self.report_file), "--force-overwrite",
            str(self.app_path)
        ] + self.config.get("app_args", [])

    def run_ncu(self):
        """TODO"""
        cmd = self._build_ncu_command()
        print(f"[INFO] ncu command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd,
                                    check=True,
                                    capture_output=True,
                                    text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            # print(f"[ERROR] launch ncu failed: {e.stderr}")
            print(e)
            print(e.stdout)
            print(e.stderr)
            sys.exit(2)

    def parse_report(self) -> Dict[str, Dict[str, float]]:
        """TODO"""
        report = {}
        with open(self.report_file, "r", encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                kernel = row["Kernel Name"]
                report[kernel] = {
                    field: float(row[field])
                    for field in row.keys()
                    if field not in ["ID", "Kernel Name"]
                }

        return report

    def validate_performance(self, report: Dict) -> Tuple[bool, List[str]]:
        """TODO"""
        failures = []
        all_pass = True

        for kernel, metrics in report.items():
            if kernel not in self.thresholds:
                continue

            for metric, value in metrics.items():
                if metric not in self.thresholds[kernel]:
                    continue

                threshold = self.thresholds[kernel][metric]
                op = threshold["op"]
                ref = threshold["value"]

                if not eval(f"{value} {op} {ref}", {}, {"value": value}):
                    failures.append(
                        f"{kernel}.{metric}: {value} {op} {ref} (违反阈值)")
                    all_pass = False

        return all_pass, failures

    def generate_report(self, passed: bool, failures: List[str]):
        """生成可视化报告"""
        print("\n=== CUDA性能验证报告 ===")
        print(f"应用路径: {self.app_path}")
        print(f"分析内核: {self.config['kernel_regex']}")
        print(f"结果: {'通过' if passed else '失败'}")

        if failures:
            print("\n失败项:")
            for msg in failures:
                print(f"  - {msg}")

    def run(self):
        """Execute completed workflow"""
        self.run_ncu()
        report = self.parse_report()
        passed, failures = self.validate_performance(report)
        self.generate_report(passed, failures)

        if not passed:
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CUDA Performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c",
                        "--config",
                        required=True,
                        help="JSON config file path")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    guard = CUDAPerfGuard(config)
    guard.run()
