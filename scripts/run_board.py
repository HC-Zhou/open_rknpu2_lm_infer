#!/usr/bin/env python3
"""Run one board command under the shared submit lock and fixed clocks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


TARGETS = {
    Path("/sys/class/devfreq/fdab0000.npu"): ("min_freq", "max_freq", "cur_freq", 1000000000),
    Path("/sys/class/devfreq/dmc"): ("min_freq", "max_freq", "cur_freq", 2112000000),
    Path("/sys/devices/system/cpu/cpufreq/policy4"): ("scaling_min_freq", "scaling_max_freq", "scaling_cur_freq", 2400000),
    Path("/sys/devices/system/cpu/cpufreq/policy6"): ("scaling_min_freq", "scaling_max_freq", "scaling_cur_freq", 2400000),
}


def snapshot():
    result = {}
    for root, (minimum, maximum, current, _) in TARGETS.items():
        fields = [minimum, maximum, current]
        fields.append("scaling_governor" if minimum.startswith("scaling") else "governor")
        result[str(root)] = {field: (root / field).read_text().strip() for field in fields}
    result["thermal"] = {
        str(path.parent): {"type": (path.parent / "type").read_text().strip(), "millidegrees_c": int(path.read_text())}
        for path in Path("/sys/class/thermal").glob("thermal_zone*/temp")
    }
    return result


def write(path, value):
    subprocess.run(["sudo", "-n", "tee", str(path)], input=str(value) + "\n",
                   text=True, stdout=subprocess.DEVNULL, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wait-lock", action="store_true", help="wait for another board job to release the NPU")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("specify the executable and arguments after --")
    report = {"command": command, "started_unix": time.time(), "restore_errors": []}
    status = 1
    with open("/tmp/rknpu_lowlevel_submit.lock", "a+") as lock:
        try:
            if args.wait_lock:
                print("Waiting for the shared NPU lock", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_lock else fcntl.LOCK_NB))
        except BlockingIOError:
            raise SystemExit("NPU is in use: /tmp/rknpu_lowlevel_submit.lock")
        before = snapshot()
        report["before"] = before
        changed = []
        try:
            subprocess.run(["sudo", "-n", "true"], check=True)
            for root, (minimum, maximum, _, target) in TARGETS.items():
                changed.append(root)
                write(root / maximum, target)
                write(root / minimum, target)
            report["fixed"] = snapshot()
            for root, (minimum, maximum, _, target) in TARGETS.items():
                fixed = report["fixed"][str(root)]
                if int(fixed[minimum]) != target or int(fixed[maximum]) != target:
                    raise RuntimeError("fixed clock verification failed: " + str(root))
            env = os.environ.copy()
            env["LLM_INFER_LOCK_FD"] = str(lock.fileno())
            env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
            env.setdefault("GOMP_SPINCOUNT", "0")
            report["environment"] = {k: v for k, v in env.items() if k.startswith(("RK_NPU_", "OMP_", "GOMP_", "OPENBLAS_"))}
            start = time.monotonic()
            status = subprocess.run(command, env=env, pass_fds=(lock.fileno(),)).returncode
            report["command_elapsed_seconds"] = time.monotonic() - start
            report["after_command"] = snapshot()
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            status = 1
            report["error"] = str(error)
            print("run_board:", error, flush=True)
        finally:
            for root in reversed(changed):
                minimum, maximum, _, _ = TARGETS[root]
                try:
                    write(root / minimum, before[str(root)][minimum])
                    write(root / maximum, before[str(root)][maximum])
                except (OSError, subprocess.CalledProcessError) as error:
                    report["restore_errors"].append(str(error))
            report["restored"] = snapshot()
            if report["restore_errors"]:
                status = 1
            report["exit_code"] = status
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
