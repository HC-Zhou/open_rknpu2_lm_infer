import json
from pathlib import Path
import subprocess

root = Path("/home/firefly/mnt/3090x4/tmp/qwen35-dev/gdn-small-20261017")
items = [json.loads(line) for line in (root / "candidate-artifacts.jsonl").read_text().splitlines()]
test = next(x["executable"] for x in reversed(items)
            if x.get("target", {}).get("name") == "cpu_layers" and x.get("executable"))
with (root / "cpu-tests.log").open("w") as log:
    subprocess.run([test, "--test-threads=1"], stdout=log, stderr=subprocess.STDOUT, check=True)
print((root / "cpu-tests.log").read_text(), flush=True)
for mode in ["core", "mixer"]:
    for turn, side in enumerate(["baseline", "candidate", "candidate", "baseline"]):
        result = root / f"rust-{mode}-{turn}-{side}.json"
        command = [str(root / f"rust-small-{side}"), str(root / "cases"), str(result),
                   "--layers", "24", "--threads", "4", "--rows",
                   "1,2,3,4,5,6,7,8" if mode == "mixer" else "1,2,4,7,8,128"]
        if mode == "mixer":
            command.append("--mixer")
        with (root / f"rust-{mode}-{turn}-{side}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(result.name, flush=True)
# Single-thread oracle check and complete CPU mixer scaling.
for side in ["baseline", "candidate"]:
    for mode in ["core", "mixer"]:
        command = [str(root / f"rust-small-{side}"), str(root / "cases"),
                   str(root / f"rust-{mode}-t1-{side}.json"),
                   "--layers", "1", "--threads", "1", "--rows", "1,2,4,7,8"]
        if mode == "mixer":
            command.append("--mixer")
        with (root / f"rust-{mode}-t1-{side}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"t1 {mode} {side}", flush=True)
