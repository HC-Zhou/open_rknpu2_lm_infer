"""Exercise the alternate-board clock policy and restore after a failed command."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import json

spec = importlib.util.spec_from_file_location("run_board", Path(__file__).resolve().parents[1] / "run_board.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BoardRunnerTests(unittest.TestCase):
    def test_main_board_defaults_and_spare_policy(self):
        self.assertEqual(runner.clock_targets(2400000, False), runner.TARGETS)
        spare = runner.clock_targets(2256000, True)
        self.assertNotIn(Path("/sys/class/devfreq/dmc"), spare)
        self.assertEqual(len(spare), 3)
        for minimum, _, _, target in spare.values():
            self.assertEqual(target, 2256000 if minimum.startswith("scaling") else 1000000000)

    def test_failed_command_restores_all_changed_bounds_and_records_exit(self):
        targets = runner.clock_targets(2256000, True)
        before = {}
        for root, (minimum, maximum, current, target) in targets.items():
            before[str(root)] = {minimum: "408000", maximum: str(target), current: "408000"}
        writes = []
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "clocks.json"
            def run(command, **_kwargs):
                return subprocess.CompletedProcess(command, 7 if command == ["failure"] else 0)
            fixed = {str(root): {minimum: str(target), maximum: str(target)}
                     for root, (minimum, maximum, _current, target) in targets.items()}
            with patch("sys.argv", ["run_board", "--cpu-khz", "2256000", "--skip-dmc", "--report", str(report), "--", "failure"]), \
                 patch.object(runner, "snapshot", side_effect=[before, fixed, fixed, before]), \
                 patch.object(runner, "write", side_effect=lambda path, value: writes.append((path, str(value)))), \
                 patch.object(runner.subprocess, "run", side_effect=run), \
                 patch.object(runner.fcntl, "flock"):
                self.assertEqual(runner.main(), 7)
            data = json.loads(report.read_text())
            self.assertEqual(data["exit_code"], 7)
            self.assertEqual(data["restore_errors"], [])
            self.assertFalse(data["clock_policy"]["dmc_controlled"])
            self.assertEqual(data["restored"], before)
        restore = writes[6:]
        self.assertEqual(len(restore), 6)
        for root in targets:
            minimum, maximum, _, _ = targets[root]
            self.assertIn((root / minimum, before[str(root)][minimum]), restore)
            self.assertIn((root / maximum, before[str(root)][maximum]), restore)


if __name__ == "__main__":
    unittest.main()
