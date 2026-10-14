#!/usr/bin/env python3
"""Create a source tree with the pinned upstream NPU operator submodule."""
import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", ":native/rk_npu_matmul"],
        text=True,
    ).strip()
    output = args.output.resolve()
    if output.exists():
        parser.error("output must be a new directory")
    output.mkdir(parents=True)
    for name in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "build.rs", ".gitignore", ".gitmodules", "LICENSE", "README.md", "AGENTS.md", "CONTRIBUTING.md", "THIRD_PARTY_NOTICES.md"]:
        shutil.copy2(root / name, output / name)
    for name in ["src", "tests", "llm_generated_docs", "scripts", ".github"]:
        shutil.copytree(root / name, output / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    def git(*arguments):
        subprocess.run(["git", "-C", str(output), *arguments], check=True)

    git("init", "--quiet")
    git("add", ".gitmodules")
    git("update-index", "--add", "--cacheinfo", f"160000,{revision},native/rk_npu_matmul")
    git("submodule", "update", "--init", "--depth", "1")
    print(output)


if __name__ == "__main__":
    main()
