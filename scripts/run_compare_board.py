#!/usr/bin/env python3
"""Run a paired or ABBA complete-model comparison on an RK3588 board."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--native-bin",type=Path,required=True)
    parser.add_argument("--rust-bin",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=["short","representative"],default="short")
    parser.add_argument("--repetitions",type=int,default=3)
    parser.add_argument("--reuse-cache",action="store_true",help="use already prepared per-program caches; measured runs still require zero misses")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent
    order=["native","rust","rust","native"] if args.mode=="short" else ["native","rust"]
    caches={kind:args.cache/kind for kind in ["native","rust"]}
    for cache in caches.values(): cache.mkdir(parents=True,exist_ok=True)
    reports={"native":[],"rust":[]}
    clocks=[]
    runs=[] if args.reuse_cache else [("warm-native","native",1,False),("warm-rust","rust",1,False)]
    runs += [(f"{args.mode}-{index+1}-{kind}",kind,args.repetitions,True) for index,kind in enumerate(order)]
    for label,kind,repetitions,measured_run in runs:
        label=f"{args.mode}-{label}" if not measured_run else label
        report=args.output/f"{label}.json"
        clock=args.output/f"{label}-clocks.json"
        common=["--context","4096","--chunk","64","--threads","4","--cpu-mask","0xf0","--npu-mask","7","--attention-tile","512","--tune-cache",str(caches[kind]),"--report",str(report)]
        if kind=="native":
            logical_batch=4096 if args.mode=="representative" else 512
            command=[str(args.native_bin),"perf","--model",str(args.model),*common,"-b",str(logical_batch),"-r",str(repetitions)]
            command += ["--suite","representative"] if args.mode=="representative" else ["-p","512","-n","128"]
        else:
            command=[str(args.rust_bin),"bench","--model",str(args.model),*common,"--repetitions",str(repetitions)]
            command += ["--suite","representative"] if args.mode=="representative" else ["--prompt","512","--generate","128"]
        print(label,flush=True)
        with (args.output/f"{label}.log").open("w") as log:
            subprocess.run([sys.executable,str(scripts/"run_board.py"),"--wait-lock","--report",str(clock),"--",*command],stdout=log,stderr=subprocess.STDOUT,check=True)
        measured=json.loads(clock.read_text())
        if measured["exit_code"] or measured["restore_errors"]: raise SystemExit("clock restoration failed")
        if measured_run:
            value=json.loads(report.read_text())
            stats=value["stats"] if kind=="rust" else value
            if stats["tune_cache_misses"]: raise SystemExit(f"{label}: tuning cache changed")
            reports[kind].append(report)
            clocks.append(str(clock))
    subprocess.run([sys.executable,str(scripts/"compare_perf.py"),"--native",*[str(p) for p in reports["native"]],"--rust",*[str(p) for p in reports["rust"]],"--report",str(args.output/f"{args.mode}-comparison.json")],check=True)
    (args.output/f"{args.mode}-protocol.json").write_text(json.dumps({"order":order,"model":str(args.model),"context":4096,"chunk":64,"threads":4,"npu_mask":7,"cpu_mask":"0xf0","native_logical_batch":4096 if args.mode=="representative" else 512,"prefill_head_calls_per_run":1,"caches":{key:str(value) for key,value in caches.items()},"warmup_processes":0 if args.reuse_cache else 2,"required_tune_cache_misses":0,"clocks":clocks},indent=2)+"\n")


if __name__=="__main__":
    main()
