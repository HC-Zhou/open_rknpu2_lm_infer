#!/usr/bin/env python3
"""Compare fixed-clock board reports from native and Rust inference programs."""
import argparse
import json
import statistics
from pathlib import Path


def samples(report, rust):
    out = {}
    if rust:
        for case in report["cases"]:
            depth = case["depth"]
            for phase, count in [("prefill",case["prompt"]),("decode",case["generate"])]:
                if count:
                    out[(phase,count,depth)] = case[phase]["samples_tokens_per_second"]
    else:
        for case in report["tests"]:
            phase = "prefill" if case["n_prompt"] else "decode"
            count = case["n_prompt"] or case["n_gen"]
            out[(phase,count,case["n_depth"])] = case["samples_ts"]
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",type=Path,nargs="+",required=True)
    parser.add_argument("--rust",type=Path,nargs="+",required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    native,rust={},{}
    identities=[]
    for paths,group,kind in [(args.native,native,False),(args.rust,rust,True)]:
        for path in paths:
            value=json.loads(path.read_text())
            identities.append(value["synthetic_token_ids"])
            for key,values in samples(value,kind).items(): group.setdefault(key,[]).extend(values)
    for ids in identities[1:]:
        common=min(len(ids),len(identities[0]))
        if ids[:common]!=identities[0][:common]: raise SystemExit("input token IDs differ")
    rows=[]
    for key in native.keys() & rust.keys():
        a,b=native[key],rust[key]
        am,bm=statistics.mean(a),statistics.mean(b)
        rows.append({"phase":key[0],"tokens":key[1],"depth":key[2],"native_samples":a,"rust_samples":b,
            "native_mean":am,"rust_mean":bm,"native_stddev":statistics.stdev(a) if len(a)>1 else 0,
            "rust_stddev":statistics.stdev(b) if len(b)>1 else 0,"change_percent":(bm/am-1)*100})
    rows.sort(key=lambda row:(row["phase"],row["depth"],row["tokens"]))
    result={"native_reports":[str(x) for x in args.native],"rust_reports":[str(x) for x in args.rust],"same_token_ids":True,"results":rows}
    args.report.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
