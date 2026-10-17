#!/usr/bin/env python3
"""Export C7 depthwise Conv weights and saved runtime QKV for causal_conv_bench."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    layers = sorted(int(name.split(".layers.")[1].split(".")[0]) for name in index
                    if name.endswith(".linear_attn.conv1d.weight"))
    assert len(layers) == 24, "expected C7's 24 GDN layers"
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = []
    with (args.output / "weights.f32").open("wb") as weights_out, \
            (args.output / "trace.f32").open("wb") as trace_out:
        for layer in layers:
            name = f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"
            shard = args.model / index[name]
            with shard.open("rb") as source:
                header_size = struct.unpack("<Q", source.read(8))[0]
                entry = json.loads(source.read(header_size))[name]
                assert entry["shape"] == [8192, 1, 4] and entry["dtype"] == "BF16"
                begin, end = entry["data_offsets"]
                source.seek(8 + header_size + begin)
                raw = source.read(end - begin)
            weight = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
            assert weight.size == 8192 * 4 and np.isfinite(weight).all()
            weights_out.write(weight.tobytes())
            path = args.trace / f"p0-l{layer}-gdn_qkv.f32"
            data = np.fromfile(path, dtype="<f4").reshape(24, 8192)
            assert np.isfinite(data).all()
            trace_out.write(data.tobytes())
            metadata.append(dict(layer=layer, tensor=name, shard=shard.name, trace=str(path),
                                 weight_max_abs=float(np.max(np.abs(weight))),
                                 input_max_abs=float(np.max(np.abs(data)))))
    report = dict(model=str(args.model), layers=metadata,
                  note="Real 24-row QKV trace; benchmark seeds history with first3 rows, "
                       "cycles remaining21 for rows>21. Raw weights/activations stay outside the repository.")
    (args.output / "input-metadata.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
