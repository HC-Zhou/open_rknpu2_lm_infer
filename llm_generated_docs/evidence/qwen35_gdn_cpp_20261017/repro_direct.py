import json, subprocess, hashlib, sys
from pathlib import Path
root=Path('/home/firefly/mnt/3090x4/tmp/qwen35-dev')
out=root/'gdn-cpp-20261017'/sys.argv[1]
out.mkdir(exist_ok=True, parents=True)
records=[]
for rows in [1,128,512]:
    src=root/'gdn-align-20261017/micro'/f'input-{rows}.f32'
    expected=(root/'gdn-align-20261017/micro'/f'ggml-{rows}.f32').read_bytes()
    for index,mode in enumerate(['unfused','ggml','fused','fused','ggml','unfused']):
        dest=out/f'out-{rows}.f32'
        cmd=['/tmp/qwen35-gdn-reference-build/gdn_reference',str(src),str(dest),str(rows),'4',str(100 if rows==1 else 5),'7',mode]
        run=subprocess.run(cmd, text=True, capture_output=True, check=True)
        timing=json.loads(run.stdout)
        assert dest.read_bytes()==expected,(rows,mode)
        records.append(dict(rows=rows,mode=mode,order=index,milliseconds=timing['milliseconds'],output_and_state_exact=True))
        (out/'comparison.json').write_text(json.dumps(dict(records=records),indent=2)+'\n')
print(json.dumps(records))
