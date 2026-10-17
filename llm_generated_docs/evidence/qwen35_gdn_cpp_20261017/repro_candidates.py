import subprocess,json,sys,hashlib
from pathlib import Path
root=Path('/tmp/qwen35-runtime/native/qwen35_cpu')
base=Path('/home/firefly/mnt/3090x4/tmp/qwen35-dev/gdn-cpp-20261017')
for name,rows,tune in [('rows2',2,False),('rows4-a76',4,True),('rows2-a76',2,True)]:
    out=base/name
    out.mkdir(exist_ok=True)
    source=out/'gdn.cpp'
    source.write_text((root/'gdn.cpp').read_text().replace('constexpr size_t block_rows = 4;',f'constexpr size_t block_rows = {rows};'))
    common=['g++','-std=c++17','-O3','-DNDEBUG','-fPIC','-fopenmp','-I'+str(root)]
    if tune: common+=['-mtune=cortex-a76']
    commands=[common+['-c',str(source),'-o','/tmp/qwen35-gdn-candidate.o'],
        ['g++','-std=c++17','-O3','-DNDEBUG','-fopenmp','-DGDN_HAVE_LOCAL_CPU','-I'+str(root),'-I/tmp/qwen35-ggml/include','/tmp/qwen35-runtime/gdn_reference/main.cpp','/tmp/qwen35-gdn-candidate.o','-L/tmp/qwen35-gdn-reference-build/ggml/src','-lggml','-lggml-cpu','-lggml-base','-Wl,-rpath,/tmp/qwen35-gdn-reference-build/ggml/src','-o','/tmp/qwen35-gdn-reference-build/gdn_reference']]
    for cmd in commands: subprocess.run(cmd,check=True)
    (out/'build.json').write_text(json.dumps(dict(commands=commands,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n')
    cmd=['python3','/tmp/qwen35-runtime/scripts/run_board.py','--wait-lock','--cpu-khz','2256000','--skip-dmc','--report',str(base/(name+'-clocks.json')),'--','taskset','0xf0','python3','/tmp/qwen35-direct-bench.py',name]
    with (out/'run.log').open('w') as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,flush=True)
