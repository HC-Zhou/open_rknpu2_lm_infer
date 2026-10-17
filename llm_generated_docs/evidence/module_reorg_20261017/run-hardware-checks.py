import re
from pathlib import Path
import subprocess

work=Path(__file__).resolve().parent
runtime=Path('/tmp/qwen35-runtime')
log=(work/'tests.log').read_text()
relative=re.search(r'Running unittests src/lib.rs \(([^)]+)\)', log).group(1)
binary=runtime/relative
for label,filters in [('native-operators',['npu::tests::']),
                      ('narrow-projection',['npu::narrow_projection::narrow_w8_gdn_gates_match_integer_oracle','--exact'])]:
    with (work/f'{label}.log').open('w') as output:
        subprocess.run(['python3',str(runtime/'scripts/run_board.py'),'--wait-lock','--cpu-khz','2256000','--skip-dmc',
                        '--report',str(work/f'{label}-clocks.json'),'--',str(binary),*filters,
                        '--ignored','--test-threads=1'],cwd=runtime,stdout=output,stderr=subprocess.STDOUT,check=True)
    print(label,'PASS',flush=True)
