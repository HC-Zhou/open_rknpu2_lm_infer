import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

paths = sorted(Path('target/float-autotune/release/build').glob('open_rknpu2_lm_infer-*/out/npu_bindings.rs'),
               key=lambda p: p.stat().st_mtime, reverse=True)
bindings = next(p for p in paths if 'rk_npu_gdn_config' in p.read_text())
env = os.environ.copy()
env['RKNPU2_GDN_BINDINGS'] = str(bindings.resolve())
subprocess.run(['rustc', '--edition=2024', '-O', str(Path(__file__).with_name('sweep.rs')),
                '-L', 'native=' + str(bindings.parent / 'build'),
                '-l', 'static=rk_npu_matmul', '-l', 'stdc++', '-l', 'gomp',
                '-l', 'pthread', '-l', 'm', '-l', 'dl',
                '-o', 'logs/gdn-prefill-search'], check=True, env=env)
out = Path('logs/gdn-prefill-tuned')
out.mkdir(exist_ok=True)
stem = 'gdn-search' + ('-' + sys.argv[1] if len(sys.argv) > 1 else '')
with (out / (stem + '.jsonl')).open('w') as f:
    subprocess.run(['logs/gdn-prefill-search', *sys.argv[2:]], stdout=f, check=True,
                   pass_fds=(int(os.environ['LLM_INFER_LOCK_FD']),))
records = [json.loads(s) for s in (out / (stem + '.jsonl')).read_text().splitlines()]
for row in records:
    row['median_us'] = statistics.median(row['total_us'])
    row['mean_us'] = statistics.mean(row['total_us'])
summary = {str(m): sorted((r for r in records if r['rows'] == m), key=lambda r: r['median_us'])
           for m in sorted({r['rows'] for r in records})}
(out / (stem + '.json')).write_text(json.dumps(summary, indent=2) + '\n')
for m, rows in summary.items():
    print(m, 'best candidates', json.dumps(rows[:4]), flush=True)
