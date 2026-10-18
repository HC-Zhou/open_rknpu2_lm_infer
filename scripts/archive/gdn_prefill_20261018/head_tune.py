import os
from pathlib import Path
import subprocess

bindings = max(Path('target/float-autotune/release/build').glob('open_rknpu2_lm_infer-*/out/npu_bindings.rs'),
               key=lambda p: p.stat().st_mtime)
env = os.environ.copy()
env['RKNPU2_GDN_BINDINGS'] = str(bindings.resolve())
subprocess.run(['rustc', '--edition=2024', '-O', str(Path(__file__).with_name('head_tune.rs')),
                '-L', 'native=' + str(bindings.parent / 'build'),
                '-l', 'static=rk_npu_matmul', '-l', 'stdc++', '-l', 'gomp',
                '-l', 'pthread', '-l', 'm', '-l', 'dl',
                '-o', 'logs/gdn-prefill-head-tune'], check=True, env=env)
subprocess.run(['logs/gdn-prefill-head-tune'], check=True,
               pass_fds=(int(os.environ['LLM_INFER_LOCK_FD']),))
