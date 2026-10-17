import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['before', 'after'], required=True)
args = parser.parse_args()
work = Path(__file__).resolve().parent
runtime = Path('/tmp/qwen35-runtime')
binary = Path('/tmp/qwen35-before-module-reorg') if args.phase == 'before' else runtime/'target/release/open_rknpu2_lm_infer'
models = {
    'qwen35': Path('/home/firefly/mnt/3090x4/llm/w4a8_quant/transfer/qwen35/models/C7'),
    'qwen3': Path('/home/firefly/mnt/3090x4/llm/w4a8_quant/Qwen3-4B-W4A8-Mixed-LMHeadW8'),
}
windows = json.loads((work.parent/'c7-ppl32/tokens.json').read_text())[:2]
prompt = windows[0][:129]
continuation = windows[0][129:133]
records = []
for name, model in models.items():
    # Qwen3 has a different tokenizer/vocabulary. This fixed in-range sequence
    # checks refactor equivalence only, not cross-model language-model quality.
    inputs = [[t % 10000 for t in w] for w in windows] if name == 'qwen3' else windows
    eval_prompt = [t % 10000 for t in prompt] if name == 'qwen3' else prompt
    eval_continuation = [t % 10000 for t in continuation] if name == 'qwen3' else continuation
    for suffix, value in [('windows', inputs), ('prompt', eval_prompt), ('continuation', eval_continuation)]:
        (work/f'{name}-{suffix}.json').write_text(json.dumps(value)+'\n')
    common = [str(binary), '--model', str(model), '--ctx-size', '512', '--ubatch-size', '128',
              '--threads', '4', '--cpu-mask', '0xf0', '--npu-mask', '7', '--attention-tile', '512',
              '--no-tune', '--tune-cache', str(work/f'{name}-cache')]
    for mode in ['eval', 'ppl']:
        label = f'{args.phase}-{name}-{mode}'
        report = work/f'{label}.json'
        command = common+[mode, '--tokens-file', str(work/f'{name}-prompt.json' if mode=='eval' else work/f'{name}-windows.json'), '--report', str(report)]
        if mode == 'eval':
            command += ['--continuation-file', str(work/f'{name}-continuation.json'), '--dump-logits', str(work/f'{label}.f32')]
        else:
            command += ['--score-start', '257']
        with (work/f'{label}.log').open('w') as log:
            subprocess.run(['python3', str(runtime/'scripts/run_board.py'), '--wait-lock', '--cpu-khz', '2256000', '--skip-dmc',
                            '--report', str(work/f'{label}-clocks.json'), '--', *command], cwd=runtime, stdout=log, stderr=subprocess.STDOUT, check=True)
        records.append({'label': label, 'command': command})
        print(label, 'PASS', flush=True)
(work/f'{args.phase}-commands.json').write_text(json.dumps({'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'runs': records}, indent=2)+'\n')
