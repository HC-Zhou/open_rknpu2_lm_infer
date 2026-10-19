#!/usr/bin/env python3
"""Compare real CLI MTP and AR output, stats, stdin and image generation."""
import copy
import json
import subprocess
import time

from vision_support import arguments, command, environment


def main():
    args = arguments(__doc__)
    assert args.spec_type == 'draft-mtp' and args.spec_draft_n_max > 0, 'enable --spec-type draft-mtp'
    report = {'cases': [], 'passed': False}

    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')

    def run(name, prompt, ar=False, image=None, stdin=False):
        options = copy.copy(args)
        if ar:
            options.spec_type = 'none'
            options.spec_draft_p_min = 0.
        cmd = command(options) + ['run', '--conversation', '--temp', '0',
                                  '-n', '32', '--stats']
        if image:
            cmd += ['--image', str(args.photos/image)]
        if not stdin:
            cmd += ['-p', prompt]
        start = time.monotonic()
        result = subprocess.run(cmd, input=prompt if stdin else None,
                                text=True, capture_output=True, timeout=300,
                                env=environment('warn'), close_fds=True)
        (args.output/(name+'.stdout')).write_text(result.stdout)
        (args.output/(name+'.stderr')).write_text(result.stderr)
        row = {'name': name, 'command': cmd, 'exit_code': result.returncode,
               'text': result.stdout, 'elapsed_seconds': time.monotonic()-start}
        report['cases'].append(row)
        save()
        assert result.returncode == 0, result.stderr
        lines = result.stderr.splitlines()
        stats_start = next(i for i, line in enumerate(lines) if line == '{')
        stats = json.loads('\n'.join(lines[stats_start:]))
        row['result'] = stats
        assert stats['completion_tokens'] > 0 and result.stdout.strip(), row
        assert stats['stats']['speculation_enabled'] == (options.spec_type == 'draft-mtp')
        if options.spec_type == 'draft-mtp':
            assert stats['timings']['spec_rounds'] > 0, row
            assert stats['stats']['spec_draft_p_min'] == options.spec_draft_p_min
            assert stats['stats']['spec_draft_sampling'] == options.spec_draft_sampling
        save()
        print(name, result.stdout.strip(), flush=True)
        return row

    try:
        prompt = 'Explain the difference between TCP and UDP in a few sentences.'
        baseline = run('ar', prompt, ar=True)
        speculative = run('mtp-stdin', prompt, stdin=True)
        assert speculative['text'] == baseline['text']
        assert speculative['result']['completion_tokens'] == baseline['result']['completion_tokens']
        if args.spec_draft_p_min > 0:
            assert speculative['result']['timings']['mtp_confidence_stops'] > 0
        image = run('mtp-image', 'How many cats are in this photograph? Answer briefly.', image='cats.png')
        assert any(word in image['text'].lower() for word in ('two', '2', 'pair'))
        assert image['result']['stats']['vision_encoder_calls'] > 0
        report['passed'] = True
    finally:
        save()


if __name__ == '__main__':
    main()
