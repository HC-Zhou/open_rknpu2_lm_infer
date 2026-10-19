#!/usr/bin/env python3
"""Summarize complete MTP generation calls and optional repeated ABBA blocks."""
import argparse
import collections
import json
import statistics
from pathlib import Path


def summarize(data):
    samples = data['samples']
    reference = {r['prompt_index']: r['tokens'] for r in samples if r['draft'] == 0}
    greedy = data['temperature'] == 0
    rows = collections.defaultdict(list)
    for record in samples:
        rows[record['prompt_index'], record['draft'], record.get('draft_p_min', 0),
             record.get('keep_low_confidence', False)].append(record)
        if greedy:
            assert record['tokens'] == reference[record['prompt_index']], 'greedy token mismatch'
        assert record['stats']['tune_calls'] == 0, 'online tuning in inference'
        assert record['stats']['weight_reload_calls'] == 0, 'weight reload in inference'
    result = {'schedule': data['schedule'], 'greedy_tokens_equal': greedy,
              'records': len(samples), 'results': []}

    def quant_counter(record, snapshot, key):
        return sum(d[key] for d in record[snapshot]['domains'] if d['loaded_w4'] or d['loaded_w8'])

    for (prompt, draft, p_min, keep), group in sorted(rows.items()):
        rounds = [r for record in group for r in record['rounds']]
        proposed = sum(r['proposed'] for r in rounds)
        attempts = sum(r.get('draft_attempts', r['proposed']) for r in rounds)
        accepted = sum(r['accepted'] for r in rounds)
        tokens = sum(r['decode_tokens'] for r in group)
        elapsed = sum(r['decode_ms'] for r in group)
        row = {'prompt_index': prompt, 'draft': draft, 'samples': len(group),
               'draft_p_min': p_min, 'keep_low_confidence': keep,
               'prompt_tokens': len(group[0]['prompt']),
               'decode_tokens': tokens, 'decode_ms': elapsed,
               'decode_tps': tokens * 1000 / elapsed,
               'decode_tps_sd': statistics.stdev(r['decode_tps'] for r in group) if len(group) > 1 else None,
               'ttft_ms_mean': statistics.mean(r['ttft_ms'] for r in group),
               'generation_ms_mean': statistics.mean(r['generation_ms'] for r in group),
               'request_ms_mean': statistics.mean(r.get('request_ms', r['generation_ms']) for r in group),
               'proposed': proposed, 'accepted': accepted,
               'draft_attempts': attempts, 'discarded_attempts': attempts-proposed,
               'acceptance': accepted / proposed if proposed else None,
               'accepted_per_attempt': accepted / attempts if attempts else None,
               'accepted_histogram': dict(sorted(collections.Counter(r['accepted'] for r in rounds).items())),
               'quant_cache_misses_total_max': max(quant_counter(r, 'stats', 'tune_cache_misses') for r in group),
               'quant_fallbacks_total_max': max(quant_counter(r, 'stats', 'strategy_fallbacks') for r in group),
               'quant_fallback_calls_in_run': sum(
                   quant_counter(r, 'stats', 'strategy_fallback_calls')
                   - quant_counter(r, 'stats_before', 'strategy_fallback_calls')
                   for r in group)}
        for name in ('draft_ms', 'verify_ms', 'repair_ms'):
            row[name+'_per_round'] = statistics.mean(r[name] for r in rounds) if rounds else None
        if draft:
            full = [r for r in rounds if r['proposed'] == draft]
            row['rounds'] = len(rounds)
            row['full_proposal_rounds'] = len(full)
            row['full_proposal_prefix_histogram'] = dict(sorted(collections.Counter(r['accepted'] for r in full).items()))
            row['marginal_acceptance'] = [sum(r['accepted'] >= i for r in full)/len(full)
                                          for i in range(1, draft+1)] if full else []
            row['accepted_tokens_per_round'] = accepted/len(rounds)
            row['emitted_tokens_per_round'] = tokens/len(rounds)
            row['verify_logical_rows'] = sum(r['proposed']+1 for r in rounds)
            row['verify_physical_rows'] = sum(1 << r['proposed'].bit_length() for r in rounds)
            row['overhead_ms_per_round'] = (elapsed-sum(r[k] for r in rounds for k in
                ('draft_ms', 'verify_ms', 'repair_ms')))/len(rounds)
            for suffix in ('calls', 'rows'):
                key = 'attention_verify_' + suffix
                row[key] = sum(r['stats'].get(key,
                    r['stats'].get('attention_verify_batch_' + suffix, 0)
                    + r['stats'].get('attention_verify_serial_' + suffix, 0)) for r in group)
            baseline = rows[prompt, 0, 0, False]
            base_tps = sum(r['decode_tokens'] for r in baseline)*1000/sum(r['decode_ms'] for r in baseline)
            row['decode_speedup'] = row['decode_tps']/base_tps
            row['request_speedup'] = statistics.mean(r.get('request_ms', r['generation_ms']) for r in baseline)/row['request_ms_mean']
            fixed = rows[prompt, draft, 0, False]
            fixed_tps = sum(r['decode_tokens'] for r in fixed)*1000/sum(r['decode_ms'] for r in fixed)
            row['speedup_vs_fixed_draft'] = row['decode_tps']/fixed_tps
            row['draft_attempts_per_round'] = attempts/len(rounds)
            row['confidence_stops'] = sum(r.get('confidence_stopped', False) for r in rounds)
        head_calls = attempts+len(rounds)+len(group) if draft else sum(len(r['tokens']) for r in group)
        row['head_ms_mean'] = statistics.mean(r['stats']['head_ms'] for r in group)
        row['head_ms_per_call'] = sum(r['stats']['head_ms'] for r in group)/head_calls
        result['results'].append(row)
    return result


def abba(data):
    schedule = data['schedule']
    assert len(schedule) % 4 == 0
    result = []
    for block in range(len(schedule)//4):
        start = 4*block
        a, b, c, d = schedule[start:start+4]
        assert a == d == 0 and b == c and b > 0, 'expected repeated 0,K,K,0'
        selected = [r for r in data['samples'] if start <= r['block'] < start+4]
        for prompt in sorted({r['prompt_index'] for r in selected}):
            group = [r for r in selected if r['prompt_index'] == prompt]
            modes = [[r for r in group if r['draft'] == k] for k in (0, b)]
            assert all(len(r) == 2 for r in modes), 'ABBA requires one sample per configuration'
            rates = [sum(r['decode_tokens'] for r in mode)*1000/sum(r['decode_ms'] for r in mode)
                     for mode in modes]
            result.append(dict(block=block, prompt_index=prompt, draft=b,
                               ar_tps=rates[0], mtp_tps=rates[1], speedup=rates[1]/rates[0]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--abba', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    result = summarize(data)
    if args.abba:
        result['abba'] = abba(data)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for row in result['results']:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
