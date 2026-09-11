import json, os, statistics

if os.path.exists('_probe_data.jsonl'):
    rows = []
    for line in open('_probe_data.jsonl', encoding='utf-8'):
        try: rows.append(json.loads(line))
        except: pass
    
    ok = [r for r in rows if r.get('status')=='ok']
    errs = [r for r in rows if r.get('status')=='error']
    print('Probe: ok=%d err=%d total=%d' % (len(ok), len(errs), len(rows)))
    
    if ok:
        tps = [r['gen_tps'] for r in ok if r.get('gen_tps',0)>0]
        gate = [r.get('gate_wait_ms',0) for r in ok]
        comm = [r.get('comm_ms',0) for r in ok]
        print('gen_tps: avg=%.1f min=%.1f max=%.1f (N=%d)' % (statistics.mean(tps), min(tps), max(tps), len(tps)))
        print('gate_ms: avg=%.0f min=%.0f max=%.0f' % (statistics.mean(gate), min(gate), max(gate)))
        print('comm_ms: avg=%.0f' % statistics.mean(comm))
        print()
        print('最近 20 条:')
        for r in ok[-20:]:
            g = r.get('gate_wait_ms', 0)
            c = r.get('comm_ms', 0)
            print('  tps=%5.1f gate=%7.0fms comm=%6.0fms pt=%s gt=%s' % (
                r['gen_tps'], g, c, r.get('prompt_tokens','?'), r.get('completion_tokens','?')))
    
    if errs:
        print()
        print('错误 (前 5):')
        for r in errs[:5]:
            print('  gate=%sms err=%s' % (r.get('gate_wait_ms','?'), str(r.get('error',''))[:100]))
else:
    print('No probe file')
