# -*- coding: utf-8 -*-
"""
探针版 PDF 翻译 benchmark — 三层 timing + 详细日志
=================================================
探针层级:
  L1 PDF 结构解析 (BabelDOC Parse stages)
  L2 批量翻译调用 (translate_batch / translate 单次)
  L3 llama-server HTTP 请求 (request/response timing)

输出:
  - 控制台 (flush=True, 行缓冲)
  - bench_probe.log (完整 JSON 日志)
"""
import sys, os, time, json, logging
sys.path.insert(0, r'D:\PostNews\Translate')
os.chdir(r'D:\PostNews\Translate')
os.environ['ORT_DISABLE_CUDA'] = '1'

# ── 行缓冲 stdout ──
sys.stdout.reconfigure(line_buffering=True)

# ── 全局日志文件 ──
LOG_PATH = r'D:\PostNews\Translate\bench_probe.log'
_log_fh = open(LOG_PATH, 'w', encoding='utf-8')

def P(msg, **kv):
    """打印 + 写日志"""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    if kv:
        line += " | " + " ".join(f"{k}={v}" for k,v in kv.items())
    print(line, flush=True)
    _log_fh.write(json.dumps({"t": time.time(), "msg": msg, **kv}, ensure_ascii=False) + "\n")
    _log_fh.flush()

# ══════════════════════════════════════════════════
#  L3 探针: 给 llama_http_engine 打 monkey-patch
# ══════════════════════════════════════════════════
P("=== L3 注入探针: monkey-patch LlamaBatchEngine.translate ===")
import llama_http_engine as lhe
_orig_translate = lhe.LlamaBatchEngine.translate

def _probed_translate(self, text, max_tokens=256):
    t0 = time.time()
    result = _orig_translate(self, text, max_tokens)
    dt = (time.time() - t0) * 1000
    L = len(text)
    R = len(result) if result else 0
    P("L3 translate", dt_ms=f"{dt:.0f}", chars_in=L, chars_out=R, tok_per_ms=f"{R/max(dt,1):.2f}", preview=text[:40].replace('\n',' '))
    return result

lhe.LlamaBatchEngine.translate = _probed_translate
P("  ✅ 探针已注入")

# ══════════════════════════════════════════════════
#  启动 llama-server
# ══════════════════════════════════════════════════
P("=== 启动 llama-server ===")
t0 = time.time()
ok = lhe.ensure_server_ready()
P("  启动完成", ok=ok, elapsed_s=f"{time.time()-t0:.1f}")

# GPU 验证
import requests
t0 = time.time()
r = requests.post('http://127.0.0.1:8090/v1/chat/completions', json={
    'model':'qwen',
    'messages':[
        {'role':'system','content':'你是专业地质学术翻译。直接输出中文译文。'},
        {'role':'user','content':'The Mississippian MVT deposits of the Appalachian basin.'}
    ],
    'max_tokens':64,'temperature':0.1
}, timeout=15)
dt = (time.time()-t0)*1000
zh = r.json()['choices'][0]['message']['content']
P(f"  GPU 冒烟测试: {zh}", dt_ms=f"{dt:.0f}")

# ══════════════════════════════════════════════════
#  L1+L2 探针: BabelDOC progress_callback
# ══════════════════════════════════════════════════
P("\n=== 准备 PDF ===")
import fitz
pdf = r'D:\PostNews\Translate\uploads\MVT_mesozoic_v2.pdf'
doc = fitz.open(pdf)
n_pages = doc.page_count
P(f"  页数", n_pages=n_pages)
doc.close()

# stage timing accumulator
_stage_times = {}
_stage_t0 = {}
_seg_times = []        # list of (i, dt_ms, chars)
_seg_counter = [0]

def _progress_cb(pct, stage, cur, tot):
    """BabelDOC 各阶段回调"""
    now = time.time()
    if stage not in _stage_t0:
        _stage_t0[stage] = now
        P(f"L1 stage 开始", stage=stage, pct=pct, cur=cur, tot=tot,
          active_calls=len(_seg_times), total_elapsed=f"{now-T0:.1f}s")
    
    # 翻译阶段: 统计段数
    if stage == 'Translate Paragraphs':
        c = _seg_counter[0]
        if cur != c and cur > 0:
            # 说明翻译了新段落
            _seg_counter[0] = cur
            if len(_seg_times) % 20 == 0:  # 每 20 段打一条
                P(f"L2 translate progress", cur=cur, tot=tot,
                  elapsed=f"{now-T0:.1f}s", active_calls=len(_seg_times))

T0 = time.time()  # 全局开始

# ══════════════════════════════════════════════════
#  跑翻译!
# ══════════════════════════════════════════════════
P("\n═════════════════════════════════════════════")
P(f"  开始翻译 MVT_mesozoic_v2.pdf ({n_pages} 页)")
P("═══════════════════════════════════════════════\n")

from run_babeldoc import translate_pdf

t_start = time.time()
try:
    mono, dual = translate_pdf(pdf, r'D:\PostNews\Translate\translated',
        progress_callback=_progress_cb,
        no_dual=True, no_mono=False)
except Exception as e:
    P(f"❌ 翻译崩溃", err=str(e))
    import traceback
    P(traceback.format_exc())
    mono = None

t_end = time.time()
total_dt = t_end - t_start

# ══════════════════════════════════════════════════
#  汇总报告
# ══════════════════════════════════════════════════
P("\n═════════════════════════════════════════════")
P("  📊 汇总报告")
P("═══════════════════════════════════════════════")
P(f"总耗时", total_s=f"{total_dt:.1f}")

# Stage timing
for s, t0s in sorted(_stage_t0.items(), key=lambda x: x[1]):
    dur = time.time() - t0s
    P(f"  stage", name=s, total_s=f"{dur:.1f}", pct=f"{dur/total_dt*100:.0f}%")

# Engine stats
eng = lhe._engine_instance if hasattr(lhe, '_engine_instance') else None
if eng:
    s = eng._stats
    avg_tps = (s['total_tokens']/s['total_ms']*1000) if s['total_ms'] else 0
    avg_call_ms = (s['total_ms']/s['calls']) if s['calls'] else 0
    P(f"  engine", calls=s['calls'], total_tokens=s['total_tokens'],
      avg_tps=f"{avg_tps:.1f}", avg_call_ms=f"{avg_call_ms:.0f}")

# 产物
if mono and os.path.exists(mono):
    kb = os.path.getsize(mono)/1024
    P(f"✅ 译版", path=mono, kb=f"{kb:.1f}")
else:
    P("❌ 无译版产出")

_log_fh.close()
P(f"\n完整日志: {LOG_PATH}")
