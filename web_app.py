"""
GeoTranslate Web v4 — 原生 iframe PDF + llama.cpp 本地翻译
启动: py -3.12 web_app.py
打开: http://127.0.0.1:5555
"""
import os, sys, time, json, uuid, threading, logging, subprocess, shutil, glob, math, re
from pathlib import Path
from queue import Queue
from flask import Flask, request, jsonify, send_from_directory, render_template_string, Response

# ── 日志 ──
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('geo')

# ── 目录 ──
BASE = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE / 'uploads'
TRANSLATED_DIR = BASE / 'translated'

# ── 端口 & 进程锁 (防多 Flask 启动冲突) ──
PORT = 5555
PID_FILE = BASE / f'.flask_{PORT}.pid'

def _port_in_use(port):
    """测试端口是否已被监听"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('0.0.0.0', port))
        s.close()
        return False  # bind 成功 = 没人占
    except OSError:
        return True   # bind 失败 = 已被占用

def _pid_alive(pid):
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x100000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return True
    except: pass
    return False

def _acquire_flask_lock():
    """启动前检查 + 拿锁:
    1. 读 PID file → 旧进程还活着? → 拒绝或 kill
    2. 端口被占? → 说明已经有 Flask 在跑 → 拒绝
    返回 True 表示本进程可以启动
    """
    # 1. PID file 记录了旧 Flask
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            if _pid_alive(old_pid):
                # 端口也在监听 → 已经有一个活着的 Flask 在跑
                if _port_in_use(PORT):
                    print(f'[锁] PID file 显示 {old_pid} 还活着 + 端口 {PORT} 已监听 → 拒绝重复启动')
                    return False
                else:
                    # PID file 里的进程还活着但端口没绑 → 旧锁无效
                    print(f'[锁] PID file {old_pid} 活着但端口空闲 → 清理旧锁')
                    PID_FILE.unlink(missing_ok=True)
            else:
                # 进程已死 → 清理旧锁
                print(f'[锁] PID file {old_pid} 已死 → 清理')
                PID_FILE.unlink(missing_ok=True)
        except Exception:
            PID_FILE.unlink(missing_ok=True)

    # 2. 端口直接被占 (无 PID file 或 file 损坏)
    if _port_in_use(PORT):
        print(f'[锁] 端口 {PORT} 已被占用 (无有效 PID file) → 拒绝启动')
        return False

    # 3. 拿锁
    PID_FILE.write_text(str(os.getpid()))
    import atexit
    def _release_lock():
        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
        except: pass
    atexit.register(_release_lock)
    print(f'[锁] ✅ 拿锁成功 PID={os.getpid()} port={PORT}')
    return True

# ── onnxruntime: DML 优先 + CPU fallback (GPU 加速 DocLayout-YOLO) ──
try:
    import onnxruntime as _ort
    _real_providers = _ort.get_available_providers()
    if 'DmlExecutionProvider' in _real_providers:
        _ort.get_available_providers = lambda: ['DmlExecutionProvider', 'CPUExecutionProvider']
    elif 'CUDAExecutionProvider' in _real_providers:
        _ort.get_available_providers = lambda: ['CPUExecutionProvider']  # 避开 CUDA 801 错误
except Exception:
    pass

# ── 预加载全局资源 (只加载一次, 所有翻译复用) ──
# DocLayout-YOLO (PyTorch CPU 推理, 每次加载 ~5s!)
try:
    from babeldoc.docvision.doclayout import DocLayoutModel
    print('[预加载] DocLayout-YOLO...', flush=True)
    DOC_LAYOUT_MODEL = DocLayoutModel.load_available()
    print('[预加载] DocLayout 就绪', flush=True)
except Exception as _e:
    print(f'[预加载] DocLayout 失败: {_e}', flush=True)
    DOC_LAYOUT_MODEL = None

# GeoBabelTranslator (llama-server HTTP, 常驻引擎)
try:
    from run_babeldoc import _get_translator
    _translator_fn = None  # translate_pdf 内部会自己建
except Exception:
    pass

UPLOAD_DIR.mkdir(exist_ok=True)
TRANSLATED_DIR.mkdir(exist_ok=True)

# ── Flask ──
app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

# ── 加载 HTML 模板 ──
INDEX_HTML_PATH = BASE / 'index.html'
if INDEX_HTML_PATH.exists():
    INDEX_HTML = INDEX_HTML_PATH.read_text(encoding='utf-8')
    logger.info('HTML template loaded from index.html (%d bytes)', len(INDEX_HTML))
else:
    INDEX_HTML = '<h1>index.html not found</h1>'
    logger.error('index.html not found, extract from pyc first')

# ── Job 状态 (thread-safe dict) ──
jobs = {}
jobs_lock = threading.Lock()
MAX_JOBS = 12
QUEUE = Queue(maxsize=64)
_worker_started = False

# ── 引擎单例 ──
_engine = None
_engine_lock = threading.Lock()

def get_translator():
    global _engine
    with _engine_lock:
        if _engine is None:
            from llama_http_engine import LlamaBatchEngine
            _engine = LlamaBatchEngine()
        return _engine

# ── 缓存头 (让 PDF 可缓存 1 小时) ──
@app.after_request
def _set_cache(response):
    p = request.path
    if p.endswith('.pdf') or p.endswith('.js') or p.endswith('.css'):
        response.headers['Cache-Control'] = 'public, max-age=3600, must-revalidate'
    else:
        response.headers['Cache-Control'] = 'no-cache'
    return response

# ── 首页 ──
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

# ── 缓存查找: 按文件名 stem 匹配 translated 目录 ──
def _find_cache(name):
    """返回 {exists, mono_url, dual_url, stem} 或 None"""
    if not name: return None
    name = Path(name).name
    stem = Path(name).stem
    # 去掉 hash 后缀: _xxxxxxxx 或 __xxxxxxxx (8 hex)
    stem_clean = re.sub(r'[_]{1,2}[a-f0-9]{8}$', '', stem, flags=re.IGNORECASE)

    # 1. 精确 stem 匹配
    for f in TRANSLATED_DIR.iterdir():
        if f.is_file() and stem in f.stem:
            if f.name.endswith('.zh.mono.pdf'):
                return {
                    'exists': True,
                    'mono_url': f'/translated/{f.name}',
                    'dual_url': None,
                    'stem': stem,
                }

    # 2. 模糊匹配: 去 hash 后的 stem 包含
    if stem_clean != stem:
        for f in TRANSLATED_DIR.iterdir():
            if f.is_file() and stem_clean in f.stem and f.name.endswith('.zh.mono.pdf'):
                return {
                    'exists': True,
                    'mono_url': f'/translated/{f.name}',
                    'dual_url': None,
                    'stem': stem,
                }
    return None

# ── API: health + 缓存检查 ──
@app.route('/api/check', methods=['GET', 'POST'])
def api_check():
    # POST: {name: "xxx.pdf"} → 查缓存
    if request.method == 'POST':
        data = request.get_json(force=True) if request.data else {}
        name = data.get('name', '')
        cache = _find_cache(name)
        if cache:
            return jsonify({
                'exists': True,
                'translated_url': cache['mono_url'],
                'dual_url': cache['dual_url'],
                'stem': cache['stem'],
                'cache_hit': True,
            })
        return jsonify({'exists': False, 'cache_hit': False})

    # GET: llama health check (兼容旧行为)
    try:
        r = __import__('requests').get('http://127.0.0.1:8090/health', timeout=2)
        return jsonify({'ok': True, 'llama': r.json()})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 503

# ── API: upload + 入队 ──
def _validate_pdf(f):
    if not f or not f.filename.lower().endswith('.pdf'):
        return 'only .pdf'
    f.seek(0, 2)
    if f.tell() > 200 * 1024 * 1024:
        return 'too large (>200MB)'
    f.seek(0)
    return None

@app.route('/api/upload', methods=['POST'])
def api_upload():
    f = request.files.get('file')
    err = _validate_pdf(f)
    if err: return jsonify({'error': err}), 400

    fname = f.filename
    stem = Path(fname).stem
    safe = f"{stem}_{uuid.uuid4().hex[:8]}.pdf"
    save_path = UPLOAD_DIR / safe
    f.save(str(save_path))
    original_url = f'/uploads/{safe}'

    # ══ 缓存优先 ══
    cache = _find_cache(fname)
    if cache:
        logger.info('✅ Cache hit: %s → %s', fname, cache['mono_url'])
        return jsonify({
            'status': 'done',
            'original_url': original_url,
            'mono_url': cache['mono_url'],
            'dual_url': cache['dual_url'],
            'name': fname,
            'cache_hit': True,
        })

    # 缓存未命中 → 正常入队
    job_id = uuid.uuid4().hex
    now = time.time()

    with jobs_lock:
        # 清理旧 job
        to_del = [k for k, v in jobs.items() if v.get('status') in ('done','failed') and now - v.get('ended_at',0) > 3600]
        for k in to_del: del jobs[k]

        jobs[job_id] = {
            'job_id': job_id,
            'name': fname,
            'path': str(save_path),
            'status': 'queued',
            'progress': 0,
            'message': '等待中...',
            'created_at': now,
            'started_at': 0,
            'ended_at': 0,
            'elapsed': 0,
            'error': None,
            'original_url': original_url,
            'mono_url': None,
            'dual_url': None,
        }

    QUEUE.put(job_id)
    _ensure_worker()

    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'original_url': original_url,
        'name': fname,
    })

# ── 队列 worker (单线程, 串行翻译) ──
def _ensure_worker():
    global _worker_started
    if _worker_started: return
    _worker_started = True
    t = threading.Thread(target=_queue_worker, daemon=True)
    t.start()
    logger.info('Queue worker started')

def _queue_worker():
    while True:
        try:
            job_id = QUEUE.get(timeout=1)
        except:
            continue

        with jobs_lock:
            j = jobs.get(job_id)
            if not j or j.get('status') not in ('queued','translating'):
                if j: j['status'] = 'cancelled'
                continue
            j['status'] = 'translating'
            j['started_at'] = time.time()
            j['progress'] = 0
            j['message'] = '初始化...'

        try:
            _run_translate(job_id)
        except Exception as e:
            logger.exception('translate failed')
            with jobs_lock:
                j = jobs.get(job_id)
                if j:
                    j['status'] = 'failed'
                    j['error'] = str(e)
                    j['ended_at'] = time.time()
                    j['elapsed'] = j['ended_at'] - j.get('started_at', j['created_at'])

# ── 翻译主入口 ──
def _translated_path(stem, suffix):
    return TRANSLATED_DIR / f'{stem}.zh.{suffix}.pdf'

def _dual_path(stem):
    return TRANSLATED_DIR / f'{stem}.zh.dual.pdf'

def _make_dual_pdf(orig_pdf: Path, mono_pdf: Path, dual_pdf: Path) -> bool:
    """PyMuPDF 逐页左右拼接: 左=原文 右=译文 (A4+A4→宽A3)
    保留所有 link annotations (kind=2 URI + kind=4 内部跳转), 右半坐标偏移 +page_w
    
    比 BabelDOC 自己生成 dual 快得多 (省一轮 Typesetting + Save PDF)
    """
    try:
        import fitz
        
        s1 = fitz.open(str(orig_pdf))
        s2 = fitz.open(str(mono_pdf))
        n = max(len(s1), len(s2))
        if n == 0:
            s1.close(); s2.close()
            return False

        ref = s1[0] if len(s1) > 0 else s2[0]
        page_w = ref.rect.width
        page_h = ref.rect.height

        dst = fitz.open()
        for i in range(n):
            pg = dst.new_page(width=page_w*2, height=page_h)
            
            # 左半: 原文 (坐标不变)
            if i < len(s1):
                pg.show_pdf_page(fitz.Rect(0, 0, page_w, page_h), s1, i)
                for lk in s1[i].get_links():
                    pg.insert_link(lk)
            
            # 右半: 译文 (坐标 +page_w)
            if i < len(s2):
                pg.show_pdf_page(fitz.Rect(page_w, 0, page_w*2, page_h), s2, i)
                for lk in s2[i].get_links():
                    r = lk.get("from")
                    if r:
                        lk["from"] = fitz.Rect(r.x0 + page_w, r.y0, r.x1 + page_w, r.y1)
                    t = lk.get("to")
                    if isinstance(t, fitz.Rect):
                        lk["to"] = fitz.Rect(t.x0 + page_w, t.y0, t.x1 + page_w, t.y1)
                    elif isinstance(t, fitz.Point):
                        lk["to"] = fitz.Point(t.x + page_w, t.y)
                    pg.insert_link(lk)

        dst.save(str(dual_pdf), garbage=4, deflate=True)
        s1.close(); s2.close(); dst.close()
        
        logger.info('✅ PyMuPDF dual: %s → %s (%d pages, %.1fMB)',
                    orig_pdf.name, dual_pdf.name, n, dual_pdf.stat().st_size/1e6)
        return True
    except Exception as e:
        logger.warning('PyMuPDF dual 拼接失败 (非致命): %s', e)
        return False

def _run_translate(job_id):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j: return
        src_path = j['path']
        name = j['name']
        stem = Path(src_path).stem  # ← 用 hash 后的 src_path stem, 才对得上 BabelDOC 产物名

    from run_babeldoc import translate_pdf as babel_translate

    # ── 用户指定 4 大阶段权重: 解析20% 翻译70% 排版8% 输出2% ──
    _GROUP_WEIGHT = {
        'parse':    20.0,
        'translate': 70.0,
        'typeset':   8.0,
        'output':    2.0,
    }
    _STAGE_TO_GROUP = {
        'Parse PDF and Create IR':      'parse',
        'DetectScannedFile':             'parse',
        'Parse Page Layout':             'parse',
        'Parse Table':                   'parse',
        'Parse Paragraphs':              'parse',
        'Parse Formulas and Styles':     'parse',
        'Translate Paragraphs':          'translate',
        'Typesetting':                   'typeset',
        'Add Fonts':                     'typeset',
        'Generate drawing instructions': 'typeset',
        'Subset font':                   'output',
        'Save PDF':                      'output',
    }
    _STAGE_CN = {
        'Parse PDF and Create IR':      '解析PDF结构',
        'DetectScannedFile':             '检测扫描件',
        'Parse Page Layout':             '页面布局识别',
        'Parse Table':                   '表格识别',
        'Parse Paragraphs':              '段落识别',
        'Parse Formulas and Styles':     '公式与样式识别',
        'Translate Paragraphs':          '翻译中',
        'Typesetting':                   '排版',
        'Add Fonts':                     '嵌入字体',
        'Generate drawing instructions': '生成绘图',
        'Subset font':                   '字体子集化',
        'Save PDF':                      '保存PDF',
    }

    def _calc_overall(sp):
        total = 0.0
        for g, gw in _GROUP_WEIGHT.items():
            stgs = [s for s, grp in _STAGE_TO_GROUP.items() if grp == g]
            gp = sum(sp.get(s, 0.0) for s in stgs) / max(len(stgs), 1) / 100.0
            total += gp * gw
        return int(total)

    def _on_progress(overall_pct, stage, cur, tot, stage_prog=None):
        try:
            if not jobs_lock.acquire(timeout=2.0): return
            try:
                j = jobs.get(job_id)
                if not j: return
                if 'stage_progresses' not in j:
                    j['stage_progresses'] = {}
                j['stage_starts'] = j.get('stage_starts', {})
                j['stage_starts'][stage] = True
                # 优先用 BabelDOC 给的 stage_progress (最准!), 兜底自己算
                if stage_prog is not None and stage_prog > 0:
                    j['stage_progresses'][stage] = min(100.0, float(stage_prog))
                elif stage == 'Translate Paragraphs':
                    j['_babel_cur'] = cur
                    j['_babel_tot'] = tot
                    j['stage_progresses'][stage] = min(100.0, 100.0 * cur / max(tot, 1))
                elif tot > 0:
                    j['stage_progresses'][stage] = min(100.0, float(cur) / tot * 100.0)
                j['progress'] = max(j.get('progress', 0), _calc_overall(j['stage_progresses']))
                j['current_stage'] = stage
                cn = _STAGE_CN.get(stage, stage)
                j['message'] = f'{cn} ({cur}/{tot})' if tot > 0 else cn
            finally:
                jobs_lock.release()
        except Exception as e:
            logger.debug('_on_progress err: %s', e)

    _hb_stop = threading.Event()
    _hb_thread = threading.Thread(target=lambda: _hb_stop.wait(), daemon=True)
    _hb_thread.start()

    try:
        logger.info('Translate start: %s', src_path)
        babel_translate(
            str(src_path),
            str(TRANSLATED_DIR),
            progress_callback=_on_progress,
            doc_layout_model=globals().get('DOC_LAYOUT_MODEL'),
        )
        logger.info('Translate done')

        _hb_stop.set()

        mono_path = _translated_path(stem, 'mono')
        dual_path = _dual_path(stem)

        mono_url = f'/translated/{mono_path.name}' if mono_path.exists() else None

        # dual: PyMuPDF 拼接 (不再依赖 BabelDOC 的 no_dual=False)
        dual_url = None
        if mono_path.exists():
            if _make_dual_pdf(Path(src_path), mono_path, dual_path):
                dual_url = f'/translated/{dual_path.name}'

        # 兜底: 找任何译版
        if not mono_url:
            for pat in [f'{stem}*.zh.mono.pdf', f'{stem}*.zh.dual.pdf', f'{stem}*.zh.pdf']:
                found = list(TRANSLATED_DIR.glob(pat))
                if found:
                    mono_path = found[0]
                    mono_url = f'/translated/{mono_path.name}'
                    break

        with jobs_lock:
            j = jobs.get(job_id)
            if j:
                j['status'] = 'done'
                j['progress'] = 100
                j['message'] = '完成'
                j['mono_url'] = mono_url
                j['dual_url'] = dual_url
                j['ended_at'] = time.time()
                j['elapsed'] = j['ended_at'] - j.get('started_at', j['created_at'])

    except Exception as e:
        _hb_stop.set()
        raise

# ── API: status ──
def _queue_position(job_id):
    position = 1  # 先算自己
    for qid in list(QUEUE.queue):
        if qid == job_id: break
        position += 1
    active_count = sum(1 for j in jobs.values() if j.get('status') == 'translating')
    return position + active_count

@app.route('/api/status/<job_id>')
def api_status(job_id):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j: return jsonify({'error': 'not found'}), 404

        data = {k: v for k, v in j.items() if k != 'path'}

        if j['status'] == 'queued':
            data['queue_position'] = _queue_position(job_id)
        else:
            data['queue_position'] = 0

        # elapsed
        if j['started_at'] > 0:
            end = j['ended_at'] or time.time()
            data['elapsed'] = round(end - j['started_at'], 1)

        return jsonify(data)

# ── API: debug ──
@app.route('/api/_debug')
def api_debug():
    with jobs_lock:
        job_data = {}
        for k, v in jobs.items():
            job_data[k] = {kk: vv for kk, vv in v.items() if kk != 'path'}

    # 引擎状态
    eng = None
    try:
        eng = get_translator()
    except: pass

    eng_stats = {}
    if eng:
        try:
            eng_stats = {
                'server_url': eng.server_url if hasattr(eng, 'server_url') else '?',
                'gate_size': eng._GATE._value if hasattr(eng, '_GATE') else '?',
            }
        except: pass

    return jsonify({
        'jobs': job_data,
        'engine': eng_stats,
        'queue_size': QUEUE.qsize(),
    })

# ── 静态文件 ──
@app.route('/uploads/<path:f>')
def serve_upload(f):
    return send_from_directory(str(UPLOAD_DIR), f)

@app.route('/translated/<path:f>')
def serve_output(f):
    return send_from_directory(str(TRANSLATED_DIR), f)

# ── 清理 (后台) ──
def _cleanup():
    now = time.time()
    for d in [UPLOAD_DIR, TRANSLATED_DIR]:
        for p in d.glob('*'):
            if p.is_file() and now - p.stat().st_mtime > 7200:
                try: p.unlink()
                except: pass

threading.Thread(target=lambda: [time.sleep(300) or _cleanup() for _ in iter(int,1)], daemon=True).start()

# ── 启动 ──
if __name__ == '__main__':
    _ensure_worker()
    print('=' * 50)
    print('  GeoTranslate Web v4')
    print(f'  http://127.0.0.1:{PORT}')
    print('=' * 50)

    # ── 三层防冲突: PID file + 端口检查 + debug 强制关闭 ──
    if not _acquire_flask_lock():
        print('❌ 启动被拒绝: 已有 Flask 在运行')
        sys.exit(1)

    try:
        import waitress
        print('[启动] waitress (生产级 WSGI)')
        waitress.serve(app, host='0.0.0.0', port=PORT, threads=8)
    except ImportError:
        print('[启动] Flask 内置 WSGI (建议 pip install waitress)')
        app.run(host='0.0.0.0', port=PORT, threaded=True, debug=False, use_reloader=False)
