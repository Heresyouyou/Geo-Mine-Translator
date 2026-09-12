"""run_babeldoc.py — 翻译入口 (CLI + web_app 共用)

BabelDOC v2 API: TranslationConfig 对象 → hl.translate(config)
签名: translate_pdf(pdf_path, output_dir, progress_callback=None, no_dual=True, no_mono=False)
返回: (mono_path, dual_path)  tuple
"""
import sys, os, time, logging, glob, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ═══ Patch onnxruntime: DML 优先 + CPU fallback (GPU 加速 DocLayout-YOLO, 省 Parse Page Layout 12s) ═══
import onnxruntime as _ort
_real_providers = _ort.get_available_providers()
if 'DmlExecutionProvider' in _real_providers:
    _ort.get_available_providers = lambda: ['DmlExecutionProvider', 'CPUExecutionProvider']
    print(f'[ORT] Patched → DML-first (providers: {_ort.get_available_providers()})')
elif 'CUDAExecutionProvider' in _real_providers:
    # 有 CUDA 但之前 801 错误, 走 CPU fallback
    _ort.get_available_providers = lambda: ['CPUExecutionProvider']
    print('[ORT] Patched → CPU-only (有 CUDA 但曾有 801 错误)')
else:
    print(f'[ORT] keep default providers: {_real_providers}')

# ═══ Patch BabelDOC 字体: 西文 Times New Roman/Arial + 中文 思源宋体/黑体 ═══
# 关键洞察: fontmap.map_in_type() 用 font_id.lower() 有没有 "serif" 来做衬线/无衬线过滤
#   serif=True 要求 font_id 含 "serif"
#   serif=False 要求 font_id 不含 "serif"
# 所以英文字体必须 rename 成带 "Serif" 或不带 (sans)
from babeldoc.assets import embedding_assets_metadata as _fam
import shutil as _shutil, hashlib as _hashlib
_font_cache = Path.home() / '.cache/babeldoc/fonts'
_font_cache.mkdir(parents=True, exist_ok=True)

def _ensure_font(src_ttf, dst_filename, bold, italic, serif):
    """复制字体到缓存 + 注册 metadata, 返回 dst_filename"""
    dst_path = _font_cache / dst_filename
    if not dst_path.exists() and Path(src_ttf).exists():
        _shutil.copy2(src_ttf, dst_path)
        print(f'[FONT] Copied {Path(src_ttf).name} → {dst_filename}')
    if dst_filename not in _fam.EMBEDDING_FONT_METADATA:
        with open(dst_path, 'rb') as _f: _raw = _f.read()
        _fam.EMBEDDING_FONT_METADATA[dst_filename] = {
            'file_name': dst_filename,
            'font_name': Path(src_ttf).stem,
            'ascent': 891, 'descent': -216, 'encoding_length': 1,
            'serif': 1 if serif else 0, 'monospace': 0,
            'bold': 1 if bold else 0, 'italic': 1 if italic else 0,
            'size': len(_raw),
            'sha3_256': _hashlib.sha3_256(_raw).hexdigest(),
        }
        print(f'[FONT] Registered {dst_filename} (bold={bold}, italic={italic}, serif={serif})')
    return dst_filename

# ── 衬线 (serif) 西文字体: Times New Roman ──
_TNR_REG = _ensure_font('C:/Windows/Fonts/times.ttf',   'TimesNewRoman-Serif-Regular.ttf', bold=False, italic=False, serif=True)
_TNR_BLD = _ensure_font('C:/Windows/Fonts/timesbd.ttf',  'TimesNewRoman-Serif-Bold.ttf',    bold=True,  italic=False, serif=True)
_TNR_ITA = _ensure_font('C:/Windows/Fonts/timesi.ttf',  'TimesNewRoman-Serif-Italic.ttf',  bold=False, italic=True,  serif=True)
_TNR_BI  = _ensure_font('C:/Windows/Fonts/timesbi.ttf', 'TimesNewRoman-Serif-BoldItalic.ttf', bold=True, italic=True, serif=True)

# ── 无衬线 (sans) 西文字体: Arial ──
_ARL_REG = _ensure_font('C:/Windows/Fonts/arial.ttf',    'Arial-Sans-Regular.ttf', bold=False, italic=False, serif=False)
_ARL_BLD = _ensure_font('C:/Windows/Fonts/arialbd.ttf',  'Arial-Sans-Bold.ttf',    bold=True,  italic=False, serif=False)

# ── 重排 CN_FONT_FAMILY: 西文字体优先, CJK 字体兜底 ──
# normal: 西文 serif → 中文 serif → 西文 sans → 中文 sans
_fam.CN_FONT_FAMILY['normal'] = [
    _TNR_REG, _TNR_BLD, _TNR_ITA, _TNR_BI,                          # Serif 西文 (优先)
    'SourceHanSerifCN-Regular.ttf', 'SourceHanSerifCN-Bold.ttf',  # 中文宋体 (serif)
    _ARL_REG, _ARL_BLD,                                             # Sans 西文
    'SourceHanSansCN-Regular.ttf', 'SourceHanSansCN-Bold.ttf',    # 中文黑体 (sans)
]
_fam.CN_FONT_FAMILY['fallback'] = [
    _TNR_REG, _TNR_BLD,
    'SourceHanSerifCN-Regular.ttf',
    'GoNotoKurrent-Regular.ttf', 'GoNotoKurrent-Bold.ttf',
]
_fam.CN_FONT_FAMILY['base'] = ['SourceHanSansCN-Regular.ttf']

print(f'[FONT] ✅ CN_FONT_FAMILY patched:')
print(f'       normal: serif西文→宋体→sans西文→黑体 (共 {len(_fam.CN_FONT_FAMILY["normal"])} 个)')
print(f'       fallback: TimesNewRoman → SourceHanSerifCN → GoNoto')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('babeldoc_runner')


def translate_pdf(pdf_path, output_dir=None, doc_layout_model=None,
                  progress_callback=None,
                  no_dual=True, no_mono=False,
                  lang_in='en', lang_out='zh', min_text_length=5):
    """翻译单个 PDF — 返回 (mono_path, dual_path) 或抛异常"""
    _t0 = time.time()
    pdf_path = str(pdf_path)
    if not Path(pdf_path).exists():
        raise FileNotFoundError(pdf_path)
    if output_dir is None:
        output_dir = str(Path(pdf_path).parent)

    logger.info('=== 翻译 1 个 PDF ===')
    logger.info(f'➡️  {pdf_path}')

    import babeldoc.high_level as hl
    from babeldoc.translation_config import TranslationConfig
    from babeldoc.progress_monitor import ProgressMonitor
    from babeldoc.document_il.translator.translator import set_translate_rate_limiter
    from babeldoc.docvision.doclayout import DocLayoutModel
    from babeldoc_translator import GeoBabelTranslator

    # rate_limiter 必须 >= LlamaBatchEngine._GATE (N_PARALLEL=4)
    # 设为 4: BabelDOC 最多开 4 并发, 刚好填满 llama-server 4 slot, 零排队
    # 之前 16 > 8: BabelDOC 开太多线程, 大量 segment 在 LlamaBatchEngine._GATE 前排队雪崩
    set_translate_rate_limiter(9999)  # 关掉 BabelDOC QPS 节流 — 让 LlamaBatchEngine._GATE 全权控制并发

    translator = GeoBabelTranslator(lang_in, lang_out, ignore_cache=True)  # True=每次调LLM, False=命中SQLite持久缓存
    doc_layout = doc_layout_model if doc_layout_model is not None else DocLayoutModel.load_available()

    # ── 桥接 BabelDOC v2 ProgressMonitor → web_app _on_progress(pct, stage, cur, tot) ──
    _cb_log = open('_babel_callback.log', 'w', encoding='utf-8')
    _t0_cb = time.time()
    def _babel_callback(**kwargs):
        """BabelDOC v2 progress_change_callback 桥接到 web_app 的签名"""
        _cb_log.write(f"[{time.time()-_t0_cb:8.3f}s] CALL {kwargs}\n"); _cb_log.flush()
        if not progress_callback:
            return
        # kwargs 里包含 type, stage, stage_progress, stage_current, stage_total, overall_progress
        btype = kwargs.get('type', '')
        stage_name = kwargs.get('stage', '')
        cur = kwargs.get('stage_current', 0)
        tot = kwargs.get('stage_total', 0)
        stage_prog = kwargs.get('stage_progress', 0.0)  # BabelDOC 直接给当前阶段 0-100
        overall = kwargs.get('overall_progress', 0.0)
        pct = int(overall * 100) if overall < 1 else int(overall)

        try:
            if btype in ('progress_start', 'progress_update', 'progress_end'):
                # 统一签名: progress_callback(overall_pct, stage_name, cur, tot, stage_progress)
                progress_callback(pct, stage_name, cur, tot, stage_prog)
            # 忽略 type='stage_summary' (只是阶段权重信息, 不含进度)
        except Exception as e:
            logger.debug(f'progress_callback error: {e}')
            pass

    # ── BabelDOC v2 stages (用官方常量避免版本漂移) ──
    TRANSLATE_STAGES = hl.TRANSLATE_STAGES
    pm = ProgressMonitor(TRANSLATE_STAGES, progress_change_callback=_babel_callback)

    logger.info(f'BabelDOC rate_limiter=4 | 引擎=llama-server HTTP 常驻')

    if progress_callback:
        try:
            progress_callback(14, 'Parse PDF and Create Intermediate Representation', 0, 0)
        except Exception:
            pass

    try:
        # ═══ 翻前清理: 只删当前文件 stem 匹配的旧译版 (并发安全) ═══
        _stem = Path(pdf_path).stem
        _pat_mono = str(Path(output_dir) / f'{_stem}*.zh.mono.pdf')
        _pat_dual = str(Path(output_dir) / f'{_stem}*.zh.dual.pdf')
        _pat_nw_mono = str(Path(output_dir) / f'{_stem}*no_watermark.zh.mono.pdf')
        _pat_nw_dual = str(Path(output_dir) / f'{_stem}*no_watermark.zh.dual.pdf')
        _old_files = glob.glob(_pat_mono) + glob.glob(_pat_dual) + glob.glob(_pat_nw_mono) + glob.glob(_pat_nw_dual)
        for _old in _old_files:
            try:
                os.remove(_old)
                logger.info(f'清理旧译版: {os.path.basename(_old)}')
            except OSError:
                pass
        logger.info(f'[cleanup] 删除了 {len(_old_files)} 个旧译版 (stem={_stem})')

        # 新版 BabelDOC: TranslationConfig 对象 + 直接调 do_translate 注入我们的 pm
        # 注意: hl.translate() 会自己 new ProgressMonitor, 忽略 config.progress_monitor!
        # ═══ BabelDOC 配置优化: 全部显式声明 (避免隐式默认值产生双倍渲染/多阶段) ═══
        # 节省预估: San Albino 30.5min → 26min (-14%, 一次) ; 重翻可叠加
        # 1) watermark_output_mode='no_watermark' : 关 BabelDOC 自带水印 → Save PDF 阶段不双份渲染 (~1min)
        # 2) use_side_by_side_dual=False         : 双语版由 webapp.py 用 PyMuPDF 拼, 不让 BabelDOC 做
        #                                         (省 Typesetting + Generate drawing + Save 双份, 共 ~2min)
        # 3) table_model=None                    : 显式关闭表格 OCR (rapidocr) — 论文 PDF 都是数字版, 不需
        # 4) enhance_compatibility=False         : 已设 False (默认), 跳过兼容增强的字体调整 (~10s)
        # 5) qps=24                              : 翻译限流从 16 → 24, 让 LlamaBatchEngine 排队更平滑
        from babeldoc.translation_config import WatermarkOutputMode
        config = TranslationConfig(
            translator=translator,
            input_file=pdf_path,
            lang_in=lang_in,
            lang_out=lang_out,
            output_dir=output_dir,
            doc_layout_model=doc_layout,
            min_text_length=min_text_length,
            no_dual=no_dual,
            no_mono=no_mono,
            use_side_by_side_dual=False,           # ← 关 BabelDOC dual, webapp.py 自行 PyMuPDF 拼
            skip_scanned_detection=True,            # ← 论文都是数字 PDF (省 0:44)
            table_model=None,                       # ← 显式关 OCR (rapidocr) — 已默认
            watermark_output_mode=WatermarkOutputMode.NoWatermark,  # ← 省 ~1min
            enhance_compatibility=False,            # ← 已设 False, 显式声明
            qps=10,                                    # ← 和 GATE=N_PARALLEL 对齐, 零排队
            # progress_monitor 参数传了也没用, hl.translate 会忽略它
        )

        # 用 with pm: 让 ProgressMonitor 正确触发 finish 回调
        with pm:
            result = hl.do_translate(pm, config)

        _dt = time.time() - _t0
        logger.info(f'完成! 耗时 {_dt:.1f}s')

        if progress_callback:
            try:
                progress_callback(100, '完成!')
            except Exception:
                pass

        # 找产物 (翻前已清理, 再用 stem 验证确保匹配正确文件)
        _stem = Path(pdf_path).stem
        mono = glob.glob(str(Path(output_dir) / f'{_stem}*.zh.mono.pdf'))
        dual = glob.glob(str(Path(output_dir) / f'{_stem}*.zh.dual.pdf'))
        # fallback: 用通用 glob (兼容 BabelDOC 命名变化)
        if not mono:
            mono = glob.glob(str(Path(output_dir) / '*.zh.mono.pdf'))
            logger.warning('stem 匹配 mono 为空, 退而用通用 glob: %s', mono[:1])
        if not dual:
            dual = glob.glob(str(Path(output_dir) / '*.zh.dual.pdf'))

        mono_path = mono[0] if mono else None
        dual_path = dual[0] if dual else None

        if not mono_path and not dual_path:
            raise RuntimeError(f'BabelDOC 未生成译版 PDF, result={result}')

        logger.info(f'✅ mono={mono_path}')
        logger.info(f'✅ dual={dual_path}')

        return mono_path, dual_path

    except Exception as e:
        _dt = time.time() - _t0
        logger.error(f'翻译失败 after {_dt:.1f}s: {e}')
        logger.error(traceback.format_exc())
        raise


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_babeldoc.py <pdf_path>')
        sys.exit(1)
    mono, dual = translate_pdf(sys.argv[1])
    print(f'\n✅ mono={mono}\n✅ dual={dual}')


if __name__ == '__main__':
    main()













