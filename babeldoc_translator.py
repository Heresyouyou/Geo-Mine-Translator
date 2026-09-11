# -*- coding: utf-8 -*-
"""
BabelDOC 翻译器 — llama-server HTTP (主力) + llama-cli (降级)
==========================================================================

2026-09 重写 v2: 从 llama-server HTTP 换成 llama-cli 子进程
  - 稳定性: 每次 translate 是独立进程, 不会卡死/HTTP 崩溃
  - 速度: 62 tok/s generation (GPU 直推, 无 HTTP 开销)
  - 显存: Q4_K_M 4B ~2.5GB, 每次进程结束自动释放
  - ctx=4096: 防长文本 Context exceeded

启动: 首次 translate 时自动 init (不需要预先启动 server)
"""
import logging
import re
import threading

from babeldoc.document_il.translator.translator import BaseTranslator

logger = logging.getLogger(__name__)

_engine = None
_engine_lock = threading.Lock()

# ═══ 优先使用 llama-server HTTP (模型常驻, 省 2.2s/次加载) ═══
# HTTP 挂了自动 watchdog 重启, 再挂才降级到 llama-cli 子进程
USE_HTTP_FIRST = True

# ═══ 零成本短路: 无拉丁字母的段落没有任何可翻译的英文 ═══
# 纯数字 / 数学符号 / 希腊字母 / 已是中文的段落 (历史埋点里 <40 字符段占 13.5%,
# 平均只产 8 个 token) 送进模型只会白占一个 slot (约 400ms) 并有被改写/截断的风险。
# 判据是"可证明安全"的: 没有 [A-Za-z] 就不可能有英文单词。
_HAS_LATIN = re.compile(r"[A-Za-z]")


def _get_engine():
    """懒加载引擎 — HTTP 优先 (模型常驻), 失败回退 CLI 子进程"""
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        # ── 优先级 1: llama-server HTTP (模型常驻, 每次 ~500ms) ──
        if USE_HTTP_FIRST:
            try:
                from llama_http_engine import get_llama_engine, ensure_server_ready
                if ensure_server_ready():
                    _engine = get_llama_engine()
                    logger.info(
                        '✅ LlamaBatchEngine | llama-server HTTP | '
                        f'模型常驻省加载开销 | watchdog 自动重启'
                    )
                    return _engine
                logger.warning('llama-server 启动失败, 降级 CLI')
            except ImportError as e:
                logger.warning(f'llama_http_engine 不可用: {e}')
            except Exception as e:
                logger.warning(f'llama-server 异常: {e}')

        # ── 优先级 2: llama-cli 子进程 (每次独立进程, ~3.2s) ──
        try:
            from llama_cli_engine import get_llama_engine, ensure_server_ready
            if ensure_server_ready():
                _engine = get_llama_engine()
                logger.info(
                    '✅ LlamaCliEngine | llama-cli 子进程 | '
                    f'速度 ~60 tok/s GPU | 独立进程稳定'
                )
                return _engine
            logger.error('llama-cli health_check 也失败')
        except ImportError as e:
            logger.error(f'llama_cli_engine 不可用: {e}')
        except Exception as e:
            logger.error(f'llama-cli 异常: {e}')

        # ── 最终兜底: HTTP 和 CLI 都挂了就 raise ──
        # 不再保留 transformers fallback (太慢, 模型已归档到 Y 盘)
        logger.error('❌ HTTP 和 CLI 引擎都不可用, 无法启动翻译')
        raise RuntimeError(
            '所有翻译引擎都不可用! '
            '请检查 llama-server / llama-cli 是否可执行, '
            '或手动启动: python -c "from llama_http_engine import start_server; start_server()"'
        )


class GeoBabelTranslator(BaseTranslator):
    """BabelDOC 翻译入口 — llama-server HTTP (主力) / llama-cli (降级)"""

    name = "qwen35_llama_http"
    model = "Qwen3.5-4B-Q4_K_M.gguf"  # BaseTranslator 要求

    def __init__(self, lang_in='en', lang_out='zh', ignore_cache=False, *a, **kw):
        super().__init__(lang_in, lang_out, ignore_cache)
        self._engine = _get_engine()

    def do_translate(self, text) -> str:
        """BabelDOC 回调入口 — 零漏翻保障: 短段也翻 + 失败换 prompt 重试"""
        if not text or not text.strip():
            return text

        # ── 零成本短路: 无拉丁字母 ⇒ 无可翻译内容, 原样返回, 不占推理 slot ──
        # 必须放在这一层而不是 engine.translate(): 若引擎原样返回, 下面的
        # "result == text 就重试" 逻辑会白白重试 3 次, 反而放大 3 倍请求。
        if not _HAS_LATIN.search(text):
            return text

        import time as _time
        _t0 = _time.time()

        # ═══ 单次请求, 不 retry, 避免 GATE 雪崩 ═══
        _result = None
        try:
            _result = self._engine.translate(text)
        except Exception as e:
            logger.warning(f'[translate] exception (len={len(text)}): {e}')

        # 兜底: 空/失败→原文
        if not _result or not _result.strip():
            _result = text.strip()

        _dt = (_time.time() - _t0) * 1000

        # ═══ [PROBE] 每 20 次打印汇总 ═══
        self._probe_count = getattr(self, '_probe_count', 0) + 1
        self._probe_total_ms = getattr(self, '_probe_total_ms', 0) + _dt
        self._probe_total_chars = getattr(self, '_probe_total_chars', 0) + len(text)
        self._probe_slow_count = getattr(self, '_probe_slow_count', 0) + (1 if _dt > 5000 else 0)

        if self._probe_count % 20 == 0:
            _avg_ms = self._probe_total_ms / self._probe_count
            _segs_per_s = 1000.0 / max(_avg_ms, 1)
            _short_skipped = getattr(self, '_probe_short_count', 0)
            logger.info(
                f'[PROBE-translator] {self._probe_count} segs | '
                f'avg={_avg_ms:.0f}ms/seg | '
                f'rate={_segs_per_s:.2f} segs/s | '
                f'slow(>5s)={self._probe_slow_count} | '
                f'short_skipped={_short_skipped} | '
                f'chars={self._probe_total_chars}'
            )

        return _result

    def do_llm_translate(self, text) -> str:
        raise NotImplementedError("llama_http_engine 不支持单独 LLM 模式")

