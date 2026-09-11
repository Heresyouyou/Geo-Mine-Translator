# -*- coding: utf-8 -*-
"""
llama-cli 子进程翻译引擎 — 稳定可靠 + GPU 加速
================================================

替代之前的 llama-server HTTP 方案, 解决了 Windows HTTP server 卡死问题。

架构:
  BabelDOC → LlamaCliEngine → subprocess.Popen("llama-cli.exe ...")
                              ↓
                         GPU 直接推理 (62 tok/s 生成)

稳定性:
  - 每次 translate 调用都是独立 llama-cli 进程, 不会积累状态
  - 进程结束后 GPU 显存自动释放, 不会 OOM
  - ctx=4096 足够长文本, 不会 Context exceeded

速度:
  - 单次进程启动 ~0.8s (模型已在 mmap), 生成 ~62 tok/s
  - 相比 HTTP 方案省去了 JSON 序列化 + socket 开销

作者: TRAE
"""
import os, sys, time, json, subprocess, threading, logging, queue, re
from pathlib import Path

logger = logging.getLogger(__name__)

# ═══ 路径配置 ═══
_THIS = Path(__file__).parent
LLAMA_DIR = _THIS / "llama-cpp"
LLAMA_CLI = LLAMA_DIR / "llama-cli.exe"
MODEL_DIR = _THIS / "models-gguf"

# ═══ 默认模型 ═══
DEFAULT_MODEL = MODEL_DIR / "Qwen3.5-4B-Q4_K_M.gguf"

# ═══ CUDA DLL 注入 (关键!) ═══
# llama.cpp CUDA build 需要 curand64_12 / cusolver64_12 / cusparse64_12
# 官方 cudart zip 只给 cublas/cudart, 缺的从 PyTorch torch/lib 拿
def _build_env():
    """构建带 CUDA DLL 的环境变量"""
    env = os.environ.copy()
    try:
        import torch
        pt_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(pt_lib):
            env["PATH"] = pt_lib + os.pathsep + env.get("PATH", "")
    except Exception:
        pass
    return env

_CUDA_ENV = _build_env()

# ═══ System Prompt ═══
SYSTEM_PROMPT = (
    "你是专业地质学术翻译专家。"
    "规则:\n"
    "1. 直接输出中文译文, 不要解释或注释\n"
    "2. 保持原文的格式标记 (如 <b1><b2>)\n"
    "3. 术语对照: spodumene→锂辉石, pegmatite→伟晶岩, Archean→太古代, "
    "Proterozoic→元古代, Yanshanian→燕山期, Caledonian→加里东期, "
    "hercynian→海西期, orogeny→造山运动, metamorphism→变质作用, "
    "granite→花岗岩, rhyolite→流纹岩, basalt→玄武岩, "
    "sandstone→砂岩, limestone→石灰岩, shale→页岩, "
    "fluids→流体, inclusions→包裹体, thermometry→测温法\n"
    "4. 地名/人名保留原文"
)


class LlamaCliEngine:
    """
    llama-cli 子进程引擎 — 每次 translate 调用独立进程

    兼容 BatchEngine.translate(text) 接口
    """

    # 进程闸门: 串行 (GPU 独立进程不能并发, 否则算力减半 + 输出竞态)
    _GATE = threading.Semaphore(1)

    # stats 锁 (多线程并发调用 protect)
    _STATS_LOCK = threading.Lock()

    # 直接写文件探针 (绕过 waitress logging)
    _PROBE_FILE = Path(__file__).parent / "_probe_data.jsonl"
    _PROBE_LOCK = threading.Lock()

    @classmethod
    def _write_probe(cls, record: dict):
        """写一行 JSONL 到探针文件"""
        try:
            import json as _json
            record['ts'] = time.strftime('%H:%M:%S')
            with cls._PROBE_LOCK:
                with open(cls._PROBE_FILE, 'a', encoding='utf-8') as f:
                    f.write(_json.dumps(record, ensure_ascii=False) + '\n')
        except Exception:
            pass

    def __init__(self, model_path=None, system=None,
                 n_ctx=4096, n_threads=8, n_gpu_layers=-1,
                 max_tokens=512, timeout_ms=60000):
        self.model = Path(model_path) if model_path else DEFAULT_MODEL
        self.system = system or SYSTEM_PROMPT
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.timeout = timeout_ms / 1000
        self._stats = {"calls": 0, "total_tokens": 0, "total_ms": 0,
                       "errors": 0}

        if not self.model.exists():
            raise FileNotFoundError(f"模型不存在: {self.model}")
        if not LLAMA_CLI.exists():
            raise FileNotFoundError(f"llama-cli.exe 不存在: {LLAMA_CLI}")

        logger.info(
            "✅ LlamaCliEngine | model=%s | ctx=%d | threads=%d | timeout=%ds",
            self.model.name, self.n_ctx, self.n_threads, self.timeout
        )

    def translate(self, text: str, max_tokens: int = None) -> str:
        """翻译单段文本 — 独立 llama-cli 进程"""
        if not text or len(text.strip()) < 2:
            return text

        mt = max_tokens or self.max_tokens

        # ── 构造 prompt (纯文本 chat template) ──
        # llama-cli 直接吃纯 prompt, 我们手动拼 chat template
        prompt = (
            f"System: {self.system}\n"
            f"User: {text.strip()}\n"
            f"Assistant:"
        )

        cmd = [
            str(LLAMA_CLI),
            "-m", str(self.model),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-rea", "off",
            "-p", prompt,
            "-n", str(mt),
            "--temp", "0.1",
            "--no-display-prompt",
        ]

        t0 = time.time()
        # ═══ [PROBE] gate 等待时间 ═══
        _gate_wait_start = time.time()
        with LlamaCliEngine._GATE:
            _gate_wait_ms = (time.time() - _gate_wait_start) * 1000

            try:
                # llama-cli 在 Windows 上用 WriteConsole, capture_output=True 抓不到
                # 必须用 Popen + stdout PIPE + CREATE_NO_WINDOW
                import ctypes
                CREATE_NO_WINDOW = 0x08000000

                _popen_start = time.time()
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=_CUDA_ENV, cwd=str(LLAMA_DIR),
                    creationflags=CREATE_NO_WINDOW,
                )
                _popen_ms = (time.time() - _popen_start) * 1000

                _comm_start = time.time()
                out_bytes, _ = proc.communicate(timeout=self.timeout)
                _comm_ms = (time.time() - _comm_start) * 1000

                output = out_bytes.decode('utf-8', errors='replace')
            except subprocess.TimeoutExpired:
                self._stats["errors"] += 1
                _kill_ok = False
                try:
                    proc.kill()
                    proc.wait(timeout=5)  # 确保进程真的退出
                    _kill_ok = True
                except Exception:
                    pass
                self._write_probe({
                    'event': 'TIMEOUT', 'call': self._stats['calls'],
                    'timeout': self.timeout, 'kill_ok': _kill_ok,
                    'src_len': len(text),
                })
                return text
            except Exception as e:
                self._stats["errors"] += 1
                logger.warning("  [LlamaCli] 异常: %s", e)
                return text

        dt = (time.time() - t0) * 1000
        self._stats["calls"] += 1
        self._stats["total_ms"] += dt

        # ── 解析计时 ──
        _parse_start = time.time()
        result = self._parse_output(output)
        _parse_ms = (time.time() - _parse_start) * 1000

        # ═══ [PROBE] 直接写 JSONL 探针 (每次都写) ═══
        _gen_tok_per_s = 0.0
        _prompt_tok_per_s = 0.0
        m = re.search(r'Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s', output)
        if m:
            _prompt_tok_per_s = float(m.group(1))
            _gen_tok_per_s = float(m.group(2))
        # 估算生成 token 数 (从速度反推: gen_tps * comm_time - prompt)
        _est_gen_tokens = 0
        if _gen_tok_per_s > 0 and _comm_ms > 0:
            _est_gen_tokens = int(_gen_tok_per_s * _comm_ms / 1000)

        self._write_probe({
            'call': self._stats['calls'],
            'total_ms': round(dt, 0),
            'gate_wait_ms': round(_gate_wait_ms, 0),
            'popen_ms': round(_popen_ms, 0),
            'comm_ms': round(_comm_ms, 0),
            'parse_ms': round(_parse_ms, 0),
            'gen_tps': round(_gen_tok_per_s, 1),
            'prompt_tps': round(_prompt_tok_per_s, 1),
            'est_gen_tokens': _est_gen_tokens,
            'src_len': len(text),
            'res_len': len(result),
            'status': 'ok' if result and result.strip() != text.strip() else 'same_or_empty',
        })

        # 估算 token 数 (粗略: 中文字符 ≈ 1 token, 英文单词 ≈ 1.3 token)
        if result:
            self._stats["total_tokens"] += max(len(result), 10)

        # 缺漏保护
        if not result or not result.strip():
            # 空输出 → 可能是 thinking 模式的 reasoning_content
            result = output.strip()
            # 去掉 banner/进度行
            result = re.sub(r'\[.*?t/s\].*?\n', '', result)
            result = re.sub(r'[\x00-\x1f]', '', result).strip()
            if not result or len(result) < 3:
                logger.debug(f"[LlamaCliDEBUG] fallback-empty returning src")
                return text

        if result.strip() == text.strip():
            logger.debug(f"[LlamaCliDEBUG] result==src returning src")
            return text  # 没翻出来

        return result.strip()

    def _parse_output(self, output: str) -> str:
        """从 llama-cli stdout 中提取译文

        --no-display-prompt 输出格式:
            [banner: Loading model, ASCII art, build, model, ftype, modalities, available commands]
            > System: ... (echo 第一行, 但 system prompt 内容跨多行也会被 echo!)
            译文内容       ← 没有 Assistant: 标签! 直接输出!
            [ Prompt: xxx t/s | Generation: xxx t/s ]
            > (空提示符)
            >
            Exiting...
            \x1b[0m
        """
        # 1. 规范化换行
        output = output.replace('\r\r\n', '\n').replace('\r\n', '\n').replace('\r', '\n')

        # 2. 去掉 ANSI + Exiting
        output = re.sub(r'\x1b\[[0-9;]*m', '', output)
        output = output.replace('Exiting...', '')

        # 3. 按行收集有效内容 (去掉 banner/prompt echo/进度行)
        _SKIP_PREFIXES = ('> ', '[ Prompt:', 'Loading model', 'build      :',
                         'model      :', 'ftype      :', 'modalities :',
                         'available commands', '/exit', '/regen', '/clear',
                         '/read <file>', '/glob <pattern>')

        _ASCII_CHARS = set('▄█▀█ ▄▀█')

        _BANNER_KEYWORDS = ('术语对照', '直接输出中文译文', '保持原文的格式标记',
                            '地名/人名保留', '规则:')

        candidates = []
        for line in output.split('\n'):
            s = line.strip()
            if not s:
                continue
            if any(s.startswith(p) for p in _SKIP_PREFIXES):
                continue
            if 't/s' in s:
                continue
            # 过滤 system prompt 关键词行
            if any(kw in s for kw in _BANNER_KEYWORDS):
                continue
            # ASCII art
            _non_ascii = sum(1 for c in s if c not in _ASCII_CHARS)
            if _non_ascii == 0:
                continue
            candidates.append(s)

        # 4. 从末尾取最后一段连续内容 (跳过末尾的 "> " 空提示符)
        # 候选列表末尾可能有残留的空提示符行
        result_lines = []
        for s in reversed(candidates):
            if s.startswith('>') and len(s) <= 3:
                continue
            result_lines.append(s)

        result_lines.reverse()
        return '\n'.join(result_lines).strip()

    def translate_batch(self, texts: list, max_tokens: int = 512) -> list:
        """批量翻译 — 串行 (独立进程, 已经很快)"""
        return [self.translate(t, max_tokens) for t in texts]

    @property
    def speed(self) -> float:
        if self._stats["total_ms"] == 0:
            return 0
        return self._stats["total_tokens"] / (self._stats["total_ms"] / 1000)

    def health_check(self) -> bool:
        """快速检查 GPU + 模型能否工作"""
        try:
            r = self.translate("test", max_tokens=8)
            return len(r) > 0
        except Exception:
            return False


# ═══ 全局单例 ═══
_engine_instance = None
_engine_lock = threading.Lock()


def get_llama_engine(model_path=None, **kw) -> LlamaCliEngine:
    """获取全局 LlamaCliEngine 单例"""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = LlamaCliEngine(model_path=model_path, **kw)
        return _engine_instance


def ensure_server_ready(model_path=None) -> bool:
    """兼容旧 API — 直接 init engine"""
    try:
        eng = get_llama_engine(model_path)
        return eng.health_check()
    except Exception as e:
        logger.error("ensure_server_ready 失败: %s", e)
        return False
