# -*- coding: utf-8 -*-
"""
llama-server HTTP 引擎 — Qwen3.5-4B-Q4_K_M @ RTX 4060 8GB

架构:
  llama-server (独立进程, CUDA) ← HTTP → 这个模块 ← BabelDOC

修复版 v3 (2026-09-12):
  - N_CTX=16384, N_PARALLEL=4, --kv-unified (全局 KV pool, 不是 per-slot!)
  - ftfy 已去掉 (llama-server 版本不支持), 手动 -c 16384
  - CTX_PER_SLOT = N_CTX = 16384 (unified-KV: 所有 slot 共享)
  - --cache-prompt 复用 SYSTEM_PROMPT prefill
"""
import os, sys, time, json, signal, subprocess, threading, queue, logging, re, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

logging.getLogger("urllib3").setLevel(logging.WARNING)

# ═══ 路径 ═══
_LLAMA_DIR = Path(__file__).parent / "llama-cpp"
LLAMA_SERVER = _LLAMA_DIR / "llama-server.exe"
LLAMA_CLI = _LLAMA_DIR / "llama-cli.exe"
MODEL_DIR = Path(__file__).parent / "models-gguf"

_MODEL_CANDIDATES = [
    MODEL_DIR / "Qwen3.5-4B-Q4_K_M.gguf",
    MODEL_DIR / "Qwen3.5-0.8B-Q4_K_M.gguf",
    MODEL_DIR / "Qwen3.5-4B-Q5_K_M.gguf",
]

# ═══ 网络 ═══
SERVER_URL = "http://127.0.0.1:8090"
SERVER_PORT = 8090
SERVER_PROCESS = None

# ═══ llama-server 配置 (必须与 start_server 参数一致!) ═══
# unified-KV 模式下 -c 是**全局** KV pool size (不是 per-slot!)
N_CTX = 16384           # 全局 pool, 所有 slot 共享
N_PARALLEL = 4          # 并发 slot 数
CTX_PER_SLOT = N_CTX    # unified-KV: 共享这个 pool

# ═══ 翻译 Prompt ═══
SYSTEM_PROMPT = """你是专业地质学术论文翻译。规则:
1. 完整翻译为简体中文, 数字/化学式/同位素比值保持原样
2. 禁止 XML/HTML 标签, 翻译完整不中断

核心术语:
矿物: chalcopyrite→黄铜矿, pyrite→黄铁矿, quartz→石英, feldspar→长石, calcite→方解石, dolomite→白云石, sphalerite→闪锌矿, galena→方铅矿, bornite→斑铜矿
岩石: granite→花岗岩, basalt→玄武岩, diorite→闪长岩, gabbro→辉长岩, peridotite→橄榄岩
矿床: porphyry→斑岩型, hydrothermal→热液, skarn→矽卡岩型, epithermal→浅成低温热液型, VMS→火山成因块状硫化物
蚀变: alteration→蚀变, silicification→硅化, potassic→钾化, phyllic→绢英岩化, argillic→泥化
时代: Archean→太古宙, Proterozoic→元古宙, Mesozoic→中生代, Cenozoic→新生代, Yanshanian→燕山期
术语: isotope→同位素, mineralization→矿化"""

TERM_INJECT_PREFIX = "术语对照(必须采用; 原文中的数字与单位必须照抄, 禁止换算): "

# ═══ TermBank (地质术语动态注入) ═══
class TermBank:
    """地质术语词典 — 运行时匹配文本注入到 prompt"""
    def __init__(self, json_path=None):
        self.terms = {}
        if json_path is None:
            json_path = Path(__file__).parent / "geogpt_terms.json"
        if json_path.exists():
            try:
                import json as _j
                data = _j.loads(open(json_path, encoding='utf-8').read())
                self.terms = data if isinstance(data, dict) else {}
            except: pass

    def match(self, text: str) -> dict:
        """找出文本里出现的术语"""
        hits = {}
        text_lower = text.lower()
        for en, zh in self.terms.items():
            if en.lower() in text_lower:
                hits[en] = zh
        return hits

    def render(self, hits: dict) -> str:
        """渲染为 prompt 注入格式"""
        if not hits: return ""
        return ", ".join(f"{k}→{v}" for k, v in hits.items())

    def __bool__(self):
        return len(self.terms) > 0


# ═══ Server 管理 ═══
def _is_server_alive() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=1.5)
        return r.status_code == 200
    except:
        return False


def _find_model() -> Path:
    for m in _MODEL_CANDIDATES:
        if m.exists():
            return m
    raise FileNotFoundError(f"找不到 GGUF 模型, 搜索: {_MODEL_CANDIDATES}")


def stop_server():
    """停掉所有 server 进程"""
    global SERVER_PROCESS
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    SERVER_PROCESS = None
    time.sleep(1.5)


def wait_ready(timeout=30) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _is_server_alive():
            return True
        time.sleep(1)
    return False


def start_server(model_path=None, port=SERVER_PORT,
                n_ctx=N_CTX, n_threads=N_PARALLEL, n_gpu_layers=-1) -> bool:
    """启动 llama-server — unified-KV + 大 -c 覆盖全局 pool"""
    global SERVER_PROCESS

    if _is_server_alive():
        print(f"  [llama-server] 已在运行 ✅ ({SERVER_URL})")
        return True

    if model_path is None:
        model_path = _find_model()

    import socket
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        s.close()
    except OSError:
        print(f"  [llama-server] 端口 {port} 被占用, 尝试停旧进程...")
        stop_server()
        time.sleep(1)

    # ── CUDA DLL 注入 ──
    env = os.environ.copy()
    try:
        import torch as _torch
        _pytorch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        env["PATH"] = _pytorch_lib + os.pathsep + env.get("PATH", "")
        print(f"  [CUDA] 注入 PyTorch torch/lib → PATH")
    except ImportError:
        pass

    # ── 启动参数 ──
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(model_path),
        "-ngl", str(n_gpu_layers),
        "-c", str(n_ctx),
        "-t", str(n_threads),
        "-b", "2048",
        "-ub", "512",
        "-cb",
        "--parallel", str(N_PARALLEL),
        "--kv-unified",
        "--cache-prompt", "--reasoning", "off",
        "--cache-idle-slots",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]

    print(f"\n{'='*60}")
    print(f"  启动 llama-server (修复版 v3)")
    print(f"  模型: {model_path.name} ({model_path.stat().st_size/1024**3:.2f} GB)")
    print(f"  参数: -c={n_ctx}, parallel={N_PARALLEL}, unified-KV")
    print(f"{'='*60}\n")

    SERVER_PROCESS = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )

    if wait_ready(45):
        print(f"  [llama-server] ✅ 启动成功! ({SERVER_URL})")
        # watchdog
        import atexit
        atexit.register(stop_server)
        return True
    else:
        err = ""
        if SERVER_PROCESS:
            SERVER_PROCESS.kill()
            _, err = SERVER_PROCESS.communicate(timeout=5)
        print(f"  [llama-server] ❌ 启动失败 (exit={SERVER_PROCESS.returncode if SERVER_PROCESS else '?'})")
        if err:
            try: print(f"  输出: {err.decode(errors='replace')[:200]}")
            except: pass
        SERVER_PROCESS = None
        return False


# ═══ Watchdog 线程 ═══
_WD_LOCK = threading.Lock()
_WD_LAST_RESTART = 0

def _watchdog_loop():
    """后台 watchdog: server 挂了就重启"""
    while True:
        time.sleep(30)
        if not _is_server_alive():
            now = time.time()
            if now - _WD_LAST_RESTART < 60:
                print(f"[watchdog] 刚重启过, 跳过")
                continue
            with _WD_LOCK:
                _WD_LAST_RESTART = now
                print(f"[watchdog] 检测到 server 异常, 尝试重启...")
                try:
                    stop_server()
                except: pass
                start_server()


def _start_watchdog():
    t = threading.Thread(target=_watchdog_loop, daemon=True)
    t.start()


# ═══ LlamaBatchEngine (HTTP 客户端) ═══
class LlamaBatchEngine:
    """llama-server HTTP 客户端 — 并发闸门 + watchdog + probe 埋点"""

    _instance = None
    _init_lock = threading.Lock()

    # 并发闸门: 严格不超卖 llama-server --parallel
    _GATE = threading.Semaphore(N_PARALLEL)  # 严格对齐 llama-server --parallel, 防止 HTTP 队列爆炸

    def __init__(self):
        self.terms = TermBank()
        self._stats = {"translate_calls": 0, "gate_waits": 0, "term_segments": 0, "term_hits": 0}
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=16, pool_maxsize=32, max_retries=2
        )
        self._session.mount("http://", adapter)
        self._server_started = False
        # probe 文件
        self._probe_path = Path(__file__).parent / "_probe_data.jsonl"
        # 确保 server 活
        self._ensure_alive()
        _start_watchdog()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_alive(self):
        if _is_server_alive():
            return
        print("[LlamaBatchEngine] server 未就绪, 启动...")
        start_server()

    def _probe(self, **kwargs):
        """写入 probe 埋点"""
        try:
            row = {"t": time.time()}
            row.update(kwargs)
            with open(self._probe_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except: pass

    def translate(self, text: str, max_tokens: int = None, _depth: int = 0) -> str:
        """翻译单段 — 支持巨长段自动切分 + ctx 溢出降级

        Args:
            text: 要翻译的文本
            max_tokens: 最大生成 token (None=自动算)
            _depth: 递归深度 (内部用, 防无限切分)

        Fix v4 (2026-09-12):
          - auto-split 递归不再硬 cap 512, 改为 1500, 解决 2000 chars 学术文本截断问题
          - 总 cap 1024 → 1500 (ctx=16384, src<=5000 时安全)
          - 检测 References 上下文 → 按 (author, year) 条目切, 避免条目被截断
        """
        if not text or len(text.strip()) < 2:
            return text

        # ── Fix: References 条目检测 ──
        _is_refs = "references" in text[:200].lower() or bool(
            re.search(r"\([12]\d{3}\)", text[:500]) and "(" in text[:200]
        )
        _ref_entry_re = re.compile(r'(?=\b[A-Z][a-z]+\s+[A-Z]\b.*?\([12]\d{3}\))')

        # ═══ 巨长段自动切分成段落 / Reference 条目 ═══
        # N_CTX=16384, 生成安全区 = ctx - prompt - 500m
        # src≤5000: prompt ≈ 500 + src/3 → safe completion = 16384 - 500 - src/3 - 500 ≥ 12700
        # 学术文本: 英→中 ≈ 0.7× 原长, token ≈ 1.5/char → 2000 src → 1400 Chinese → 2100 tokens needed
        # 保守给 1500 max_tokens, 够大部分参考文献
        if _depth < 3 and len(text) > 2500:
            # References → 按条目切 (每个 entry ~300-1400 chars)
            if _is_refs:
                _entries = [e.strip() for e in _ref_entry_re.split(text)
                           if len(e.strip()) > 30]
                if len(_entries) > 1:
                    self._stats.setdefault("ref_splits", 0)
                    self._stats["ref_splits"] += 1
                    return "".join(
                        self.translate(e, max_tokens=min(1500, max(128, int(len(e) * 0.8) + 80)),
                                       _depth=_depth + 1)
                        for e in _entries if e.strip()
                    )

            # 普通长段 → 按段落/句子切 (chunk=2000)
            parts = self._split_text(text, max_chunk=2000)
            if len(parts) > 1:
                self._stats.setdefault("auto_splits", 0)
                self._stats["auto_splits"] += 1
                # 递归翻每个子块 — cap 提升到 1500 (之前 512 导致严重截断!)
                return "".join(
                    self.translate(p, max_tokens=min(1500, max(128, int(len(p) * 0.8) + 80)),
                                   _depth=_depth + 1)
                    for p in parts if p.strip()
                )

        self._ensure_alive()

        _src_chars = len(text.strip())

        # ── 术语注入 (v4: 放到 SYSTEM prompt, 不再污染 USER message) ──
        _gloss, _nhit = "", 0
        if self.terms:
            _hits = self.terms.match(text)
            if _hits:
                _nhit = len(_hits)
                # 放到 system prompt 里当指令, 绝对不能放 user message 当正文!
                _gloss = (f"\n\n术语对照(硬约束, 必须采用): {self.terms.render(_hits)}"
                         f"\n翻译时遇到这些术语必须用对应的中文, 禁止输出术语表本身!")
                self._stats["term_hits"] += _nhit
                self._stats["term_segments"] += 1

        # ── 动态 max_tokens (v4: cap=1500, ctx=16384 够放 src≤5000 的) ──
        if max_tokens is None:
            _est_prompt = int(_src_chars * 0.5) + 800 + len(_gloss) // 3
            _safe_cap = CTX_PER_SLOT - _est_prompt - 500
            _calc = int(_src_chars * 0.8) + 80
            max_tokens = min(1500, max(128, min(_calc, max(128, _safe_cap))))

        # ── 构造消息 (术语表移到 system prompt, user message 只放要翻译的文本) ──
        _dyn_system = SYSTEM_PROMPT + _gloss
        if _src_chars < 100:
            _user_msg = (f"Translate this English text to concise Chinese. "
                        f"Keep it short but complete. Output ONLY the Chinese translation:\n"
                        f"{text.strip()}")
        else:
            _user_msg = text.strip()

        payload = {
            "model": "default",
            "messages": [
                {"role": "system", "content": _dyn_system},
                {"role": "user", "content": _user_msg},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }

        # ── 并发闸门 + 发送 ──
        _gate_wait_start = time.time()
        with LlamaBatchEngine._GATE:
            _gate_wait_ms = (time.time() - _gate_wait_start) * 1000

            t0 = time.time()
            try:
                r = self._session.post(
                    f"{SERVER_URL}/v1/chat/completions",
                    json=payload, timeout=180
                )
                comm_ms = (time.time() - t0) * 1000

                if r.status_code != 200:
                    err = r.text[:200]

                    # ═══ Fix 1b: Context size exceeded → 切两半重试 ═══
                    if "Context size has been exceeded" in err and _depth < 3:
                        self._stats.setdefault("ctx_overflow_splits", 0)
                        self._stats["ctx_overflow_splits"] += 1
                        parts = self._split_text(text, max_chunk=len(text) // 2)
                        if len(parts) > 1:
                            return "".join(
                                self.translate(p, max_tokens=1024, _depth=_depth + 1)
                                for p in parts if p.strip()
                            )

                    self._probe(status="http_error", http_status=r.status_code,
                               src_len=_src_chars, gate_wait_ms=_gate_wait_ms, error=err)
                    raise RuntimeError(f"llama-server HTTP {r.status_code}: {err}")

                resp = r.json()
                usage = resp.get("usage", {})
                gen = usage.get("completion_tokens", 0)
                pt = usage.get("prompt_tokens", 0)
                content = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                gen_tps = gen / (comm_ms / 1000) if comm_ms > 0 else 0

                self._probe(status="ok", src_len=_src_chars,
                            prompt_tokens=pt, completion_tokens=gen,
                            comm_ms=round(comm_ms, 0), gate_wait_ms=round(_gate_wait_ms, 0),
                            gen_tps=round(gen_tps, 1))
                self._stats["translate_calls"] += 1
                if _gate_wait_ms > 10:
                    self._stats["gate_waits"] += 1
                return content or text.strip()

            except Exception as e:
                comm_ms = (time.time() - t0) * 1000

                # ═══ Fix 1b 兜底: 任何异常 + 长段 → 切两半再试 ═══
                if _depth < 3 and _src_chars > 1500:
                    self._stats.setdefault("err_fallback_splits", 0)
                    self._stats["err_fallback_splits"] += 1
                    parts = self._split_text(text, max_chunk=len(text) // 2)
                    if len(parts) > 1:
                        return "".join(
                            self.translate(p, max_tokens=1024, _depth=_depth + 1)
                            for p in parts if p.strip()
                        )

                self._probe(status="error", src_len=_src_chars,
                            gate_wait_ms=round(_gate_wait_ms, 0),
                            comm_ms=round(comm_ms, 0), error=str(e)[:200])
                raise

    @staticmethod
    def _split_text(text: str, max_chunk: int = 2000) -> list:
        """按段落/句子切分长文本, 优先尊重段落边界

        策略: \n\n 分段 → 段过长按句号分 → 还过长硬切
        """
        text = text.strip()
        if len(text) <= max_chunk:
            return [text]

        result = []
        # 1) 按 \n\n (段落) 分
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 <= max_chunk:
                buf = (buf + "\n\n" + para).strip()
            else:
                if buf:
                    result.append(buf)
                    buf = ""

                # 段落本身就超长 → 按句子切
                if len(para) > max_chunk:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sent_buf = ""
                    for sent in sentences:
                        if len(sent_buf) + len(sent) + 1 <= max_chunk:
                            sent_buf = (sent_buf + " " + sent).strip()
                        else:
                            if sent_buf:
                                result.append(sent_buf)
                            # 单句超长 → 硬切
                            if len(sent) > max_chunk:
                                for i in range(0, len(sent), max_chunk):
                                    result.append(sent[i:i + max_chunk])
                                sent_buf = ""
                            else:
                                sent_buf = sent
                    if sent_buf:
                        buf = sent_buf
                else:
                    buf = para

        if buf:
            result.append(buf)

        return result

    def average_tps(self) -> float:
        """平均生成速度 (从 probe 数据)"""
        try:
            import statistics
            tps_vals = []
            if self._probe_path.exists():
                for line in open(self._probe_path, encoding="utf-8"):
                    try:
                        r = json.loads(line)
                        if r.get("status") == "ok" and r.get("gen_tps", 0) > 0:
                            tps_vals.append(r["gen_tps"])
                    except: pass
            if tps_vals:
                return statistics.mean(tps_vals)
        except: pass
        return 0.0


# ═══ 全局单例入口 ═══
_engine = None
_engine_lock = threading.Lock()


def get_llama_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = LlamaBatchEngine.get_instance()
    return _engine


def ensure_server_ready(model_path=None) -> bool:
    """确保 server 已启动"""
    if _is_server_alive():
        return True
    print("[ensure_server] llama-server 未就绪, 启动...")
    return start_server(model_path)


# ── CLI 入口 ──
if __name__ == "__main__":
    start_server()









